"""Tests for RTRL (Real-Time Recurrent Learning) support."""

import jax
import jax.numpy as jnp
import pytest


B, T, F = 2, 4, 8  # batch, time, features


class TestLRULocalJacobian:
    """Test LRUCell.local_jacobian against jax.jacrev of a single step."""

    @pytest.fixture
    def setup(self):
        from memorax.networks.sequence_models import LRUCell
        from memorax.networks.sequence_models.memoroid import Memoroid

        H = 16
        cell = LRUCell(features=F, hidden_dim=H)
        model = Memoroid(cell=cell)

        key = jax.random.key(42)
        inputs = jax.random.normal(key, (B, T, F))
        mask = jnp.zeros((B, T))

        params = model.init(key, inputs, mask)
        return cell, model, params, inputs, mask, key, H

    def test_jacobian_shapes(self, setup):
        cell, model, params, inputs, mask, key, H = setup

        carry = jnp.zeros((B, T, H), dtype=jnp.complex64)
        z = cell.apply({"params": params["params"]["cell"]}, inputs)
        jacobians = cell.apply(
            {"params": params["params"]["cell"]},
            carry, z, inputs,
            method="local_jacobian",
        )

        assert jacobians["nu_log"].shape == (B, T, H)
        assert jacobians["theta_log"].shape == (B, T, H)
        assert jacobians["gamma_log"].shape == (B, T, H)
        assert jacobians["B_real"].shape == (B, T, H, F)
        assert jacobians["B_imag"].shape == (B, T, H, F)

    def test_jacobian_vs_jacrev(self, setup):
        """Compare local_jacobian against jax.jacrev for a single step."""
        cell, model, params, inputs, mask, key, H = setup

        cell_params = params["params"]["cell"]
        x_single = inputs[0, 0]
        carry_single = jnp.zeros((H,), dtype=jnp.complex64)

        def one_step(nu_log, theta_log, gamma_log, B_real, B_imag, carry, x):
            lam = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
            B_norm = (B_real + 1j * B_imag) * jnp.exp(gamma_log)[:, None]
            return lam * carry + B_norm @ x

        ref_jacs = jax.jacrev(one_step, argnums=(0, 1, 2, 3, 4), holomorphic=True)(
            cell_params["nu_log"].astype(jnp.complex64),
            cell_params["theta_log"].astype(jnp.complex64),
            cell_params["gamma_log"].astype(jnp.complex64),
            cell_params["B_real"].astype(jnp.complex64),
            cell_params["B_imag"].astype(jnp.complex64),
            carry_single,
            x_single.astype(jnp.complex64),
        )

        z = cell.apply({"params": cell_params}, inputs)
        h_prev = jnp.zeros((B, T, H), dtype=jnp.complex64)
        our_jacs = cell.apply(
            {"params": cell_params},
            h_prev, z, inputs,
            method="local_jacobian",
        )

        param_names = ["nu_log", "theta_log", "gamma_log", "B_real", "B_imag"]
        for i, name in enumerate(param_names):
            our = our_jacs[name][0, 0]
            ref = ref_jacs[i]
            if ref.ndim == 2 and ref.shape[0] == ref.shape[1]:
                ref = jnp.diag(ref)
            elif ref.ndim == 3 and ref.shape[0] == ref.shape[1]:
                ref = jax.vmap(lambda j: ref[j, j, :])(jnp.arange(ref.shape[0]))
            assert jnp.allclose(our, ref, atol=1e-5), (
                f"Jacobian mismatch for {name}: max diff {jnp.max(jnp.abs(our - ref))}"
            )


class TestRTRLWithLRU:
    """Test RTRL wrapping Memoroid with LRUCell."""

    @pytest.fixture
    def setup(self):
        from memorax.networks.sequence_models import LRUCell
        from memorax.networks.sequence_models.memoroid import Memoroid
        from memorax.networks.sequence_models.rtrl import RTRL

        H = 16
        cell = LRUCell(features=F, hidden_dim=H)
        memoroid = Memoroid(cell=cell)
        rtrl_model = RTRL(sequence_model=memoroid)

        key = jax.random.key(0)
        inputs = jax.random.normal(key, (B, T, F))
        mask = jnp.zeros((B, T))

        params = rtrl_model.init(key, inputs, mask)
        return rtrl_model, memoroid, params, inputs, mask, key, H

    def test_forward_output_matches_memoroid(self, setup):
        """RTRL output y should match Memoroid output for same params."""
        rtrl_model, memoroid, params, inputs, mask, key, H = setup

        (_, rtrl_y) = rtrl_model.apply(params, inputs, mask)

        # Extract inner Memoroid params (nested under "sequence_model")
        mem_params = {"params": params["params"]["sequence_model"]}
        (_, mem_y) = memoroid.apply(mem_params, inputs, mask)

        assert jnp.allclose(rtrl_y, mem_y, atol=1e-5), (
            f"Output mismatch: max diff {jnp.max(jnp.abs(rtrl_y - mem_y))}"
        )

    def test_carry_structure(self, setup):
        """Carry should be ((decay, state), sensitivity_dict)."""
        rtrl_model, _, params, inputs, mask, key, H = setup

        carry, y = rtrl_model.apply(params, inputs, mask)
        inner_carry, sensitivity = carry

        assert len(inner_carry) == 2
        assert inner_carry[0].shape == (B, 1, H)
        assert inner_carry[1].shape == (B, 1, H)

        assert isinstance(sensitivity, dict)
        assert "nu_log" in sensitivity
        assert "theta_log" in sensitivity
        assert "gamma_log" in sensitivity
        assert "B_real" in sensitivity
        assert "B_imag" in sensitivity

    def test_sensitivity_shapes(self, setup):
        rtrl_model, _, params, inputs, mask, key, H = setup

        carry, y = rtrl_model.apply(params, inputs, mask)
        _, sensitivity = carry

        assert sensitivity["nu_log"].shape == (B, 1, H)
        assert sensitivity["theta_log"].shape == (B, 1, H)
        assert sensitivity["gamma_log"].shape == (B, 1, H)
        assert sensitivity["B_real"].shape == (B, 1, H, F)
        assert sensitivity["B_imag"].shape == (B, 1, H, F)

    def test_sensitivity_nonzero(self, setup):
        """After processing a sequence, sensitivities should be nonzero."""
        rtrl_model, _, params, inputs, mask, key, H = setup

        carry, _ = rtrl_model.apply(params, inputs, mask)
        _, sensitivity = carry

        for name, s in sensitivity.items():
            assert jnp.any(s != 0), f"Sensitivity '{name}' is all zeros"

    def test_mask_resets_sensitivity(self, setup):
        """Mask resets should affect sensitivity accumulation."""
        rtrl_model, _, params, inputs, mask, key, H = setup

        all_reset_mask = jnp.ones((B, T))
        carry_reset, _ = rtrl_model.apply(params, inputs, all_reset_mask)
        _, sens_reset = carry_reset

        no_reset_mask = jnp.zeros((B, T))
        carry_no_reset, _ = rtrl_model.apply(params, inputs, no_reset_mask)
        _, sens_no_reset = carry_no_reset

        for name in sens_reset:
            assert not jnp.allclose(sens_reset[name], sens_no_reset[name], atol=1e-6), (
                f"Sensitivity '{name}' should differ with/without resets"
            )

    def test_carry_propagation(self, setup):
        """Carry from chunk 1 feeds correctly into chunk 2."""
        rtrl_model, _, params, inputs, mask, key, H = setup

        chunk1 = inputs[:, :2]
        chunk2 = inputs[:, 2:]
        mask1 = mask[:, :2]
        mask2 = mask[:, 2:]

        carry1, y1 = rtrl_model.apply(params, chunk1, mask1)
        carry2, y2 = rtrl_model.apply(params, chunk2, mask2, carry1)

        carry_full, y_full = rtrl_model.apply(params, inputs, mask)

        y_concat = jnp.concatenate([y1, y2], axis=1)
        assert jnp.allclose(y_concat, y_full, atol=1e-5), (
            f"Chunked output mismatch: max diff {jnp.max(jnp.abs(y_concat - y_full))}"
        )

    def test_jit_compatible(self, setup):
        rtrl_model, _, params, inputs, mask, key, H = setup

        @jax.jit
        def forward(inputs, mask):
            return rtrl_model.apply(params, inputs, mask)

        carry, y = forward(inputs, mask)
        carry2, y2 = forward(inputs, mask)
        assert jnp.allclose(y, y2)


class TestRTRLWithS5:
    """Test RTRL wrapping Memoroid with S5Cell."""

    @pytest.fixture
    def setup(self):
        from memorax.networks.sequence_models import S5Cell
        from memorax.networks.sequence_models.memoroid import Memoroid
        from memorax.networks.sequence_models.rtrl import RTRL

        H = 16
        cell = S5Cell(features=F, state_size=H)
        memoroid = Memoroid(cell=cell)
        rtrl_model = RTRL(sequence_model=memoroid)

        key = jax.random.key(0)
        inputs = jax.random.normal(key, (B, T, F))
        mask = jnp.zeros((B, T))

        params = rtrl_model.init(key, inputs, mask)
        return rtrl_model, memoroid, params, inputs, mask, key, H

    def test_forward_output_matches_memoroid(self, setup):
        rtrl_model, memoroid, params, inputs, mask, key, H = setup

        (_, rtrl_y) = rtrl_model.apply(params, inputs, mask)

        mem_params = {"params": params["params"]["sequence_model"]}
        (_, mem_y) = memoroid.apply(mem_params, inputs, mask)

        assert jnp.allclose(rtrl_y, mem_y, atol=1e-5)

    def test_carry_structure(self, setup):
        rtrl_model, _, params, inputs, mask, key, H = setup

        carry, y = rtrl_model.apply(params, inputs, mask)
        inner_carry, sensitivity = carry

        assert len(inner_carry) == 2
        assert isinstance(sensitivity, dict)
        assert "lambda_real" in sensitivity
        assert "lambda_imag" in sensitivity
        assert "log_step" in sensitivity
        assert "b" in sensitivity

    def test_sensitivity_nonzero(self, setup):
        rtrl_model, _, params, inputs, mask, key, H = setup

        carry, _ = rtrl_model.apply(params, inputs, mask)
        _, sensitivity = carry

        for name, s in sensitivity.items():
            assert jnp.any(s != 0), f"Sensitivity '{name}' is all zeros"

    def test_jit_compatible(self, setup):
        rtrl_model, _, params, inputs, mask, key, H = setup

        @jax.jit
        def forward(inputs, mask):
            return rtrl_model.apply(params, inputs, mask)

        carry, y = forward(inputs, mask)
        carry2, y2 = forward(inputs, mask)
        assert jnp.allclose(y, y2)


class TestRTRLWithUnsupportedModel:
    """Test RTRL wrapping a model that doesn't implement local_jacobian."""

    def test_unsupported_model_raises(self):
        from memorax.networks.sequence_models.rtrl import RTRL
        from memorax.networks.sequence_models.wrappers import SequenceModelWrapper
        from flax import linen as nn

        wrapped = SequenceModelWrapper(network=nn.Dense(features=F))
        model = RTRL(sequence_model=wrapped)

        key = jax.random.key(0)
        inputs = jax.random.normal(key, (B, T, F))
        mask = jnp.zeros((B, T))

        with pytest.raises(AssertionError, match="does not support RTRL"):
            model.init(key, inputs, mask)


class TestPhantomGradient:
    """Test the phantom gradient trick for cross-chunk gradient propagation."""

    @pytest.fixture
    def lru_setup(self):
        from memorax.networks.sequence_models import LRUCell
        from memorax.networks.sequence_models.memoroid import Memoroid
        from memorax.networks.sequence_models.rtrl import RTRL

        H = 16
        cell = LRUCell(features=F, hidden_dim=H)
        memoroid = Memoroid(cell=cell)
        rtrl_model = RTRL(sequence_model=memoroid)

        key = jax.random.key(42)
        inputs = jax.random.normal(key, (B, T, F))
        mask = jnp.zeros((B, T))

        params = rtrl_model.init(key, inputs, mask)
        return rtrl_model, memoroid, params, inputs, mask, key, H

    def test_zero_sensitivity_matches_memoroid(self, lru_setup):
        """With zero initial sensitivity, RTRL grads should match plain Memoroid."""
        rtrl_model, memoroid, params, inputs, mask, key, H = lru_setup

        def loss_rtrl(p):
            _, y = rtrl_model.apply(p, inputs, mask)
            return jnp.sum(y.real)

        def loss_memoroid(p):
            mem_p = {"params": p["params"]["sequence_model"]}
            _, y = memoroid.apply(mem_p, inputs, mask)
            return jnp.sum(y.real)

        grad_rtrl = jax.grad(loss_rtrl)(params)
        grad_mem = jax.grad(loss_memoroid)(params)

        cell_grad_rtrl = grad_rtrl["params"]["sequence_model"]["cell"]
        cell_grad_mem = grad_mem["params"]["sequence_model"]["cell"]

        for name in cell_grad_mem:
            assert jnp.allclose(cell_grad_rtrl[name], cell_grad_mem[name], atol=1e-5), (
                f"Gradient mismatch for {name}: "
                f"max diff {jnp.max(jnp.abs(cell_grad_rtrl[name] - cell_grad_mem[name]))}"
            )

    def test_two_chunk_vs_single_chunk_gradient(self, lru_setup):
        """Two chunks with RTRL should match full BPTT for linear recurrences."""
        rtrl_model, memoroid, params, inputs, mask, key, H = lru_setup

        chunk1, chunk2 = inputs[:, :T // 2], inputs[:, T // 2:]
        mask1, mask2 = mask[:, :T // 2], mask[:, T // 2:]

        def loss_full_bptt(p):
            mem_p = {"params": p["params"]["sequence_model"]}
            _, y = memoroid.apply(mem_p, inputs, mask)
            return jnp.sum(y.real)

        def loss_rtrl_two_chunks(p):
            carry1, y1 = rtrl_model.apply(p, chunk1, mask1)
            carry1 = jax.lax.stop_gradient(carry1)
            _, y2 = rtrl_model.apply(p, chunk2, mask2, carry1)
            return jnp.sum(y1.real) + jnp.sum(y2.real)

        grad_full = jax.grad(loss_full_bptt)(params)
        grad_rtrl = jax.grad(loss_rtrl_two_chunks)(params)

        cell_grad_full = grad_full["params"]["sequence_model"]["cell"]
        cell_grad_rtrl = grad_rtrl["params"]["sequence_model"]["cell"]

        for name in cell_grad_full:
            assert jnp.allclose(cell_grad_rtrl[name], cell_grad_full[name], atol=1e-2), (
                f"Gradient mismatch for {name}: "
                f"max diff {jnp.max(jnp.abs(cell_grad_rtrl[name] - cell_grad_full[name]))}"
            )

    def test_correction_nonzero(self, lru_setup):
        """Nonzero sensitivity should produce different grads from truncated BPTT."""
        rtrl_model, memoroid, params, inputs, mask, key, H = lru_setup

        chunk1, chunk2 = inputs[:, :T // 2], inputs[:, T // 2:]
        mask1, mask2 = mask[:, :T // 2], mask[:, T // 2:]

        def loss_truncated(p):
            mem_p = {"params": p["params"]["sequence_model"]}
            carry1, y1 = memoroid.apply(mem_p, chunk1, mask1)
            carry1 = jax.lax.stop_gradient(carry1)
            _, y2 = memoroid.apply(mem_p, chunk2, mask2, carry1)
            return jnp.sum(y1.real) + jnp.sum(y2.real)

        def loss_rtrl_two_chunks(p):
            carry1, y1 = rtrl_model.apply(p, chunk1, mask1)
            carry1 = jax.lax.stop_gradient(carry1)
            _, y2 = rtrl_model.apply(p, chunk2, mask2, carry1)
            return jnp.sum(y1.real) + jnp.sum(y2.real)

        grad_trunc = jax.grad(loss_truncated)(params)
        grad_rtrl = jax.grad(loss_rtrl_two_chunks)(params)

        cell_grad_trunc = grad_trunc["params"]["sequence_model"]["cell"]
        cell_grad_rtrl = grad_rtrl["params"]["sequence_model"]["cell"]

        any_different = False
        for name in cell_grad_trunc:
            if not jnp.allclose(cell_grad_rtrl[name], cell_grad_trunc[name], atol=1e-6):
                any_different = True
                break

        assert any_different, "RTRL grads should differ from truncated BPTT when sensitivity is nonzero"
