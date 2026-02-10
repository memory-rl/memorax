"""Tests for the linear recurrence Pallas kernel.

Tests kernel correctness, gradients, reset masks, shape handling,
and equivalence with associative_scan for LRU, S5, and Mamba.
"""

import jax
import jax.numpy as jnp
import pytest
from flax.linen import initializers

from memorax.kernels.pallas.linear_recurrence import linear_recurrence


# ---------- Helper: naive Python-style loop reference ----------


def _naive_linear_recurrence(x, a, initial_carry, mask):
    """Reference implementation via explicit Python loop."""
    B, T = x.shape[:2]
    h = initial_carry[:, 0]
    h_list = []
    for t in range(T):
        x_t = x[:, t]
        a_t = a[:, t]
        m_t = mask[:, t]
        # Expand mask to match state dims
        m_exp = m_t
        while m_exp.ndim < h.ndim:
            m_exp = m_exp[..., jnp.newaxis]
        h_new = a_t * h + x_t
        h = jnp.where(m_exp, x_t, h_new)
        h_list.append(h)
    h_seq = jnp.stack(h_list, axis=1)
    h_last = h[:, jnp.newaxis]
    return h_seq, h_last


# ==================== Raw Kernel Tests ====================


class TestKernelCorrectness:
    """Compare kernel output against naive Python loop."""

    def test_basic_real(self):
        key = jax.random.key(0)
        B, T, D = 2, 8, 4
        x = jax.random.normal(key, (B, T, D))
        a = jax.random.uniform(jax.random.key(1), (B, T, D), minval=0.5, maxval=1.0)
        init = jnp.zeros((B, 1, D))
        mask = jnp.zeros((B, T))

        h_seq, h_last = linear_recurrence(x, a, init, mask)
        h_seq_ref, h_last_ref = _naive_linear_recurrence(x, a, init, mask)

        assert jnp.allclose(h_seq, h_seq_ref, atol=1e-5)
        assert jnp.allclose(h_last, h_last_ref, atol=1e-5)

    def test_basic_complex(self):
        key = jax.random.key(2)
        B, T, D = 2, 8, 4
        x = jax.random.normal(key, (B, T, D)) + 1j * jax.random.normal(
            jax.random.key(3), (B, T, D)
        )
        a = 0.9 * jnp.exp(
            1j * jax.random.uniform(jax.random.key(4), (B, T, D), maxval=6.28)
        )
        init = jnp.zeros((B, 1, D), dtype=jnp.complex64)
        mask = jnp.zeros((B, T))

        h_seq, h_last = linear_recurrence(x, a, init, mask)
        h_seq_ref, h_last_ref = _naive_linear_recurrence(x, a, init, mask)

        assert jnp.allclose(h_seq, h_seq_ref, atol=1e-5)
        assert jnp.allclose(h_last, h_last_ref, atol=1e-5)

    def test_with_nonzero_init(self):
        key = jax.random.key(5)
        B, T, D = 3, 6, 8
        x = jax.random.normal(key, (B, T, D))
        a = jax.random.uniform(jax.random.key(6), (B, T, D), minval=0.8, maxval=1.0)
        init = jax.random.normal(jax.random.key(7), (B, 1, D))
        mask = jnp.zeros((B, T))

        h_seq, h_last = linear_recurrence(x, a, init, mask)
        h_seq_ref, h_last_ref = _naive_linear_recurrence(x, a, init, mask)

        assert jnp.allclose(h_seq, h_seq_ref, atol=1e-5)
        assert jnp.allclose(h_last, h_last_ref, atol=1e-5)


class TestResetMask:
    """Verify state resets at mask boundaries."""

    def test_full_reset(self):
        """With mask=1 everywhere, each h[t] should equal x[t]."""
        B, T, D = 2, 4, 3
        x = jax.random.normal(jax.random.key(0), (B, T, D))
        a = jnp.ones((B, T, D)) * 0.9
        init = jnp.ones((B, 1, D))
        mask = jnp.ones((B, T))

        h_seq, _ = linear_recurrence(x, a, init, mask)
        assert jnp.allclose(h_seq, x, atol=1e-6)

    def test_partial_reset(self):
        """Reset at t=2, verify h[2]=x[2] and h[3] uses h[2]."""
        B, T, D = 1, 4, 2
        x = jnp.ones((B, T, D))
        a = jnp.ones((B, T, D)) * 0.5
        init = jnp.zeros((B, 1, D))
        mask = jnp.array([[0.0, 0.0, 1.0, 0.0]])

        h_seq, h_last = linear_recurrence(x, a, init, mask)
        h_ref, _ = _naive_linear_recurrence(x, a, init, mask)

        assert jnp.allclose(h_seq, h_ref, atol=1e-6)
        # At reset: h[2] = x[2] = 1
        assert jnp.allclose(h_seq[0, 2], jnp.ones(D), atol=1e-6)
        # After reset: h[3] = 0.5 * 1 + 1 = 1.5
        assert jnp.allclose(h_seq[0, 3], jnp.ones(D) * 1.5, atol=1e-6)

    def test_reset_blocks_gradient(self):
        """Gradient should not flow through reset boundaries."""
        B, T, D = 1, 4, 2
        a = jnp.ones((B, T, D)) * 0.5
        mask = jnp.array([[0.0, 0.0, 1.0, 0.0]])

        def f(init):
            x = jnp.ones((B, T, D))
            h_seq, _ = linear_recurrence(x, a, init, mask)
            # Sum outputs after the reset
            return jnp.sum(h_seq[:, 2:])

        init = jnp.ones((B, 1, D))
        grad = jax.grad(f)(init)
        # init doesn't affect h[2:] because of reset at t=2
        assert jnp.allclose(grad, jnp.zeros_like(init), atol=1e-6)


class TestShapes:
    """Various batch/seq/dim sizes."""

    @pytest.mark.parametrize("B,T,D", [(1, 1, 1), (4, 16, 32), (2, 3, 7)])
    def test_shapes_real(self, B, T, D):
        x = jax.random.normal(jax.random.key(0), (B, T, D))
        a = jnp.ones((B, T, D)) * 0.9
        init = jnp.zeros((B, 1, D))
        mask = jnp.zeros((B, T))

        h_seq, h_last = linear_recurrence(x, a, init, mask)
        assert h_seq.shape == (B, T, D)
        assert h_last.shape == (B, 1, D)

    def test_multidim_state(self):
        """Mamba-like multi-dim state: (B, T, H, N, D)."""
        B, T, H, N, D = 2, 4, 4, 8, 8
        x = jax.random.normal(jax.random.key(0), (B, T, H, N, D))
        a = jnp.ones((B, T, H, 1, 1)) * 0.9  # Broadcast
        a = jnp.broadcast_to(a, x.shape)
        init = jnp.zeros((B, 1, H, N, D))
        mask = jnp.zeros((B, T))

        h_seq, h_last = linear_recurrence(x, a, init, mask)
        h_ref, _ = _naive_linear_recurrence(x, a, init, mask)
        assert h_seq.shape == (B, T, H, N, D)
        assert jnp.allclose(h_seq, h_ref, atol=1e-5)


class TestGradients:
    """Gradient correctness: compare custom_vjp grad against naive loop grad."""

    @staticmethod
    def _naive_fn(x, a, init, mask):
        h_seq, h_last = _naive_linear_recurrence(x, a, init, mask)
        return jnp.sum(h_seq.real) + jnp.sum(h_last.real)

    @staticmethod
    def _kernel_fn(x, a, init, mask):
        h_seq, h_last = linear_recurrence(x, a, init, mask)
        return jnp.sum(h_seq.real) + jnp.sum(h_last.real)

    def test_grad_x_real(self):
        B, T, D = 1, 4, 3
        x = jax.random.normal(jax.random.key(0), (B, T, D))
        a = jax.random.uniform(jax.random.key(1), (B, T, D), minval=0.5, maxval=0.99)
        init = jax.random.normal(jax.random.key(2), (B, 1, D)) * 0.1
        mask = jnp.zeros((B, T))

        grad_kernel = jax.grad(self._kernel_fn, argnums=0)(x, a, init, mask)
        grad_naive = jax.grad(self._naive_fn, argnums=0)(x, a, init, mask)
        assert jnp.allclose(grad_kernel, grad_naive, atol=1e-4)

    def test_grad_a_real(self):
        B, T, D = 1, 4, 3
        x = jax.random.normal(jax.random.key(0), (B, T, D))
        a = jax.random.uniform(jax.random.key(1), (B, T, D), minval=0.5, maxval=0.99)
        init = jax.random.normal(jax.random.key(2), (B, 1, D)) * 0.1
        mask = jnp.zeros((B, T))

        grad_kernel = jax.grad(self._kernel_fn, argnums=1)(x, a, init, mask)
        grad_naive = jax.grad(self._naive_fn, argnums=1)(x, a, init, mask)
        assert jnp.allclose(grad_kernel, grad_naive, atol=1e-4)

    def test_grad_init_real(self):
        B, T, D = 1, 4, 3
        x = jax.random.normal(jax.random.key(0), (B, T, D))
        a = jax.random.uniform(jax.random.key(1), (B, T, D), minval=0.5, maxval=0.99)
        init = jax.random.normal(jax.random.key(2), (B, 1, D)) * 0.1
        mask = jnp.zeros((B, T))

        grad_kernel = jax.grad(self._kernel_fn, argnums=2)(x, a, init, mask)
        grad_naive = jax.grad(self._naive_fn, argnums=2)(x, a, init, mask)
        assert jnp.allclose(grad_kernel, grad_naive, atol=1e-4)

    def test_grad_complex(self):
        B, T, D = 1, 4, 3
        key = jax.random.key(10)
        x = (
            jax.random.normal(key, (B, T, D))
            + 1j * jax.random.normal(jax.random.key(11), (B, T, D))
        ).astype(jnp.complex64)
        a = (
            0.9
            * jnp.exp(
                1j * jax.random.uniform(jax.random.key(12), (B, T, D), maxval=6.28)
            )
        ).astype(jnp.complex64)
        init = jnp.zeros((B, 1, D), dtype=jnp.complex64)
        mask = jnp.zeros((B, T))

        for argnums in [0, 1, 2]:
            grad_kernel = jax.grad(self._kernel_fn, argnums=argnums)(
                x, a, init, mask
            )
            grad_naive = jax.grad(self._naive_fn, argnums=argnums)(x, a, init, mask)
            assert jnp.allclose(grad_kernel, grad_naive, atol=1e-3), (
                f"Complex grad mismatch for argnums={argnums}"
            )

    def test_grad_with_mask(self):
        B, T, D = 1, 6, 2
        x = jax.random.normal(jax.random.key(0), (B, T, D))
        a = jax.random.uniform(jax.random.key(1), (B, T, D), minval=0.5, maxval=0.99)
        init = jax.random.normal(jax.random.key(2), (B, 1, D)) * 0.1
        mask = jnp.array([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])

        for argnums in [0, 1, 2]:
            grad_kernel = jax.grad(self._kernel_fn, argnums=argnums)(
                x, a, init, mask
            )
            grad_naive = jax.grad(self._naive_fn, argnums=argnums)(x, a, init, mask)
            assert jnp.allclose(grad_kernel, grad_naive, atol=1e-4), (
                f"Masked grad mismatch for argnums={argnums}"
            )


class TestJit:
    """JIT compilation."""

    def test_jit(self):
        B, T, D = 2, 8, 4
        x = jax.random.normal(jax.random.key(0), (B, T, D))
        a = jnp.ones((B, T, D)) * 0.9
        init = jnp.zeros((B, 1, D))
        mask = jnp.zeros((B, T))

        h1, l1 = jax.jit(linear_recurrence)(x, a, init, mask)
        h2, l2 = linear_recurrence(x, a, init, mask)
        assert jnp.allclose(h1, h2, atol=1e-6)
        assert jnp.allclose(l1, l2, atol=1e-6)


# ==================== Equivalence with Associative Scan ====================


class TestEquivalenceWithAssociativeScan:
    """Verify Memoroid(scan_fn=linear_recurrence) matches default associative_scan."""

    def _run_both_scans(self, cell_cls, cell_kwargs, features):
        from memorax.kernels import linear_recurrence as lr_scan
        from memorax.networks.sequence_models.memoroid import (
            Memoroid,
            associative_scan,
        )

        key = jax.random.key(42)
        B, T = 2, 8
        inputs = jax.random.normal(key, (B, T, features))
        mask = jnp.zeros((B, T))

        cell = cell_cls(**cell_kwargs)

        model_assoc = Memoroid(cell=cell, scan_fn=associative_scan)
        model_lr = Memoroid(cell=cell, scan_fn=lr_scan)

        params = model_assoc.init(jax.random.key(0), inputs, mask)

        carry_a, out_a = model_assoc.apply(params, inputs, mask)
        carry_l, out_l = model_lr.apply(params, inputs, mask)

        return carry_a, out_a, carry_l, out_l

    def test_lru_equivalence(self):
        from memorax.networks.sequence_models import LRUCell

        carry_a, out_a, carry_l, out_l = self._run_both_scans(
            LRUCell, dict(features=16, hidden_dim=32), features=16
        )
        assert jnp.allclose(out_a, out_l, atol=1e-4), (
            f"LRU output mismatch: max diff={jnp.max(jnp.abs(out_a - out_l))}"
        )

    def test_s5_equivalence(self):
        from memorax.networks.sequence_models import S5Cell

        carry_a, out_a, carry_l, out_l = self._run_both_scans(
            S5Cell, dict(features=16, state_size=32), features=16
        )
        assert jnp.allclose(out_a, out_l, atol=1e-4), (
            f"S5 output mismatch: max diff={jnp.max(jnp.abs(out_a - out_l))}"
        )

    def test_mamba_equivalence(self):
        from memorax.networks.sequence_models import MambaCell

        carry_a, out_a, carry_l, out_l = self._run_both_scans(
            MambaCell,
            dict(features=16, num_heads=4, head_dim=8, state_dim=16),
            features=16,
        )
        assert jnp.allclose(out_a, out_l, atol=1e-4), (
            f"Mamba output mismatch: max diff={jnp.max(jnp.abs(out_a - out_l))}"
        )

    def test_lru_equivalence_with_mask(self):
        from memorax.kernels import linear_recurrence as lr_scan
        from memorax.networks.sequence_models import LRUCell
        from memorax.networks.sequence_models.memoroid import (
            Memoroid,
            associative_scan,
        )

        key = jax.random.key(42)
        B, T, features = 2, 8, 16
        inputs = jax.random.normal(key, (B, T, features))
        mask = jnp.array([[0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0, 0]]).astype(
            jnp.float32
        )

        cell = LRUCell(features=features, hidden_dim=32)
        model_a = Memoroid(cell=cell, scan_fn=associative_scan)
        model_l = Memoroid(cell=cell, scan_fn=lr_scan)

        params = model_a.init(jax.random.key(0), inputs, mask)
        _, out_a = model_a.apply(params, inputs, mask)
        _, out_l = model_l.apply(params, inputs, mask)

        assert jnp.allclose(out_a, out_l, atol=1e-4), (
            f"LRU masked output mismatch: max diff={jnp.max(jnp.abs(out_a - out_l))}"
        )
