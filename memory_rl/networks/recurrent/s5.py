from typing import Any, Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.recurrent import RNNCellBase
from jax.nn.initializers import lecun_normal, normal


def _make_hippo(n: int) -> jnp.ndarray:
    p = jnp.sqrt(1.0 + 2.0 * jnp.arange(n))
    a = p[:, None] * p[None, :]
    a = jnp.tril(a) - jnp.diag(jnp.arange(n))
    return -a


def _make_nplr_hippo(n: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a = _make_hippo(n)
    p = jnp.sqrt(jnp.arange(n) + 0.5)
    b = jnp.sqrt(2.0 * jnp.arange(n) + 1.0)
    return a, p, b


def _make_dplr_hippo(
    n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    a, p, b = _make_nplr_hippo(n)
    s = a + p[:, None] * p[None, :]
    s_diag = jnp.diag(s)
    lambda_real = jnp.mean(s_diag) * jnp.ones_like(s_diag)
    lambda_imag, v = jnp.linalg.eigh(s * (-1j))
    p = v.conj().T @ p
    b_orig = b
    b = v.conj().T @ b
    return lambda_real + 1j * lambda_imag, p, v.conj().T @ b, v, b_orig


def _log_step_initializer(dt_min: float = 0.001, dt_max: float = 0.1) -> Callable:
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)

    return init


def _init_log_steps(key, input_tuple):
    h, dt_min, dt_max = input_tuple
    logs = []
    for _ in range(h):
        key, sk = jax.random.split(key)
        logs.append(_log_step_initializer(dt_min, dt_max)(sk, (1,)))
    return jnp.asarray(logs)


def _init_v_inv_b(
    init_fun: Callable, rng: jax.Array, shape: Tuple[int, int], vinv: jnp.ndarray
) -> jnp.ndarray:
    b = init_fun(rng, shape)
    vinv_b = vinv.astype(jnp.complex64) @ b.astype(jnp.complex64)
    r = vinv_b.real
    i = vinv_b.imag
    return jnp.concatenate([r[..., None], i[..., None]], axis=-1)


def _trunc_standard_normal(key, shape):
    h, p, _ = shape
    cs = []
    for _ in range(h):
        key, sk = jax.random.split(key)
        cs.append(lecun_normal()(sk, (1, p, 2)))
    return jnp.asarray(cs)[:, 0]


def _init_cv(
    init_fun: Callable, rng: jax.Array, shape: Tuple[int, int, int], v: jnp.ndarray
) -> jnp.ndarray:
    c_ = init_fun(rng, shape)
    c = c_[..., 0] + 1j * c_[..., 1]
    cv = c @ v.astype(jnp.complex64)
    r = cv.real
    i = cv.imag
    return jnp.concatenate([r[..., None], i[..., None]], axis=-1)


def _discretize_bilinear(
    lam: jnp.ndarray, b_tilde: jnp.ndarray, delta: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ident = jnp.ones(lam.shape[0], dtype=jnp.complex64)
    bl = 1.0 / (ident - (delta / 2.0) * lam)
    lambda_bar = bl * (ident + (delta / 2.0) * lam)
    b_bar = (bl * delta)[..., None] * b_tilde
    return lambda_bar, b_bar


def _discretize_zoh(
    lam: jnp.ndarray, b_tilde: jnp.ndarray, delta: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ident = jnp.ones(lam.shape[0], dtype=jnp.complex64)
    lambda_bar = jnp.exp(lam * delta)
    b_bar = (1.0 / lam * (lambda_bar - ident))[..., None] * b_tilde
    return lambda_bar, b_bar


class S5Cell(RNNCellBase):
    features: int
    state_size: int
    c_init: str = "trunc_standard_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    conj_sym: bool = False
    clip_eigs: bool = False
    step_rescale: float = 1.0
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    kernel_init: Any = None
    bias_init: Any = None

    @property
    def num_feature_axes(self) -> int:
        return 1

    def setup(self):
        lam0, _, _, v, _ = _make_dplr_hippo(self.state_size)
        self._v = v.astype(jnp.complex64)
        self._vinv = self._v.conj().T
        self._lambda_re_init = lam0.real.astype(self.param_dtype)
        self._lambda_im_init = lam0.imag.astype(self.param_dtype)

    def _discretized_params(self):
        lambda_re = self.param(
            "lambda_re", lambda rng, shape: self._lambda_re_init, (self.state_size,)
        )
        lambda_im = self.param(
            "lambda_im", lambda rng, shape: self._lambda_im_init, (self.state_size,)
        )
        if self.clip_eigs:
            lam = jnp.clip(lambda_re.astype(jnp.float32), a_max=-1e-4).astype(
                jnp.float32
            ) + 1j * lambda_im.astype(jnp.float32)
        else:
            lam = lambda_re.astype(jnp.float32) + 1j * lambda_im.astype(jnp.float32)
        b = self.param(
            "b",
            lambda rng, shape: _init_v_inv_b(
                lecun_normal(), rng, (self.state_size, self.features), self._vinv
            ),
            (self.state_size, self.features, 2),
        )
        b_tilde = b[..., 0].astype(jnp.float32) + 1j * b[..., 1].astype(jnp.float32)
        if self.c_init == "complex_normal":
            c = self.param(
                "c", normal(stddev=0.5**0.5), (self.features, self.state_size, 2)
            )
            c_tilde = c[..., 0].astype(jnp.float32) + 1j * c[..., 1].astype(jnp.float32)
        elif self.c_init == "lecun_normal":
            c = self.param(
                "c",
                lambda rng, shape: _init_cv(
                    lecun_normal(), rng, (self.features, self.state_size, 2), self._v
                ),
                (self.features, self.state_size, 2),
            )
            c_tilde = c[..., 0].astype(jnp.float32) + 1j * c[..., 1].astype(jnp.float32)
        elif self.c_init == "trunc_standard_normal":
            c = self.param(
                "c",
                lambda rng, shape: _init_cv(
                    _trunc_standard_normal,
                    rng,
                    (self.features, self.state_size, 2),
                    self._v,
                ),
                (self.features, self.state_size, 2),
            )
            c_tilde = c[..., 0].astype(jnp.float32) + 1j * c[..., 1].astype(jnp.float32)
        else:
            raise ValueError("invalid c_init")
        d = self.param("d", normal(stddev=1.0), (self.features,))
        log_step = self.param(
            "log_step", _init_log_steps, (self.state_size, self.dt_min, self.dt_max)
        )
        step = self.step_rescale * jnp.exp(log_step[:, 0].astype(jnp.float32))
        if self.discretization == "zoh":
            lambda_bar, b_bar = _discretize_zoh(
                lam.astype(jnp.complex64),
                b_tilde.astype(jnp.complex64),
                step.astype(jnp.complex64),
            )
        elif self.discretization == "bilinear":
            lambda_bar, b_bar = _discretize_bilinear(
                lam.astype(jnp.complex64),
                b_tilde.astype(jnp.complex64),
                step.astype(jnp.complex64),
            )
        else:
            raise ValueError("invalid discretization")
        return lambda_bar, b_bar, c_tilde, d.astype(self.dtype)

    @staticmethod
    def _binop_reset(qi, qj):
        ai, bi, ci = qi
        aj, bj, cj = qj
        cj = cj[..., None]
        aout = (aj * ai) * (1.0 - cj) + aj * cj
        bout = (aj * bi + bj) * (1.0 - cj) + bj * cj
        cout = ci * (1.0 - cj[..., 0]) + cj[..., 0]
        return aout, bout, cout

    @nn.compact
    def __call__(
        self, carry: jnp.ndarray, inputs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        lambda_bar, b_bar, c_tilde, d = self._discretized_params()
        u = inputs.astype(self.dtype)
        bu = jnp.einsum("ph,...h->...p", b_bar, u).astype(jnp.complex64)
        x_next = lambda_bar.astype(jnp.complex64) * carry.astype(jnp.complex64) + bu
        y = jnp.einsum("hp,...p->...h", c_tilde, x_next).real
        if self.conj_sym:
            y = 2.0 * y
        y = y + u * d
        y = y.astype(self.dtype)
        return x_next, y

    @nn.compact
    def apply_parallel(
        self, carry: jnp.ndarray, inputs: jnp.ndarray, resets: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        lambda_bar, b_bar, c_tilde, d = self._discretized_params()
        t, *bshape, h = inputs.shape
        u = inputs.reshape(t, -1, h)
        m = resets.reshape(t, -1).astype(self.dtype)
        x0 = carry.reshape(-1, self.state_size).astype(jnp.complex64)
        a = jnp.broadcast_to(
            lambda_bar[None, None, :], (t, u.shape[1], self.state_size)
        )
        bu = jnp.einsum("ph,tbh->tbp", b_bar, u).astype(jnp.complex64)
        a = jnp.concatenate(
            [jnp.ones((1, u.shape[1], self.state_size), a.dtype), a], axis=0
        )
        bu = jnp.concatenate([x0[None, ...], bu], axis=0)
        m = jnp.concatenate([jnp.zeros((1, u.shape[1]), m.dtype), m], axis=0)
        _, xs, _ = jax.lax.associative_scan(self._binop_reset, (a, bu, m))
        xs = xs[1:]
        y = jnp.einsum("hp,tbp->tbh", c_tilde, xs).real
        if self.conj_sym:
            y = 2.0 * y
        y = y + jnp.einsum("h,tbh->tbh", d, u)
        y = y.reshape((t, *bshape, h)).astype(self.dtype)
        x_t = xs[-1].reshape((*bshape, self.state_size))
        return x_t, y

    def initialize_carry(
        self, rng: jax.Array, input_shape: Tuple[int, ...]
    ) -> jnp.ndarray:
        batch_dims = input_shape[:-1]
        return jnp.zeros(batch_dims + (self.state_size,), dtype=jnp.complex64)
