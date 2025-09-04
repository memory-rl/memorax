import jax
import jax.numpy as jnp
from jax import lax

def get_num_episodes(transitions):
    return transitions.done.sum()

def get_episode_lengths(transitions):
    done = transitions.done

    def step(carry_len, done_t):
        curr_len = carry_len + 1
        out = jnp.where(done_t, curr_len, jnp.zeros_like(curr_len))
        next_len = jnp.where(done_t, jnp.zeros_like(curr_len), curr_len)
        return next_len, out

    init_len = jnp.zeros_like(done[0], dtype=jnp.int32)
    _, lengths_at_done = lax.scan(step, init_len, done)

    return lengths_at_done[transitions.done]

def get_episodic_returns(transitions):
    r = transitions.reward
    done = transitions.done

    def step(carry_sum, inp):
        r_t, d_t = inp
        s = carry_sum + r_t
        out = jnp.where(d_t, s, jnp.zeros_like(s))
        next_s = jnp.where(d_t, jnp.zeros_like(s), s)
        return next_s, out

    init_sum = jnp.zeros_like(r[0])
    _, returns_at_done = lax.scan(step, init_sum, (r, done))
    return returns_at_done[transitions.done]

def get_discounted_episodic_returns(transitions, gamma: float):
    r = transitions.reward
    done = transitions.done
    gamma = jnp.asarray(gamma, dtype=r.dtype)

    def step(carry, inp):
        gsum, pow_ = carry
        r_t, d_t = inp
        gsum_new = gsum + r_t * pow_
        out = jnp.where(d_t, gsum_new, jnp.zeros_like(gsum_new))
        gsum_next = jnp.where(d_t, jnp.zeros_like(gsum_new), gsum_new)
        pow_next = jnp.where(d_t, jnp.ones_like(pow_), pow_ * gamma)
        return (gsum_next, pow_next), out

    init = (jnp.zeros_like(r[0]), jnp.ones_like(r[0]))
    _, disc_returns_at_done = lax.scan(step, init, (r, done))
    return disc_returns_at_done[transitions.done]

def mean_and_ci95(x, axis=0):
    """Mean and normal-approx 95% CI (fast). For bootstrap, compute offline."""
    mean = jnp.mean(x, axis=axis)
    std = jnp.std(x, axis=axis, ddof=1)
    n = x.shape[axis]
    half = 1.96 * std / jnp.sqrt(n)
    return mean, jnp.stack([mean - half, mean + half], axis=0)

def success_rate_from_returns(episode_returns, threshold=0.0):
    """Fraction of episodes with return > threshold."""
    return jnp.mean((episode_returns > threshold).astype(jnp.float32))

def normalized_return(episode_returns_H, episode_returns_Href, eps=1e-8, paired=True):
    """
    If paired=True: expects equal-length vectors (per-seed pairing): mean(R_H / R_Href).
    Else: ratio of means.
    """
    if paired:
        ratio = episode_returns_H / jnp.maximum(episode_returns_Href, eps)
        return jnp.mean(ratio)
    else:
        num = jnp.mean(episode_returns_H)
        den = jnp.maximum(jnp.mean(episode_returns_Href), eps)
        return num / den

def trapezoid_auc(x_steps, y_returns, T_max=None, normalize=False):
    """
    x_steps: [K], strictly increasing eval steps.
    y_returns: [K], mean eval returns at those steps.
    If normalize=True, divide by (T_max - x_steps[0]) where T_max defaults to last x.
    """
    area = jnp.trapz(y_returns, x_steps)
    if normalize:
        if T_max is None:
            T_max = x_steps[-1]
        denom = jnp.maximum((T_max - x_steps[0]), 1e-8)
        return area / denom
    return area

def tbptt_penalty(return_tbptt, return_full, eps=1e-8):
    """1 - R(TBPTT)/R(full-BPTT); scalar or vectorized."""
    return 1.0 - (return_tbptt / jnp.maximum(return_full, eps))

def horizon_breakpoint(horizons, normalized_returns, threshold=0.8):
    """
    Smallest horizon H where normalized_return < threshold; NaN if none.
    horizons: [M] ints, normalized_returns: [M] floats (same order).
    """
    cond = normalized_returns < threshold
    idx = jnp.argmax(cond)  # returns 0 if all False; handle separately
    has = jnp.any(cond)
    return jnp.where(has, horizons[idx], jnp.nan)

def generalization_gap(train_returns_vec, test_return_scalar):
    """
    Gap = R_test - mean(R_train_set). Positive means good extrapolation.
    """
    return test_return_scalar - jnp.mean(train_returns_vec)

# ---------- memory capacity probes (DMS / delayed-match) ----------

def lag_probe_accuracy_linear(hidden_by_lag, labels, W, b=None):
    """
    hidden_by_lag: [L, N, D] hidden states captured at different lags.
    labels:        [N] int32 in [0, C).
    W:             [D, C] linear probe weights; b: [C] or None.
    Returns: acc_per_lag [L]
    """
    if b is None:
        b = jnp.zeros((W.shape[-1],), dtype=hidden_by_lag.dtype)
    logits = jnp.einsum('lnd,dc->lnc', hidden_by_lag, W) + b  # [L,N,C]
    preds = jnp.argmax(logits, axis=-1)                       # [L,N]
    acc = jnp.mean((preds == labels[None, :]).astype(jnp.float32), axis=-1)
    return acc

def lag_at_threshold(acc_per_lag, lags, threshold=0.8):
    """
    First lag L where acc >= threshold; NaN if none.
    acc_per_lag: [L], lags: [L] increasing.
    """
    cond = acc_per_lag >= threshold
    idx = jnp.argmax(cond)
    has = jnp.any(cond)
    return jnp.where(has, lags[idx], jnp.nan)

def classification_mi_bits_from_logits(logits, labels, eps=1e-8):
    """
    Estimate I(Y;Z) = H(Y) - H(Y|Z) using cross-entropy on predicted posteriors.
    logits: [N,C], labels: [N] int32. Returns MI in bits.
    """
    probs = jax.nn.softmax(logits, axis=-1) + eps
    n = labels.shape[0]
    # H(Y): from empirical class frequency
    num_classes = probs.shape[-1]
    counts = jnp.bincount(labels, length=num_classes).astype(jnp.float32)
    py = counts / jnp.maximum(jnp.sum(counts), 1.0)
    H_y = -jnp.sum(jnp.where(py > 0, py * jnp.log(py), 0.0))
    # H(Y|Z): average cross-entropy
    label_onehot = jax.nn.one_hot(labels, num_classes)
    CE = -jnp.sum(label_onehot * jnp.log(probs)) / n
    I_nats = H_y - CE
    return I_nats / jnp.log(2.0)

def classification_mi_bits_per_lag(logits_by_lag, labels):
    """
    logits_by_lag: [L,N,C]; returns MI_bits per lag: [L].
    """
    def mi_l(logits):
        return classification_mi_bits_from_logits(logits, labels)
    return jax.vmap(mi_l, in_axes=0)(logits_by_lag)

# ---------- attention recall (transformers) ----------

def attention_recall_rates(attn_at_query, cue_pos, k=1, reduce_heads=True):
    """
    attn_at_query: [N, Hh, T] attention over source positions at the query timestep.
    cue_pos:       [N] int32 indices of the cue timestep.
    k:             top-k.
    reduce_heads:  if True, average over heads first; else treat each head separately.
    Returns: (top1_rate, topk_rate) scalars averaged over batch (and heads if reduced).
    """
    if reduce_heads:
        # average weights over heads, then take topk over T
        w = jnp.mean(attn_at_query, axis=1)  # [N,T]
        topk_idx = jnp.argsort(w, axis=-1)[:, -k:]  # [N,k]
        top1 = (jnp.argmax(w, axis=-1) == cue_pos).astype(jnp.float32).mean()
        # check if cue_pos in top-k
        cue_pos_exp = cue_pos[:, None]
        in_topk = (topk_idx == cue_pos_exp).any(axis=-1).astype(jnp.float32).mean()
        return top1, in_topk
    else:
        # per-head evaluation then average
        def per_head(head_w):
            topk_idx = jnp.argsort(head_w, axis=-1)[:, -k:]  # [N,k]
            top1 = (jnp.argmax(head_w, axis=-1) == cue_pos).astype(jnp.float32).mean()
            cue_pos_exp = cue_pos[:, None]
            in_topk = (topk_idx == cue_pos_exp).any(axis=-1).astype(jnp.float32).mean()
            return top1, in_topk
        top1_h, topk_h = jax.vmap(per_head, in_axes=1)(attn_at_query)
        return top1_h.mean(), topk_h.mean()

# ---------- episode table helpers ----------

def build_episode_table(transitions):
    """
    Returns a dict of per-episode arrays for easy logging/bootstrapping.
    Keys: 'return', 'discounted_return', 'length'
    """
    returns = get_episodic_returns(transitions)
    lengths = get_episode_lengths(transitions)
    out = {
        "return": returns,
        "length": lengths,
    }
    return out

# ---------- convenience wrappers for logging ----------

def eval_summary_from_transitions(transitions, ref_mean_return=None, success_thresh=0.0):
    """
    Compute eval summary dict from a rollout buffer (single eval pass).
    If ref_mean_return provided, also returns normalized_return.
    """
    ep = build_episode_table(transitions)
    mean_ret, ci = mean_and_ci95(ep["return"])
    succ = success_rate_from_returns(ep["return"], threshold=success_thresh)
    out = {
        "eval/return_mean": mean_ret,
        "eval/return_ci95_low": ci[0],
        "eval/return_ci95_high": ci[1],
        "eval/success_rate": succ,
        "eval/num_episodes": ep["return"].shape[0],
        "eval/episode_length_mean": jnp.mean(ep["length"]),
    }
    if ref_mean_return is not None:
        out["eval/normalized_return"] = mean_ret / jnp.maximum(ref_mean_return, 1e-8)
        out["eval/return_mean_ref"] = ref_mean_return
    return out
