// Kernels for the Qwen3.5 gated-delta-net block:
//
//  * `unit_lower_inverse_f32` - the chunked rule's triangular inverse.
//  * `causal_conv_{f32,f16,bf16}` - the depthwise causal convolution in front
//    of the rule, fused with its `silu`.
//  * `recurrent_delta_{f32,f16,bf16}` - the per-token form of the rule itself,
//    for short sequences.
//  * `gated_norm_{f32,f16,bf16}` - the gated RMSNorm that follows the rule.
//
// Compilation settings matter for all of them and are set by `metal.rs`:
// arithmetic runs in `MTLMathModeSafe`, so no product, sum or cast is
// contracted or reassociated and every one of them rounds in the tensor's
// dtype. That is what the un-fused candle op sequence does (one kernel per op,
// one rounding per op), and it is what makes these kernels reproduce it.

#include <metal_stdlib>
using namespace metal;

// `unit_lower_inverse_f32` sizes its threadgroup tile for this; keep it in
// sync with `MAX_SIZE` in `metal.rs`. 64 x 64 f32 is 16 KiB, inside the
// 32 KiB of threadgroup memory every Metal GPU advertises.
#define LATE_MAX_SIZE 64

// `recurrent_delta_*` sizes its staged `q`/`k` rows and its private per-column
// state for the largest key/value dimension it serves (keep in sync with
// `MAX_DIM` in `metal.rs`), and holds one reduction slot per SIMD group of a
// threadgroup of at most `LATE_MAX_DIM` threads.
#define LATE_MAX_DIM 128
#define LATE_MAX_SIMD_GROUPS 4

// ---------------------------------------------------------------------------
// `(I + A)^-1` for strictly lower-triangular `A`, `[N, S, S]`, f32.
//
// The chunked gated-delta rule solves `(I + A) X = I` per chunk. `(I + A)^-1`
// is the matrix whose column `j` solves `(I + A) x = e_j`, and that solve is
// the forward substitution
//
//     x[i] = e_j[i] - sum_{k < i} A[i, k] * x[k],   i = 0 .. S - 1.
//
// Since `A` is strictly lower, `x[k] = 0` for `k < j` and `x[j] = 1`, so a
// thread only ever sums `k` in `j .. i` and the result already has a unit
// diagonal and an exact zero above it. Thread `j` writes the whole column `j`
// (zeros included), so every output element is written by exactly one thread
// and the output buffer needs no prior zero fill.
//
// One threadgroup per matrix, one thread per column: `group.y` is the matrix
// index and `column` is the column. The matrix is staged in threadgroup memory
// first, so each element of `A` is fetched from device memory once and the
// substitution reads it back from the on-chip tile.
// ---------------------------------------------------------------------------
kernel void unit_lower_inverse_f32(
    const device float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& s [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]])
{
    threadgroup float tile[LATE_MAX_SIZE * LATE_MAX_SIZE];
    const uint column = local.x;

    const uint side = s;
    const device float* src = a + (size_t)group.y * side * side;
    device float* dst = out + (size_t)group.y * side * side;

    // Cooperative staging of the S x S system, one row per pass; the loop runs
    // exactly `side` times because the threadgroup is `side` threads wide.
    for (uint i = column; i < side * side; i += side) {
        tile[i] = src[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (column >= side) {
        return;
    }

    float x[LATE_MAX_SIZE];
    for (uint i = 0; i < side; i++) {
        float value = 0.0f;
        if (i == column) {
            value = 1.0f;
        } else if (i > column) {
            float sum = 0.0f;
            for (uint k = column; k < i; k++) {
                sum += tile[i * side + k] * x[k];
            }
            value = -sum;
        }
        x[i] = value;
        dst[i * side + column] = value;
    }
}

// ---------------------------------------------------------------------------
// `silu(conv1d(xs))` with left zero padding, cut to the input length.
//
// `xs` is `[B, T, C]` in `T`, `weights` is `[C, K]` in `T`, and the result is
// `[B, T, C]` in `T`. One thread per output element walks the `K` taps; `xs`
// is read through its own strides, so a non-contiguous `[B, T, C]` view needs
// no copy.
//
// This is the fused form of `padded * weight` then `sum + term` for each tap
// (each rounded in `T`), then `silu` in `T` - the arithmetic of candle's
// multiply / add / silu kernels, in the tap order the un-fused path uses.
// ---------------------------------------------------------------------------

// Candle's `usilu`: `x / (1 + exp(-x))` with the intermediate and the result
// in the tensor's dtype.
template <typename T>
METAL_FUNC T conv_silu(T x) {
    return static_cast<T>(x / (1 + exp(-x)));
}

template <typename T>
[[kernel]] void causal_conv_kernel(
    const device T* xs [[buffer(0)]],
    const device T* weights [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant uint& b [[buffer(3)]],
    constant uint& t [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& taps [[buffer(6)]],
    constant size_t& stride_b [[buffer(7)]],
    constant size_t& stride_t [[buffer(8)]],
    constant size_t& stride_c [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if ((size_t)tid >= (size_t)b * t * channels) {
        return;
    }
    const uint c = tid % channels;
    const uint rest = tid / channels;
    const uint ti = rest % t;
    const uint bi = rest / t;

    const device T* row = xs + (size_t)bi * stride_b + (size_t)c * stride_c;
    const device T* w = weights + (size_t)c * taps;

    T acc = static_cast<T>(0);
    for (uint k = 0; k < taps; ++k) {
        // `taps - 1` zeros of left padding, so output `ti` reads the sample at
        // `ti + k - (taps - 1)`; below zero the sample is a padded zero, above
        // `t - 1` it cannot be, because `ti <= t - 1` and `k <= taps - 1`.
        const int pos = (int)ti + (int)k - (int)(taps - 1);
        T sample = static_cast<T>(0);
        if (pos >= 0) {
            sample = row[(size_t)pos * stride_t];
        }
        // Deliberately two statements: the product is rounded to `T` before
        // the sum is rounded to `T` (never contracted into one rounding).
        T term = sample * w[k];
        acc = acc + term;
    }
    out[((size_t)bi * t + ti) * channels + c] = conv_silu(acc);
}

// ---------------------------------------------------------------------------
// The per-token (recurrent) gated delta rule, for short sequences.
//
// The reference (`torch_recurrent_gated_delta_rule`, the per-token form the
// chunked code reproduces) computes, for every batch `b`, value head `h`,
// value column `c` and token `i`:
//
//     q_i   = l2norm(mixed[b, i, kh*dk ..]) / sqrt(dk)     kh = h / (nv / nk)
//     k_i   = l2norm(mixed[b, i, nk*dk + kh*dk ..])
//     decay = exp(g[b, i, h])
//     state *= decay
//     mem   = sum_r state[r][c] * k_i[r]
//     delta = (v_i[c] - mem) * beta[b, i, h]
//     state[r][c] += k_i[r] * delta
//     out   = sum_r state[r][c] * q_i[r]
//
// with `state` starting at zero and `v` the model-dtype block of `mixed` past
// `2 * nk * dk`. `q`/`k`/`v` are read straight out of `mixed` (widened to f32
// once), so the convolution's output needs no scratch copy.
//
// One threadgroup per (batch, value head): `group.x` is the head and `group.y`
// the batch. The threadgroup holds `max(dk, dv)` rounded up to a power of two -
// at least one full SIMD group, so `simd_sum` is defined - and each thread owns
// one value column and keeps that column's whole `dk` state in private memory.
// Every dot product of the recurrence is therefore thread-local; the l2 norms
// are the only cross-thread quantity, staged in threadgroup memory and summed
// with `simd_sum` per SIMD group, then across the groups.
//
// As everywhere in this file each product, sum and decay is rounded on its own:
// the decay of `state[r]` is a separate step before the `mem` dot, and the
// `state` update is a rounded product added to it, exactly like the reference.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void recurrent_delta_kernel(
    const device T* mixed [[buffer(0)]],
    const device float* g [[buffer(1)]],
    const device float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& t [[buffer(4)]],
    constant uint& key_heads [[buffer(5)]],
    constant uint& value_heads [[buffer(6)]],
    constant uint& key_dim [[buffer(7)]],
    constant uint& value_dim [[buffer(8)]],
    constant size_t& stride_b [[buffer(9)]],
    constant size_t& stride_t [[buffer(10)]],
    constant size_t& stride_c [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    const uint tid = local.x;
    const uint h = group.x;
    const uint b = group.y;
    const uint nk = key_heads, nv = value_heads, dk = key_dim, dv = value_dim;
    const uint key_head = h / (nv / nk);
    const uint width = threads.x;
    const uint simd_groups = width / 32u;

    threadgroup float q_row[LATE_MAX_DIM];
    threadgroup float k_row[LATE_MAX_DIM];
    threadgroup float q_partials[LATE_MAX_SIMD_GROUPS];
    threadgroup float k_partials[LATE_MAX_SIMD_GROUPS];

    const device T* batch = mixed + (size_t)b * stride_b;
    const float dk_scale = sqrt(static_cast<float>(dk));

    float state[LATE_MAX_DIM];
    for (uint r = 0; r < dk; ++r) {
        state[r] = 0.0f;
    }

    for (uint i = 0; i < t; ++i) {
        // First barrier: the previous token's readers must be done with the
        // staged rows before this token overwrites them.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const device T* row = batch + (size_t)i * stride_t;
        for (uint r = tid; r < dk; r += width) {
            q_row[r] = static_cast<float>(row[(size_t)(key_head * dk + r) * stride_c]);
            k_row[r] = static_cast<float>(row[(size_t)((nk + key_head) * dk + r) * stride_c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float q_sq = 0.0f;
        float k_sq = 0.0f;
        for (uint r = tid; r < dk; r += width) {
            const float q_r = q_row[r] * q_row[r];
            const float k_r = k_row[r] * k_row[r];
            q_sq = q_sq + q_r;
            k_sq = k_sq + k_r;
        }
        const float q_group = simd_sum(q_sq);
        const float k_group = simd_sum(k_sq);
        if (lane == 0) {
            q_partials[simd] = q_group;
            k_partials[simd] = k_group;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = 0.0f;
        float k_total = 0.0f;
        for (uint s = 0; s < simd_groups; ++s) {
            q_total = q_total + q_partials[s];
            k_total = k_total + k_partials[s];
        }
        const float q_norm = sqrt(q_total + 1e-6f);
        const float k_norm = sqrt(k_total + 1e-6f);
        for (uint r = tid; r < dk; r += width) {
            q_row[r] = q_row[r] / q_norm / dk_scale;
            k_row[r] = k_row[r] / k_norm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < dv) {
            const uint c = tid;
            const float decay = precise::exp(g[((size_t)b * t + i) * nv + h]);
            float mem = 0.0f;
            for (uint r = 0; r < dk; ++r) {
                state[r] = state[r] * decay;
                const float product = state[r] * k_row[r];
                mem = mem + product;
            }
            const float v = static_cast<float>(row[(size_t)(2 * nk * dk + h * dv + c) * stride_c]);
            const float delta = (v - mem) * beta[((size_t)b * t + i) * nv + h];
            for (uint r = 0; r < dk; ++r) {
                const float term = k_row[r] * delta;
                state[r] = state[r] + term;
            }
            float acc = 0.0f;
            for (uint r = 0; r < dk; ++r) {
                const float product = state[r] * q_row[r];
                acc = acc + product;
            }
            out[((size_t)(b * t + i) * nv + h) * dv + c] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// The gated RMSNorm that follows the delta rule, with the exact operation
// sequence of `qwen3_5.rs`'s `GatedDeltaNet::forward` (not candle's fused
// `rms_norm`, which folds the variance reciprocal into the scale):
//
//     variance = sum_r core_f32[r] ** 2 / dv          (mean over the last dim)
//     normed   = core_f32[r] / sqrt(variance + eps)   (f32 division)
//     normed_t = static_cast<T>(normed)               (rounded to the model dtype)
//     scaled_t = normed_t * weight[r]                 (T multiply, rounded)
//     out_t    = static_cast<T>(scaled_t_f32 * z_f32 / (1 + exp(-z_f32)))
//
// `core` and `z` are `[rows, dv]` in the model dtype (contiguous, `dv` <=
// LATE_MAX_DIM), `weight` is `[dv]`, and the result is the model dtype. One
// threadgroup per row and one element per thread: the variance is the only
// cross-thread quantity, reduced in f32 with `simd_sum` and then folded across
// the SIMD groups, and `mean` is candle's `sum * (1 / dv)` rather than a
// division. The dtype roundings sit exactly where the op sequence writes a
// tensor of that dtype: the normalized value, the weight product and the final
// gated value.
// ---------------------------------------------------------------------------
template <typename T>
[[kernel]] void gated_norm_kernel(
    const device T* core [[buffer(0)]],
    const device T* z [[buffer(1)]],
    const device T* weight [[buffer(2)]],
    device T* out [[buffer(3)]],
    constant uint& value_dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint3 threads [[threads_per_threadgroup]])
{
    const uint dv = value_dim;
    const uint tid = local.x;
    const uint width = threads.x;
    const uint simd_groups = width / 32u;
    const size_t row = (size_t)group.x * dv;

    threadgroup float partials[LATE_MAX_SIMD_GROUPS];

    const float value = (tid < dv) ? static_cast<float>(core[row + (size_t)tid]) : 0.0f;
    const float square = value * value;
    const float group_sum = simd_sum(square);
    if (lane == 0) {
        partials[simd] = group_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < simd_groups; ++s) {
        total = total + partials[s];
    }
    const float inv_dv = 1.0f / static_cast<float>(dv);
    const float variance = total * inv_dv;
    const float scale = sqrt(variance + eps);

    if (tid < dv) {
        const T normed = static_cast<T>(value / scale);
        const T scaled = normed * weight[tid];
        const float gated = static_cast<float>(scaled) * conv_silu(static_cast<float>(z[row + (size_t)tid]));
        out[row + (size_t)tid] = static_cast<T>(gated);
    }
}

// Instantiate the templated kernels per dtype; `metal.rs` selects by tensor dtype.
#define init_dtype_kernel(name, func, tname, t) \
    template [[host_name(#name "_" #tname)]] [[kernel]] decltype(func<t>) func<t>;

init_dtype_kernel(causal_conv, causal_conv_kernel, f32, float)
init_dtype_kernel(causal_conv, causal_conv_kernel, f16, half)
init_dtype_kernel(recurrent_delta, recurrent_delta_kernel, f32, float)
init_dtype_kernel(recurrent_delta, recurrent_delta_kernel, f16, half)
init_dtype_kernel(gated_norm, gated_norm_kernel, f32, float)
init_dtype_kernel(gated_norm, gated_norm_kernel, f16, half)
#if defined(__HAVE_BFLOAT__)
init_dtype_kernel(causal_conv, causal_conv_kernel, bf16, bfloat)
init_dtype_kernel(recurrent_delta, recurrent_delta_kernel, bf16, bfloat)
init_dtype_kernel(gated_norm, gated_norm_kernel, bf16, bfloat)
#endif
