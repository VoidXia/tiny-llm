import mlx.core as mx
from .basics import softmax, linear
import math
from extensions import tiny_llm_ext


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    scale_factor = 1 / math.sqrt((query.shape[-1])) if scale is None else scale
    attn_weight = (query @ key.swapaxes(-2, -1)) * scale_factor
    attn_weight = attn_weight + mask if mask is not None else attn_weight
    attn_weight = softmax(attn_weight, -1) # (N, H, L, L)
    return attn_weight @ value # (N, H, L, D)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        
        self.head_dim = self.hidden_size // self.num_heads

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, E = query.shape
        assert(E == self.hidden_size)
        D = E // self.num_heads
        query = linear(query, self.wq).reshape(N, L, self.num_heads, D).transpose(0, 2, 1, 3)
        key = linear(key, self.wk).reshape(N, L, self.num_heads, D).transpose(0, 2, 1, 3)
        value = linear(value, self.wv).reshape(N, L, self.num_heads, D).transpose(0, 2, 1, 3)
        
        attn_res = scaled_dot_product_attention_simple(
            query=query,
            key=key,
            value=value,
            mask=mask
        ) # N H L D
        
        attn_res = attn_res.transpose(0, 2, 1, 3).reshape(N, L, E)
        
        return linear(attn_res, self.wo)
        


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    return mx.triu(mx.full((L, S), float('-inf'), dtype=dtype), k = S - L + 1)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    '''
    H_q = n * n_repeats - even div
    1. Reshape Q to isolate the group structure: from (N, H_q, L, D) to (N, H, n_repeats, L, D) where n_repeats = H_q // H
    2. Expand K/V by inserting a size-1 dimension for n_repeats: from (N, H, S, D) to (N, H, 1, S, D)
    3. matmul broadcasts — Q (N, H, n_repeats, L, D) @ K^T (N, H, 1, D, S) → (N, H, n_repeats, L, S). The 1 in K broadcasts across   n_repeats, so each group of Q heads sees the same K/V.
    4. Apply scale, mask, softmax, second matmul — same as standard attention.
    5. Reshape back to (N, H_q, L, D) by collapsing (H, n_repeats).
    '''
    batch_shape = key.shape[:-3]
    H, S, D = key.shape[-3:]
    H_q, L, _ = query.shape[-3:]
    n_repeats = H_q // H
    
    Q = query.reshape(*batch_shape, H, n_repeats, L, D) # (N..., H, n_repeats, L, D)
    K = key[..., None, :, :] # (N..., H, 1, S, D) -> will auto broadcast to (N..., H, n_repeats, S, D) in matmul
    V = value[..., None, :, :] # (N..., H, 1, S, D)
    
    if isinstance(mask, str) and mask == "causal":
        M = causal_mask(L, S, query.dtype)
    elif mask is not None:
        M = mx.broadcast_to(mask, (*batch_shape, H_q, L, S)) # (B, 1, L, S) -> (B, H_q, L, S)
        M = M.reshape(*batch_shape, H, n_repeats, L, S) # (B, H_q, L, S) -> (B, H, n_repeats, L, S)
    else:
        M = None

    
    attn_res = scaled_dot_product_attention_simple(Q, K, V, scale, M) # (N..., H, n_repeats, L, D)
    
    return attn_res.reshape(*batch_shape, H_q, L, D)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    factor = mx.rsqrt(query.shape[-1]) if scale is None else mx.array(scale)
    factor = factor.astype(query.dtype)

    *B, H_q, L, E = query.shape
    _, H, S, _ = key.shape
    assert H_q % H == 0
    query = query.reshape(-1, L, E)
    key = key.reshape(-1, S, E)
    value = value.reshape(-1, S, E)
    query = mx.contiguous(query)
    key = mx.contiguous(key)
    value = mx.contiguous(value)
    is_causal = mask == "causal"
    N = query.shape[0]
    if is_causal:
        mask = mx.broadcast_to(causal_mask(L, S, mx.float32), (*B, H_q, L, S))
    elif mask is None:
        mask = mx.broadcast_to(mx.zeros((L, S), dtype=mx.float32), (*B, H_q, L, S))
    else:
        mask = mx.broadcast_to(mask, (*B, H_q, L, S))
    mask = mx.contiguous(mask.reshape(N, L, S)).astype(mx.float32)
    result = tiny_llm_ext.flash_attention(
        query,
        key,
        value,
        mask,
        factor,
        is_causal=is_causal,
        num_heads=H_q,
        num_kv_heads=H,
    )
    return mx.contiguous(result.reshape(*B, H_q, L, E))
