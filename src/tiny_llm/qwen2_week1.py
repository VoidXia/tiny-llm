import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq # E=H_q*D, E
        self.bq = bq # E, 
        self.wk = wk # H*D, E
        self.bk = bk # H*D,
        self.wv = wv # H*D, E
        self.bv = bv # H*D,
        self.wo = wo # E, E=H_q*D
        
        self.D = self.hidden_size // self.num_heads
        
        self.rope = RoPE(self.D, max_seq_len, theta, traditional=False)

    def __call__(
        self,
        x: mx.array, # B, L, E
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        # E = self.hidden_size
        H_q = self.num_heads
        H = self.num_kv_heads
        D = self.D
        
        # q = linear(x, wq, bq) -> B, L, H_q, D
        q = linear(x, self.wq, self.bq).reshape(B, L, H_q, D)
        # k = linear(x, wk, bk) -> B, L, H, D
        k = linear(x, self.wk, self.bk).reshape(B, L, H, D)
        # v = linear(x, wv, bv) -> B, L, H, D
        v = linear(x, self.wv, self.bv).reshape(B, L, H, D)
        # q = rope(q, offset=slice(0, L))
        q = self.rope(q, offset=slice(0, L))
        # k = rope(k, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))
        # (transpose as needed)
        # x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
        # S == L, no KV cache
        x = scaled_dot_product_attention_grouped(q.swapaxes(-3, -2), k.swapaxes(-3, -2), v.swapaxes(-3, -2), mask=mask) # B, H_q, L, D
        # (transpose as needed)
        # x = linear(x, wo) -> B, L, E
        x = linear(x.swapaxes(-3, -2).reshape(B, L, H_q*D), self.wo)
        return x


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
