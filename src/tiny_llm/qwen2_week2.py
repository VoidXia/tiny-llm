import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
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
        x: mx.array,
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        # E = self.hidden_size
        H_q = self.num_heads
        H = self.num_kv_heads
        D = self.D
        
        # q = linear(x, wq, bq) -> B, L, H_q, D
        q = linear(x, dequantize_linear(self.wq), self.bq).reshape(B, L, H_q, D)
        # k = linear(x, wk, bk) -> B, L, H, D
        k = linear(x, dequantize_linear(self.wk), self.bk).reshape(B, L, H, D)
        # v = linear(x, wv, bv) -> B, L, H, D
        v = linear(x, dequantize_linear(self.wv), self.bv).reshape(B, L, H, D)
        # q = rope(q, offset=slice(0, L))
        q = self.rope(q, offset=[slice(i, i + L) for i in offsets])
        # k = rope(k, offset=slice(0, L))
        k = self.rope(k, offset=[slice(i, i + L) for i in offsets])
        # (transpose as needed)
        # k, v = cache.update_and_fetch(k, v) ; k/v: B, L, H, D, q: B, L', H, D
        k, v = cache.update_and_fetch(k, v)
        # x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
        x = scaled_dot_product_attention_grouped(q.swapaxes(-3, -2), k.swapaxes(-3, -2), v.swapaxes(-3, -2), mask=mask) # B, H_q, L, D
        # (transpose as needed)
        # x = linear(x, wo) -> B, L, E
        x = linear(x.swapaxes(-3, -2).reshape(B, L, H_q*D), dequantize_linear(self.wo))
        return x


class Qwen2MLP: # input: N.. x L x E output: N.. x L x E
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        gate = silu(linear(x, dequantize_linear(self.w_gate))) # L, I
        up = linear(x, dequantize_linear(self.w_up)) # L, I
        hidden = gate * up
        return linear(hidden, dequantize_linear(self.w_down))


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.H_q = num_attention_heads
        self.H = num_kv_heads
        self.E = hidden_size
        self.D = self.E // self.H_q
        self.attn = Qwen2MultiHeadAttention(self.E, self.H_q, self.H, wq, wk, wv, wo, bq, bk, bv, max_seq_len, theta)
        self.mlp = Qwen2MLP(self.D, intermediate_size, w_gate, w_up, w_down)
        self.input_layernorm = RMSNorm(self.D, w_input_layernorm, rms_norm_eps)
        self.post_attn_layernorm = RMSNorm(self.D, w_post_attention_layernorm, rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x_attn = self.attn(self.input_layernorm(x), offsets=offset, cache=cache, mask=mask)
        x_int = x + x_attn
        x_mlp = self.mlp(self.post_attn_layernorm(x_int))
        return x_int + x_mlp


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.model = mlx_model
        self.embedding = Embedding(
            vocab_size= self.model.args.vocab_size, 
            embedding_dim= self.model.args.hidden_size, 
            weight=dequantize_linear(self.model.model.embed_tokens))
        self.transformer = [Qwen2TransformerBlock(
            num_attention_heads= self.model.args.num_attention_heads,
            num_kv_heads= self.model.args.num_key_value_heads,
            hidden_size= self.model.args.hidden_size,
            intermediate_size= self.model.args.intermediate_size,
            rms_norm_eps= self.model.args.rms_norm_eps,
            wq= self.model.model.layers[i].self_attn.q_proj,
            wk= self.model.model.layers[i].self_attn.k_proj,
            wv= self.model.model.layers[i].self_attn.v_proj,
            wo= self.model.model.layers[i].self_attn.o_proj,
            bq= self.model.model.layers[i].self_attn.q_proj.bias,
            bk= self.model.model.layers[i].self_attn.k_proj.bias,
            bv= self.model.model.layers[i].self_attn.v_proj.bias,
            w_gate= self.model.model.layers[i].mlp.gate_proj,
            w_up= self.model.model.layers[i].mlp.up_proj,
            w_down= self.model.model.layers[i].mlp.down_proj,
            w_input_layernorm= self.model.model.layers[i].input_layernorm.weight,
            w_post_attention_layernorm= self.model.model.layers[i].post_attention_layernorm.weight,
            max_seq_len= self.model.args.max_position_embeddings,
            theta= self.model.args.rope_theta
        ) for i in range(self.model.args.num_hidden_layers)]
        self.rmsnorm = RMSNorm(self.model.args.hidden_size, self.model.model.norm.weight, self.model.model.norm.eps)
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        x = inputs
        x = self.embedding(x)
        for i in range(self.model.args.num_hidden_layers):
            x = self.transformer[i](x, offset=[offset], cache=cache[i], mask = "causal" if inputs.size > 1 else None)
        x = self.rmsnorm(x)
        x = self.embedding.as_linear(x) if self.model.args.tie_word_embeddings else x @ dequantize_linear(self.model.lm_head).T
        return x

