import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Note that, mean calculation should be performed with float32 accumulation to maintain precision before taking the square root, even if the input and weights are in a lower precision format (e.g., float16 or bfloat16).
        x = x.astype(mx.float32)
        # keepdims for broadcast back
        return x / (mx.sqrt(mx.mean(x**2, axis=-1, keepdims=True)) + self.eps) * self.weight
