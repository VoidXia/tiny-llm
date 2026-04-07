import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight # vocab_size x embedding_dim

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x] # add a dimension of embedding_dim (sticks)

    def as_linear(self, x: mx.array) -> mx.array: # Input: N.. x embedding_dim
        return x @ self.weight.T # Output: N.. x vocab_size
