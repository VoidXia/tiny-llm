import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.freq_base = 1.0 / (base ** (mx.arange(0, dims, 2) / float(dims)))
        self.freq = mx.arange(0, seq_len)[:, None] * self.freq_base[None, :]

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        if isinstance(offset, list):
            offset = mx.array([list(range(s.start, s.stop)) for s in offset])
        if offset is None:
            freqs = self.freq[:x.shape[1]]  # (seq_len, dims//2)
        else:
            freqs = self.freq[offset]
        cos_f = mx.cos(freqs)[..., None, :]
        sin_f = mx.sin(freqs)[..., None, :] # (L, 1, D//2)
        if self.traditional:
            x_pairs = x.reshape(*x.shape[:-1], self.dims // 2, 2)
            x1 = x_pairs[..., 0] # 0,2,4,6 ...
            x2 = x_pairs[..., 1] # 1,3,5,7 ...
        else:
            x1 = x[..., :self.dims//2] # first half
            x2 = x[..., self.dims//2:] # second half
        o1 = x1 * cos_f - x2 * sin_f
        o2 = x1 * sin_f + x2 * cos_f
        if self.traditional:
            return mx.stack([o1, o2], axis=-1).reshape(x.shape) # [o1[0], o2[0], o1[1], o2[1], ...]
        return mx.concat([o1, o2], axis=-1) # [o1[0], o1[1], ..., o2[0], o2[1], ...] 
            
