import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: mx.array):
        logprobs = copy.copy(logprobs) # DO COPY!
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        if top_k is not None:
            top_k_idx = mx.argpartition(-logprobs, top_k-1)
            logprobs[top_k_idx[top_k:]] = float('-inf')
        if top_p is not None:
            sorted_idx = mx.argsort(-logprobs) # IDX!
            cumsum = mx.cumsum(mx.exp(logprobs[sorted_idx]))
            logprobs[sorted_idx[cumsum > top_p]] = float('-inf')
        return mx.random.categorical(logprobs/temp, -1)

    return sample
