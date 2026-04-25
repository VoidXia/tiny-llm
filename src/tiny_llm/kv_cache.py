from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array, # B, H, S, D
        value: mx.array, # B, H, S, D
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """

def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask

class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.batched_keys = None
        self.batched_values = None
        self.slots: list[TinyKvCache | None] = [None] * max_active_requests

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        S = 0
        S_i = [0] * self.max_active_requests
        for i in range(self.max_active_requests):
            if self.slots[i] is None:
                continue
            key, value, S_i[i], _ = self.slots[i].update_and_fetch(keys[i:i+1], values[i:i+1], mask_length, mask)
            # [i] would not preserve first dim but [i:i+1] would keep shape (1, ...)
            S = max(S, S_i[i]) # S_i is the after append token size
        B, H, _, D = keys.shape
        out_mask = mx.full((B, 1, mask_length, S), -mx.inf, dtype=keys.dtype)
        self.batched_keys = mx.zeros((B, H, S, D), dtype=keys.dtype)
        self.batched_values = mx.zeros((B, H, S, D), dtype=keys.dtype) # B H S D
        for i in range(self.max_active_requests):
            if self.slots[i] is None:
                continue
            self.batched_keys[i, :, (S - S_i[i]):S, :] = self.slots[i].key[0]
            self.batched_values[i, :, (S - S_i[i]):S, :] = self.slots[i].value[0]
            if mask == None or mask == "causal":
                out_mask[i, :, :, (S - S_i[i]):S] = causal_mask(mask_length, S_i[i], dtype=keys.dtype)
            else:
                out_mask[i, :, :, (S - S_i[i]):S] = mask
        return self.batched_keys, self.batched_values, None, out_mask

    def add_request(self, prefilled: TinyKvCache, id: int):
        self.slots[id] = prefilled

    def remove_request(self, id: int):
        self.slots[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key = None
        self.value = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if self.key is None:
            self.key = key
        else:
            self.key = mx.concatenate([self.key, key], axis=-2)
        if self.value is None:
            self.value = value
        else:
            self.value = mx.concatenate([self.value, value], axis=-2)
        self.offset = self.key.shape[-2]
        return self.key, self.value, self.key.shape[-2], mask
