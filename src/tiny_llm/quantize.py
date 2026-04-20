import mlx.core as mx
from extensions import tiny_llm_ext
from typing import Any


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    *N, D = a.shape
    a = mx.contiguous(a.reshape(-1, D))
    b = mx.contiguous(b)
    return tiny_llm_ext.quantized_matmul(
        scales, biases, group_size, bits, a, b, transpose_b
    ).reshape(*N, -1)


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    # Copied from ref - w2d2t1
    if bias is not None:
        return (
            quantized_matmul(
                w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
            )
            + bias
        )
    else:
        return quantized_matmul(
            w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
        )
