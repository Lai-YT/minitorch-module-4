from typing import Tuple, Optional

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_h = height // kh
    new_w = width // kw

    # 1. Reshape to split height and width into (new_h, kh) and (new_w, kw)
    # Current: batch x channel x height x width
    # Target: batch x channel x new_h x kh x new_w x kw
    input = input.contiguous().view(batch, channel, new_h, kh, new_w, kw)

    # 2. Permute to bring kh and kw to the end
    # Current: 0:batch, 1:channel, 2:new_h, 3:kh, 4:new_w, 5:kw
    # Target: batch x channel x new_h x new_w x kh x kw
    input = input.permute(0, 1, 2, 4, 3, 5)

    # 3. View to merge kh and kw into a single dimension
    input = input.contiguous().view(batch, channel, new_h, new_w, kh * kw)

    return input, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape
    input, new_h, new_w = tile(input, kernel)
    return input.mean(dim=4).view(batch, channel, new_h, new_w)


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: input tensor
        dim: dimension to apply argmax

    Returns:
    -------
        one-hot tensor with 1 index of the maximum value.

    """
    out = max_reduce(input, dim)
    mask = input == out
    return mask / mask.sum(dim=dim)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max"""
        ctx.save_for_backward(input, int(dim.item()))
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max"""
        input, dim = ctx.saved_values
        return (grad_output * argmax(input, dim)), 0.0


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Apply max reduction"""
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape
    input, new_h, new_w = tile(input, kernel)
    return max(input, dim=4).view(batch, channel, new_h, new_w)


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: input tensor
        dim: dimension to apply softmax

    Returns:
    -------
        softmax tensor

    """
    exp_x = input.exp()
    return exp_x / exp_x.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: input tensor
        dim: dimension to apply logsoftmax

    Returns:
    -------
        logsoftmax tensor

    """
    max_x = max(input, dim=dim)
    return input - max_x - (input - max_x).exp().sum(dim=dim).log()


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: input tensor
        p: dropout probability
        ignore: whether to ignore dropout

    Returns:
    -------
        dropout tensor

    """
    if ignore or p == 0.0:
        return input
    if p == 1.0:
        return input.zeros(input.shape)

    mask = rand(input.shape, backend=input.backend) > p
    return input * mask * (1.0 / (1.0 - p))
