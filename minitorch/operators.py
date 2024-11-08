"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number. Enforce the returned type to float."""
    return float(-x)


def lt(x: float, y: float) -> float:
    """Check if x is less than y."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close to y. f(x) = |x - y| < 1e-2"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    $f(x) =  1.0/(1.0 + e^{-x})$ if x >=0 else $e^x/(1.0 + e^{x})$.

    sigmoid function is splitted into two parts to avoid overflow.
    math.exp(-x) will be relatively small for large positive x. math.exp(x) will be relatively small for large negative x.

    Args:
        x: Input to the function.

    Returns:
        float: The output of the sigmoid function.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU function.

    $f(x) = x$ if x > 0 else 0
    """
    return max(0.0, x)


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential function."""
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the inverse of a number."""
    return 1.0 / (x)


def log_back(x: float, d: float) -> float:
    """Computes the derivative of log(x)*d over x."""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of d/x over x."""
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the d*relu(x) over x."""
    return d if x > 0 else 0.0


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable and returns a list of results."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [f(i) for i in ls]

    return _map


def negList(x: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg)(x)


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function. It only performs on the minimum length of the two iterables."""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [f(x, y) for x, y in zip(ls1, ls2)]

    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add)(ls1, ls2)


def reduce(
    f: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function."""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for i in ls:
            val = f(val, i)
        return val

    return _reduce


def sum(x: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, 0.0)(x)


def prod(x: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, 1.0)(x)
