"""
Fast Fourier transforms for tinygrad.

Complex tensors carry a trailing dimension of size 2 with the real and imaginary parts, as `torch.view_as_real` does.

A transform is a sequence of stages, each one kernel. A stage transforms some factors of the length: one by a direct DFT (an
explicit sum, optional), the others by butterflies (small unrolled DFTs). What is transformed moves after the part that is
not transformed yet (Stockham autosort), so the result needs no reordering. `.contiguous()` ends a stage: tinygrad would
otherwise fuse the whole transform into one kernel that takes minutes to compile. Lengths with prime factors above 7 use
Bluestein's algorithm.

```python
x = Tensor.randn(4, 32, 32, 2)
k = fftn(x, dim=(-2, -1))
print((ifftn(k, dim=(-2, -1)) - x).abs().max().item())
```
"""
from __future__ import annotations

import functools
import itertools
import math
import operator
from typing import Literal

from tinygrad.dtype import DType, dtypes
from tinygrad.function import function
from tinygrad.helpers import prod
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import UOp

Norm = Literal["backward", "ortho", "forward"]
Complex = tuple[Tensor, Tensor]
Stage = tuple[int, tuple[int, ...], int]  # size of the direct DFT, radices of the butterflies, length left for the later stages

RADICES = (4, 2, 3, 5, 7)  # sizes with an unrolled butterfly, in order of preference
MAX_RADIX = 24  # largest product of the radices in a stage: its 2*MAX_RADIX real outputs are stacked, tinygrad unrolls at most 49
MAX_DIRECT = 16  # largest direct DFT in a stage. every output is a GPU thread that reads all inputs of the DFT
BLOCK_SIZE = 8  # transforms side by side in the dimension in front of the complex one. GPU kernels are slow when that has size 1

# *** complex arithmetic on (real, imag) pairs ***

def _add(a:Complex, b:Complex) -> Complex: return a[0]+b[0], a[1]+b[1]
def _sub(a:Complex, b:Complex) -> Complex: return a[0]-b[0], a[1]-b[1]
def _mul(a:Complex, b:Complex|tuple[float, float]) -> Complex: return a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]
def _split(x:Tensor) -> Complex: return x[..., 0], x[..., 1]
def _join(z:Complex) -> Tensor: return Tensor.stack(*z, dim=-1)

def _unit_root(k:int, n:int) -> tuple[float, float]:
  """Return `exp(-2j*pi*k/n)`, exact for quarter turns so that tinygrad folds the multiplication away."""
  if 4*(k := k % n) % n == 0: return ((1., 0.), (0., -1.), (-1., 0.), (0., 1.))[4*k//n]
  return math.cos(2*math.pi*k/n), -math.sin(2*math.pi*k/n)

def _unit_roots(k:Tensor, n:int, dtype:DType) -> Tensor:
  """Return `exp(-2j*pi*k/n)` for a tensor of integers `k`, which are reduced before the cast to keep the angle accurate."""
  angle = (k % n).cast(dtype) * (-2*math.pi/n)
  return Tensor.stack(angle.cos(), angle.sin(), dim=-1)

# *** plan and tables, cached ***

@functools.cache
def _plan_stages(n:int) -> tuple[Stage, ...]|None:
  """Split a transform of length `n` into stages of one kernel each, `None` if `n` has a prime factor above 7.

  Returns
  -------
    Per stage the size of its direct DFT (1 for none), which runs first, the radices of its butterflies and the length
    that is left for the later stages.
  """
  factors: list[int] = []
  for radix in RADICES:
    while n % radix == 0: factors, n = [*factors, radix], n//radix
  if n != 1: return None
  def take_factors(limit:int) -> tuple[int, ...]:
    """Remove and return the factors with the largest product up to `limit`."""
    candidates = [c for count in range(len(factors)+1) for c in set(itertools.combinations(factors, count)) if prod(c) <= limit]
    for factor in (best := max(candidates, key=prod)): factors.remove(factor)
    return best
  stages = []
  while factors:
    radices, direct_size = take_factors(MAX_RADIX), prod(take_factors(MAX_DIRECT))
    stages.append((direct_size, radices, prod(factors)))
  return tuple(stages)

@functools.cache
def _stage_tables(direct_size:int, radices:tuple[int, ...], remaining:int, dtype:DType, device:str) -> list[Tensor]:
  """Compute the weights and twiddles of a stage.

  Parameters
  ----------
  direct_size
    Size of the direct DFT, 1 for none.
  radices
    Radices of the butterflies, in the order they run.
  remaining
    Length that is left for the later stages.

  Returns
  -------
    The `(direct_size, direct_size, 2)` weights of the direct DFT, its `(direct_size, m, 2)` twiddles and the `(radix-1, m, 2)`
    twiddles of every butterfly, without its first output, which has none. `m` is the length that is not transformed after
    that part. They are views of one buffer: tinygrad does not fuse what reads more than three buffers into its consumer.
  """
  def roots(first:int, size:int, m:int, n:int) -> Tensor: return _unit_roots(Tensor.arange(first, size).reshape(-1, 1) * Tensor.arange(m), n, dtype)
  untransformed = [prod(radices[i+1:])*remaining for i in range(len(radices))]
  tables = [roots(0, direct_size, direct_size, direct_size), roots(0, direct_size, m := prod(radices)*remaining, direct_size*m)]
  tables += [roots(1, radix, m, radix*m) for radix, m in zip(radices, untransformed, strict=True)]
  buffer = Tensor.cat(*[t.reshape(-1, 2) for t in tables]).clone(device).realize()
  ends = list(itertools.accumulate(int(prod(t.shape[:-1])) for t in tables))
  return [buffer[end-int(prod(t.shape[:-1])):end].reshape(t.shape) for t, end in zip(tables, ends, strict=True)]

def _bluestein_size(n:int) -> int:
  """Return the power of two length of the transforms that Bluestein's algorithm uses for length `n`."""
  return 1 << (2*n-2).bit_length()

@functools.cache
def _bluestein_tables(n:int, dtype:DType, device:str) -> tuple[Tensor, Tensor]:
  """Compute Bluestein's `(n, 2)` chirp `exp(-1j*pi*k**2/n)` and the `(padded, 2)` transform of its wrapped around conjugate."""
  padded = _bluestein_size(n)
  prepare_tables(padded, dtype, device)
  k = Tensor.arange(n, dtype=dtypes.int64)
  chirp = _unit_roots(k*k, 2*n, dtype).clone(device).realize()
  conjugate = chirp * Tensor([1, -1], dtype=dtype, device=device)
  wrapped = conjugate.pad(((0, padded-n), None)) + conjugate[1:].flip(0).pad(((padded-n+1, 0), None))
  return chirp, _fft_1d(wrapped.reshape(1, padded, 1, 2)).reshape(padded, 2).realize()

def prepare_tables(n:int, dtype:DType, device:str) -> None:
  """Compute and cache the tables of a transform of length `n`.

  The transforms run as a `tinygrad.function.function`, whose body must not use the device. Call this before running `fft_dims`
  in the body of another function.
  """
  if (stages := _plan_stages(n)) is None: _bluestein_tables(n, dtype, device)
  for stage in stages or (): _stage_tables(*stage, dtype, device)

# *** the transform ***

def _direct_dft(x:Tensor, weights:Tensor, twiddles:Tensor|None) -> Tensor:
  """Apply a DFT along dimension 1 by summation, then the twiddles.

  Parameters
  ----------
  x
    Complex tensor `(lead, direct_size, untransformed, tail, 2)`. `tail` holds what is transformed already and the dimensions
    after the transformed one.
  weights
    `(direct_size, direct_size, 2)`, output first.
  twiddles
    `(direct_size, untransformed, 2)`, `None` for all ones.

  Returns
  -------
    `(lead, untransformed, direct_size, tail, 2)`: the outputs move after what is not transformed yet.
  """
  direct_size, untransformed = int(x.shape[1]), int(x.shape[2])
  y = functools.reduce(_add, [_mul(_split(x[:, j].unsqueeze(2)), _split(weights[:, j].reshape(1, 1, direct_size, 1, 2))) for j in range(direct_size)])
  return _join(y if twiddles is None else _mul(y, _split(twiddles.transpose(0, 1).reshape(1, untransformed, direct_size, 1, 2))))

def _butterfly(x:Tensor, twiddles:Tensor|None) -> Tensor:
  """Apply an unrolled DFT along dimension 1, then the twiddles.

  Parameters
  ----------
  x
    Complex tensor `(lead, radix, ..., 2)`.
  twiddles
    `(1, radix-1, ..., 2)`, broadcastable to `x` without its first output, which needs none. `None` for all ones.

  Returns
  -------
    Tensor with the shape of `x`.
  """
  radix = int(x.shape[1])
  xs = [_split(x[:, j]) for j in range(radix)]
  if radix == 4:
    a, b, c, d = _add(xs[0], xs[2]), _sub(xs[0], xs[2]), _add(xs[1], xs[3]), _sub(xs[1], xs[3])
    ys = [_add(a, c), (b[0]+d[1], b[1]-d[0]), _sub(a, c), (b[0]-d[1], b[1]+d[0])]
  else: ys = [functools.reduce(_add, [_mul(xj, _unit_root(j*k, radix)) for j, xj in enumerate(xs)]) for k in range(radix)]
  if twiddles is not None: ys = [ys[0]] + [_mul(y, _split(twiddles[:, k])) for k, y in enumerate(ys[1:])]
  return Tensor.stack(*[_join(y) for y in ys], dim=1)

def _bluestein_fft(x:Tensor) -> Tensor:
  """Apply the unnormalized forward transform of any length along dimension 1 of `(lead, n, trail, 2)`.

  The transform is a convolution with a chirp, which is done with transforms of a power of two length.
  """
  def mul(a:Tensor, b:Tensor) -> Tensor: return _join(_mul(_split(a), _split(b)))
  n, padded = int(x.shape[1]), _bluestein_size(int(x.shape[1]))
  chirp, chirp_spectrum = _bluestein_tables(n, x.dtype, x.device)
  y = mul(_fft_1d(mul(x, chirp.reshape(1, n, 1, 2)).pad((None, (0, padded-n), None, None))), chirp_spectrum.reshape(1, padded, 1, 2))
  return mul(_fft_1d(y.flip(-1)).flip(-1)[:, :n] * (1/padded), chirp.reshape(1, n, 1, 2))

def _fft_1d(x:Tensor) -> Tensor:
  """Apply the unnormalized forward transform along dimension 1 of `x` with shape `(lead, n, trail, 2)`."""
  shape, lead, n = x.shape, int(x.shape[0]), int(x.shape[1])
  if (stages := _plan_stages(n)) is None: return _bluestein_fft(x)
  untransformed = n  # what is transformed moves after it
  for direct_size, radices, remaining in stages:
    weights, twiddles, *butterfly_twiddles = _stage_tables(direct_size, radices, remaining, x.dtype, x.device)
    if direct_size > 1:
      untransformed //= direct_size
      x = _direct_dft(x.reshape(lead, direct_size, untransformed, -1, 2), weights, twiddles if untransformed > 1 else None)
    done: tuple[int, ...] = ()  # radices of this stage that have run, newest first
    for radix, tw in zip(radices, butterfly_twiddles, strict=True):
      untransformed //= radix
      tw = tw.reshape(1, radix-1, untransformed, *(1,)*len(done), 1, 2)
      x = _butterfly(x.reshape(lead, radix, untransformed, *done, -1, 2), tw if untransformed > 1 else None)
      x, done = x.permute(0, 2, 1, *range(3, 5+len(done))), (radix, *done)
    x = x.contiguous().reshape(shape)
  return x

def _fft_1d_blocked(x:Tensor, dim:int) -> Tensor:
  """Apply the unnormalized forward transform along `dim` with `BLOCK_SIZE` transforms side by side.

  The transforms of a block sit in the dimension in front of the complex one. The product of the other dimensions has to be
  a multiple of `BLOCK_SIZE`. The result is a view of a buffer in that layout.
  """
  order = (*(i for i in range(x.ndim-1) if i != dim), dim, x.ndim-1)
  y = x.permute(order)
  y = _fft_1d(y.reshape(-1, BLOCK_SIZE, y.shape[-2], 2).transpose(1, 2)).transpose(1, 2).reshape(y.shape)
  return y.permute(*(order.index(i) for i in range(x.ndim)))

def fft_dims(x:Tensor, dims:tuple[int, ...], scale:float, inverse:bool) -> Tensor:
  """Apply the scaled transform over `dims`, without checks and without an explicit gradient.

  The inverse is the forward transform with real and imaginary parts swapped before and after. `dims` are not negative,
  and `prepare_tables` has to have run for their lengths.
  """
  shape = x.shape[:-1]
  if inverse: x = x.flip(-1)
  if scale != 1: x = x * scale  # first, where it fuses into a kernel
  for dim in dims:
    # only the last kernel has to leave a buffer in the order of the result
    if dim != dims[-1] and int(prod(shape)) // shape[dim] % BLOCK_SIZE == 0: x = _fft_1d_blocked(x, dim)
    else: x = _fft_1d(x.reshape(int(prod(shape[:dim])), shape[dim], -1, 2)).reshape(*shape, 2)
  return x.flip(-1) if inverse else x

def _differentiable_fft(x:Tensor, dims:tuple[int, ...], scale:float, inverse:bool) -> Tensor:
  """Apply `fft_dims` with an explicit gradient: the transform in the other direction with the same scale.

  The backward pass that tinygrad derives takes minutes to compile. The tables are implicit inputs of the function and get no gradient.
  """
  def forward(x:Tensor) -> Tensor: return fft_dims(x, dims, scale, inverse)
  def backward(grad:UOp, call:UOp) -> tuple[UOp|None, ...]:
    # the inputs of the call are x and then the tables
    return (_differentiable_fft(Tensor(grad), dims, scale, not inverse).uop, *[None]*(len(call.src)-2))
  return function(forward, allow_implicit=True, grad_fxn=backward)(x)

def _fftn(x:Tensor, dim:int|tuple[int, ...]|None, norm:Norm, inverse:bool) -> Tensor:
  """Check the arguments of the public functions, pick the dtype to compute in, prepare the tables and run the transform."""
  if x.ndim < 1 or x.shape[-1] != 2: raise ValueError(f"expected a complex tensor with a trailing dimension of size 2, got shape {x.shape}")
  if not all(isinstance(s, int) for s in x.shape): raise ValueError(f"symbolic shapes are not supported, got {x.shape}")
  if not isinstance(x.device, str): raise NotImplementedError(f"tensors on several devices are not supported, got {x.device}")
  ndim, shape = x.ndim-1, x.shape[:-1]
  dims = tuple(range(ndim)) if dim is None else tuple(operator.index(d) for d in (dim if isinstance(dim, (tuple, list)) else (dim,)))
  if any(not -ndim <= d < ndim for d in dims): raise IndexError(f"{dim=} out of range for {ndim} complex dimensions")
  if len(set(dims := tuple(d % ndim for d in dims))) != len(dims): raise ValueError(f"repeated dimension in {dim=}")
  if any(shape[d] == 0 for d in dims): raise ValueError(f"cannot transform shape {x.shape} along {dim=}")
  if norm not in ("backward", "ortho", "forward"): raise ValueError(f"unknown {norm=}")
  # half precision is computed in single precision: its angles and sums are not accurate enough
  dtype = x.dtype if dtypes.is_float(x.dtype) else dtypes.default_float
  working = dtype if dtype.itemsize >= 4 else dtypes.float32
  if 0 in shape or not (dims := tuple(d for d in dims if shape[d] > 1)): return x.cast(dtype)
  for d in dims: prepare_tables(int(shape[d]), working, x.device)
  size = int(prod(shape[d] for d in dims))
  scale = {"backward": 1/size if inverse else 1, "ortho": size**-0.5, "forward": 1 if inverse else 1/size}[norm]
  return _differentiable_fft(x.cast(working), dims, scale, inverse).cast(dtype)

def fft(x:Tensor, dim:int=-1, norm:Norm="backward") -> Tensor:
  """Compute the one-dimensional discrete Fourier transform.

  Parameters
  ----------
  x
    Complex tensor with real and imaginary parts in a trailing dimension of size 2. Other dtypes than floats are cast to the
    default float.
  dim
    Dimension to transform, not counting the trailing complex dimension.
  norm
    `"backward"` (no scaling), `"ortho"` (scaled by `1/sqrt(n)`) or `"forward"` (scaled by `1/n`), with `n` the length of `dim`.

  Returns
  -------
    Transform of `x`, with the shape of `x`.
  """
  return _fftn(x, (operator.index(dim),), norm, False)

def ifft(x:Tensor, dim:int=-1, norm:Norm="backward") -> Tensor:
  """Compute the one-dimensional inverse discrete Fourier transform.

  Parameters
  ----------
  x
    Complex tensor with real and imaginary parts in a trailing dimension of size 2. Other dtypes than floats are cast to the
    default float.
  dim
    Dimension to transform, not counting the trailing complex dimension.
  norm
    `"backward"` (scaled by `1/n`), `"ortho"` (scaled by `1/sqrt(n)`) or `"forward"` (no scaling), with `n` the length of `dim`.

  Returns
  -------
    Inverse transform of `x`, with the shape of `x`.
  """
  return _fftn(x, (operator.index(dim),), norm, True)

def fftn(x:Tensor, dim:int|tuple[int, ...]|None=None, norm:Norm="backward") -> Tensor:
  """Compute the n-dimensional discrete Fourier transform.

  Parameters
  ----------
  x
    Complex tensor with real and imaginary parts in a trailing dimension of size 2. Other dtypes than floats are cast to the
    default float.
  dim
    Dimensions to transform, not counting the trailing complex dimension. `None` transforms all of them, the others are
    batch dimensions.
  norm
    `"backward"` (no scaling), `"ortho"` (scaled by `1/sqrt(n)`) or `"forward"` (scaled by `1/n`), with `n` the product of
    the lengths of `dim`.

  Returns
  -------
    Transform of `x`, with the shape of `x`.
  """
  return _fftn(x, dim, norm, False)

def ifftn(x:Tensor, dim:int|tuple[int, ...]|None=None, norm:Norm="backward") -> Tensor:
  """Compute the n-dimensional inverse discrete Fourier transform.

  Parameters
  ----------
  x
    Complex tensor with real and imaginary parts in a trailing dimension of size 2. Other dtypes than floats are cast to the
    default float.
  dim
    Dimensions to transform, not counting the trailing complex dimension. `None` transforms all of them, the others are
    batch dimensions.
  norm
    `"backward"` (scaled by `1/n`), `"ortho"` (scaled by `1/sqrt(n)`) or `"forward"` (no scaling), with `n` the product of
    the lengths of `dim`.

  Returns
  -------
    Inverse transform of `x`, with the shape of `x`.
  """
  return _fftn(x, dim, norm, True)
