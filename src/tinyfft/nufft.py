"""
Nonuniform fast Fourier transforms for tinygrad, after FINUFFT.

A transform between a uniform grid of modes and nonuniform points goes through a grid that is at least twice as fine:
the fine grid is transformed with an FFT, and a narrow kernel connects it to the points. Reading the grid at the points
(interpolation) is a gather. Adding the points to the grid (spreading) is a scatter-add, which tinygrad does not have: it
is a custom kernel of atomic adds.

```python
op = Nufft(points, (256, 256))  # points: (n_points, 2) in radians
samples = op(image)             # (..., 256, 256, 2) -> (..., n_points, 2)
image = op.adjoint(samples)
```
"""
from __future__ import annotations

import functools
import itertools
import math
import operator
from collections.abc import Sequence
from typing import cast

from tinygrad.device import Device
from tinygrad.dtype import DType, dtypes
from tinygrad.function import function
from tinygrad.helpers import prod
from tinygrad.tensor import Tensor
from tinygrad.uop import Ops
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from tinyfft.fft import fft_dims, prepare_tables

BETA = 2.30  # the kernel is exp(BETA*width*(sqrt(1-z**2)-1)) for a distance z in half widths, FINUFFT's choice for a twice as fine grid
GROUP_SIZE = 64  # GPU threads of the spreading kernel are launched in groups, one at a time is slow

# `*{0} += {1}` as an atomic operation, per device. OpenCL has no atomic add for floats: it exchanges the bits of a 32 bit float
# until no one interfered.
ATOMIC_ADD = {
  "CUDA": "atomicAdd({0}, {1});", "NV": "atomicAdd({0}, {1});", "CPU": "__atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED);",
  "CL": "{{ volatile __global unsigned int *p = (volatile __global unsigned int *)({0}); union {{ float f; unsigned int u; }} o, n; "
        "do {{ o.u = *p; n.f = o.f + ({1}); }} while (atomic_cmpxchg(p, o.u, n.u) != o.u); }}",
}

def _fine_grid_size(size:int, kernel_width:int) -> int:
  """Return the smallest even size of at least `2*size` and `2*kernel_width` whose prime factors are 2, 3 and 5.

  With `2*kernel_width` cells the kernel of a point never wraps around onto itself.
  """
  def is_smooth(m:int) -> bool:
    for p in (2, 3, 5):
      while m % p == 0: m //= p
    return m == 1
  return next(m for m in itertools.count(2*max(size, kernel_width), 2) if is_smooth(m))

def _gauss_legendre(order:int) -> list[tuple[float, float]]:
  """Return the nodes and weights of Gauss-Legendre quadrature on [-1, 1], by Newton's method on the Legendre recurrence."""
  rule = []
  for i in range(order):
    x = math.cos(math.pi * (i + 0.75) / (order + 0.5))
    for _ in range(100):
      p0, p1 = 1.0, x
      for k in range(2, order+1): p0, p1 = p1, ((2*k-1)*x*p1 - (k-1)*p0) / k
      dp = order * (x*p1 - p0) / (x*x - 1)
      x, step = x - p1/dp, p1/dp
      if abs(step) < 1e-15: break
    rule.append((x, 2 / ((1 - x*x) * dp*dp)))
  return rule

@functools.cache
def _deapodization(size:int, fine_size:int, kernel_width:int) -> list[float]:
  """Return one over the Fourier transform of the kernel at the modes `-(size//2)` to `(size-1)//2` of a dimension."""
  half = kernel_width/2
  rule = [(x*half, w*half * math.exp(BETA*kernel_width * (math.sqrt(1 - x*x) - 1))) for x, w in _gauss_legendre(2*kernel_width + 20)]
  return [1 / sum(w * math.cos(2*math.pi*k*x/fine_size) for x, w in rule) for k in range(-(size//2), (size+1)//2)]

def _interval(ndim:int, dim:int, start:int, stop:int) -> tuple[tuple[int, int]|None, ...]:
  """Return the argument of `shrink` or `pad` that acts on `dim` only."""
  return tuple((start, stop) if i == dim else None for i in range(ndim))

def _zero_kernel(out:UOp) -> UOp:
  """Fill a buffer with zeros. `custom_kernel` needs a buffer to write to, and a zero tensor is a constant."""
  i = UOp.range(out.numel(), 0)
  return cast(UOp, out.flatten()[i].store(0).end(i).sink(arg=KernelInfo(name="zero")))

def _spread(values:Tensor, cell_offsets:Tensor, kernel_weights:Tensor, n_cells:int, lane_axis:AxisType) -> Tensor:
  """Add every value, weighted with the kernel, to the grid cells around its point.

  tinygrad's scatter-add compares every cell with every value. This is a kernel of atomic adds as in the backward of
  `nn.Embedding`, with one GPU thread per point.

  Parameters
  ----------
  values
    `(n_points, n_channels)` values to spread.
  cell_offsets
    `(n_dims, n_points, kernel_width)` offsets into the flattened grid. A value goes to the sum of one offset per dimension.
  kernel_weights
    `(n_dims, n_points, kernel_width)` weights. A value is weighted with the product of one weight per dimension.
  n_cells
    Number of cells of the flattened grid.
  lane_axis
    Kind of the range over the threads of a group: `AxisType.LOCAL` on devices with thread groups, else `AxisType.WEAK`.

  Returns
  -------
    `(n_cells, n_channels)` grid.
  """
  device = str(values.device).split(":")[0]
  if device not in ATOMIC_ADD: raise NotImplementedError(f"no atomic add for {device}")
  if device == "CL" and values.dtype != dtypes.float32: raise NotImplementedError(f"the atomic add for OpenCL is for float32, got {values.dtype}")
  def spread_kernel(out:UOp, values:UOp, cell_offsets:UOp, kernel_weights:UOp) -> UOp:
    # one group more than needed: tinygrad merges the two ranges into one when the groups cover the points exactly, and the
    # threads are launched one at a time again
    n_points = values.shape[0]
    group, lane = UOp.range(n_points//GROUP_SIZE + 1, 0), UOp.range(GROUP_SIZE, 1, lane_axis)
    stencil = [UOp.range(cell_offsets.shape[2], 2+d, AxisType.LOOP) for d in range(cell_offsets.shape[0])]
    channel = UOp.range(values.shape[1], 2+len(stencil), AxisType.LOOP)
    point = (group*GROUP_SIZE + lane).clip(0, n_points-1)
    cell = functools.reduce(UOp.add, [cell_offsets[d, point, i].load().cast(dtypes.weakint) for d, i in enumerate(stencil)])
    weight = functools.reduce(UOp.mul, [kernel_weights[d, point, i].load() for d, i in enumerate(stencil)])
    value = (group*GROUP_SIZE + lane < n_points).where(values[point, channel].load() * weight, 0.0)
    atomic = UOp(Ops.CUSTOM, src=(out.index(cell, channel), value), arg=ATOMIC_ADD[device])
    return cast(UOp, atomic.end(group, lane, *stencil, channel).sink(arg=KernelInfo(name="spread", opts_to_apply=())))
  out = Tensor(Tensor.empty(n_cells, int(values.shape[1]), dtype=values.dtype, device=values.device).uop.custom_kernel(fxn=_zero_kernel)[0])
  return out.custom_kernel(values, cell_offsets, kernel_weights, fxn=spread_kernel)[0]

class Nufft:
  """Nonuniform fast Fourier transform between an image and its samples at arbitrary points.

  The forward transform (type 2 in FINUFFT) is `samples[j] = sum_k image[k] * exp(-1j * k @ points[j])`, with `k` running over
  `-(n//2), ..., (n-1)//2` in every image dimension of size `n`. `adjoint` (type 1) is its conjugate transpose. Neither is
  scaled. Each is the gradient of the other.
  """
  def __init__(self, points:Tensor, image_shape:Sequence[int], eps:float=1e-6):
    """Compute the interpolation tables for a set of points.

    Parameters
    ----------
    points
      Points in radians, shape `(..., n_points, n_dims)` with one column per image dimension. Points outside of `[-pi, pi)`
      wrap around. Dimensions in front are separate sets of points: they match the dimensions of the data directly in front
      of the image or sample dimensions. The transforms compute in the dtype of the points, at least float32.
    image_shape
      Size of the image dimensions.
    eps
      Requested relative accuracy, between 0 and 1. It sets the width of the kernel, and single precision reaches about 1e-6.
    """
    image_shape = tuple(operator.index(n) for n in image_shape)
    if not image_shape or min(image_shape) < 1: raise ValueError(f"expected positive image sizes, got {image_shape=}")
    if not 0 < eps < 1: raise ValueError(f"expected 0 < eps < 1, got {eps=}")
    if points.ndim < 2 or points.shape[-1] != len(image_shape):
      raise ValueError(f"expected points of shape (..., n_points, {len(image_shape)}), got {points.shape}")
    if not isinstance(device := points.device, str): raise NotImplementedError(f"tensors on several devices are not supported, got {device}")
    if not dtypes.is_float(points.dtype): points = points.cast(dtypes.default_float)
    if points.dtype.itemsize < 4: points = points.cast(dtypes.float32)
    self._image_shape, self._kernel_width, self._dtype = image_shape, min(max(math.ceil(-math.log10(eps)) + 1, 2), 16), points.dtype
    self._fine_shape = tuple(_fine_grid_size(n, self._kernel_width) for n in image_shape)
    self._batch_shape, self._n_points = tuple(points.shape[:-2]), int(points.shape[-2])
    n_sets, n_cells, n_dims, width = int(prod(self._batch_shape)), prod(self._fine_shape), len(image_shape), self._kernel_width
    if n_sets*n_cells >= 2**31: raise ValueError(f"the fine grids of shape {self._fine_shape} have too many cells for 32 bit indices")
    # per dimension the cells within half a width of a point, as offsets into the flattened grid, and the kernel at their distance.
    # a cell of the grid is the sum of the offsets over the dimensions, the grids of the sets of points follow each other.
    def per_dim(values:Sequence[int], dtype:DType=dtypes.int32) -> Tensor: return Tensor(list(values), dtype=dtype, device=device).reshape(-1, 1, 1)
    points = points.reshape(-1, n_dims).transpose(0, 1).unsqueeze(-1)
    position = (points - ((points + math.pi) / (2*math.pi)).floor() * (2*math.pi)) * (per_dim(self._fine_shape, points.dtype) / (2*math.pi))
    cell = (position - width/2).ceil() + Tensor.arange(width)
    distance = (cell - position) * (2/width)
    wrapped = (cell.cast(dtypes.int32) % per_dim(self._fine_shape) + per_dim(self._fine_shape)) % per_dim(self._fine_shape)
    first_cell = (Tensor.arange(n_sets*self._n_points) // max(self._n_points, 1) * n_cells).reshape(1, -1, 1) * per_dim([1] + [0]*(n_dims-1))
    strides = per_dim([prod(self._fine_shape[d+1:]) for d in range(n_dims)])
    self._cell_offsets = (wrapped * strides + first_cell).cast(dtypes.int32).contiguous().realize()
    self._kernel_weights = ((1 - distance*distance).maximum(0).sqrt() - 1).mul(BETA*width).exp().contiguous().realize()
    deapodization = Tensor.ones(*(1,)*n_dims, dtype=points.dtype, device=device)
    for d, (n, fine_size) in enumerate(zip(image_shape, self._fine_shape, strict=True)):
      factors = Tensor(_deapodization(n, fine_size, width), dtype=points.dtype, device=device)
      deapodization = deapodization * factors.reshape(n, *(1,)*(n_dims-d-1))
    self._deapodization = deapodization.unsqueeze(-1).contiguous().realize()
    # the transforms run as tinygrad functions, which must not use the device: everything they need from it is prepared here
    for n in self._fine_shape: prepare_tables(n, points.dtype, device)
    self._lane_axis = AxisType.LOCAL if Device[device].renderer.has_local else AxisType.WEAK

  def _stencil(self) -> tuple[Tensor, Tensor]:
    """Return the cells and weights of all contributions, `(n_points, kernel_width**n_dims)`, from the tables per dimension."""
    cells, weights = self._cell_offsets[0], self._kernel_weights[0]
    for d in range(1, len(self._image_shape)):
      cells = (cells.unsqueeze(2) + self._cell_offsets[d].unsqueeze(1)).reshape(cells.shape[0], -1)
      weights = (weights.unsqueeze(2) * self._kernel_weights[d].unsqueeze(1)).reshape(weights.shape[0], -1)
    return cells, weights

  def _pad_to_fine_grid(self, x:Tensor) -> Tensor:
    """Zero-pad a centered image to the fine grid, mode 0 first and the negative modes at the end."""
    for dim, (n, fine_size) in enumerate(zip(self._image_shape, self._fine_shape, strict=True), start=x.ndim-len(self._image_shape)-1):
      positive = x.shrink(_interval(x.ndim, dim, n//2, n)).pad(_interval(x.ndim, dim, 0, fine_size-n))
      x = positive.cat(x.shrink(_interval(x.ndim, dim, 0, n//2)), dim=dim)
    return x

  def _crop_from_fine_grid(self, x:Tensor) -> Tensor:
    """Take the centered image out of the fine grid, the inverse of `_pad_to_fine_grid`."""
    for dim, (n, fine_size) in enumerate(zip(self._image_shape, self._fine_shape, strict=True), start=x.ndim-len(self._image_shape)-1):
      x = x.shrink(_interval(x.ndim, dim, fine_size-n//2, fine_size)).cat(x.shrink(_interval(x.ndim, dim, 0, n-n//2)), dim=dim)
    return x

  def _forward(self, image:Tensor) -> Tensor:
    """Apply the forward transform to `(..., *batch_shape, *image_shape, 2)` in the dtype of the tables."""
    lead = image.shape[:image.ndim-len(self._batch_shape)-len(self._image_shape)-1]
    dims = tuple(range(image.ndim-len(self._image_shape)-1, image.ndim-1))
    # the padded image is a kernel of its own: fused, the first kernel of the transform is slow
    grid = fft_dims(self._pad_to_fine_grid(image * self._deapodization).contiguous(), dims, 1, False)
    # a gather is a load only from a buffer, with the cells in front
    grid = grid.reshape(int(prod(lead)), -1, 2).permute(1, 0, 2).contiguous()
    cells, weights = self._stencil()
    samples = (grid[cells] * weights.reshape(*weights.shape, 1, 1)).sum(1)
    return samples.permute(1, 0, 2).reshape(*lead, *self._batch_shape, self._n_points, 2)

  def _adjoint(self, samples:Tensor) -> Tensor:
    """Apply the adjoint transform to `(..., *batch_shape, n_points, 2)` in the dtype of the tables."""
    lead = samples.shape[:samples.ndim-len(self._batch_shape)-2]
    values = samples.reshape(int(prod(lead)), -1, 2).permute(1, 0, 2).reshape(-1, 2*int(prod(lead))).contiguous()
    n_cells = int(prod(self._batch_shape))*prod(self._fine_shape)
    grid = _spread(values, self._cell_offsets, self._kernel_weights, n_cells, self._lane_axis)
    grid = grid.reshape(-1, int(prod(lead)), 2).permute(1, 0, 2).reshape(*lead, *self._batch_shape, *self._fine_shape, 2)
    dims = tuple(range(grid.ndim-len(self._image_shape)-1, grid.ndim-1))
    return self._crop_from_fine_grid(fft_dims(grid, dims, 1, True)) * self._deapodization

  def _apply(self, x:Tensor, adjoint:bool) -> Tensor:
    """Check the data, run a transform with the other one as its gradient, and return the result in the dtype of the data."""
    source, target = ((self._n_points,), self._image_shape) if adjoint else (self._image_shape, (self._n_points,))
    expected = (*self._batch_shape, *source, 2)
    if x.ndim < len(expected) or x.shape[x.ndim-len(expected):] != expected: raise ValueError(f"expected a shape ending in {expected}, got {x.shape}")
    if not all(isinstance(s, int) for s in x.shape): raise ValueError(f"symbolic shapes are not supported, got {x.shape}")
    dtype = x.dtype if dtypes.is_float(x.dtype) else self._dtype
    if 0 in x.shape or self._n_points == 0: return Tensor.zeros(*x.shape[:x.ndim-len(source)-1], *target, 2, dtype=dtype, device=x.device)
    def backward(grad:UOp, call:UOp) -> tuple[UOp|None, ...]:
      # the inputs of the call are x and then the tables
      return (self._apply(Tensor(grad), not adjoint).uop, *[None]*(len(call.src)-2))
    return function(self._adjoint if adjoint else self._forward, allow_implicit=True, grad_fxn=backward)(x.cast(self._dtype)).cast(dtype)

  def __call__(self, image:Tensor) -> Tensor:
    """Sample the Fourier transform of an image at the points (type 2).

    Parameters
    ----------
    image
      Complex image of shape `(..., *image_shape, 2)`, with the dimensions of the sets of points directly in front of the image.

    Returns
    -------
      Complex samples of shape `(..., n_points, 2)`.
    """
    return self._apply(image, False)

  def adjoint(self, samples:Tensor) -> Tensor:
    """Apply the adjoint transform, from samples at the points to an image (type 1).

    It needs a device with an atomic add, see `ATOMIC_ADD`.

    Parameters
    ----------
    samples
      Complex samples of shape `(..., n_points, 2)`, with the dimensions of the sets of points directly in front of the samples.

    Returns
    -------
      Complex image of shape `(..., *image_shape, 2)`.
    """
    return self._apply(samples, True)
