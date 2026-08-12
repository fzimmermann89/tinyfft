import numpy as np
import pytest
from tinygrad import Tensor, TinyJit, dtypes

from tinyfft import Nufft


def to_tensor(x:np.ndarray) -> Tensor:
  return Tensor(np.ascontiguousarray(x.astype(np.complex64)).view(np.float32).reshape(*x.shape, 2))

def to_numpy(x:Tensor) -> np.ndarray:
  out = x.numpy()
  return out[..., 0] + 1j*out[..., 1]

def rand(*shape:int) -> np.ndarray:
  rng = np.random.default_rng(1)
  return rng.standard_normal(shape) + 1j*rng.standard_normal(shape)

def phases(points:np.ndarray, shape:tuple[int, ...]) -> np.ndarray:
  """exp(-1j * k @ x) for all points x and all centered modes k, with shape (points, *shape)."""
  modes = np.meshgrid(*[np.arange(-(n//2), (n+1)//2) for n in shape], indexing="ij")
  return np.exp(-1j * sum(points[:, d].reshape(-1, *(1,)*len(shape)) * k for d, k in enumerate(modes)))

def error(actual:Tensor, expected:np.ndarray) -> float:
  assert actual.shape == (*expected.shape, 2)
  return float(np.linalg.norm(to_numpy(actual) - expected) / np.linalg.norm(expected))

CASES = [((64,), 300), ((33,), 200), ((24, 30), 400), ((17, 20), 300), ((12, 10, 14), 500)]

@pytest.mark.parametrize("shape,m", CASES)
def test_forward(shape:tuple[int, ...], m:int):
  points, image = np.random.default_rng(0).uniform(-np.pi, np.pi, (m, len(shape))), rand(*shape)
  op = Nufft(Tensor(points.astype(np.float32)), shape, eps=1e-5)
  assert error(op(to_tensor(image)), np.tensordot(phases(points, shape), image, len(shape))) < 5e-5

@pytest.mark.parametrize("shape,m", CASES)
def test_adjoint(shape:tuple[int, ...], m:int):
  points, samples = np.random.default_rng(0).uniform(-np.pi, np.pi, (m, len(shape))), rand(m)
  op = Nufft(Tensor(points.astype(np.float32)), shape, eps=1e-5)
  assert error(op.adjoint(to_tensor(samples)), np.tensordot(samples, phases(points, shape).conj(), 1)) < 5e-5

@pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4, 1e-5])
def test_accuracy(eps:float):
  points, image = np.random.default_rng(0).uniform(-np.pi, np.pi, (500, 2)), rand(32, 32)
  op = Nufft(Tensor(points.astype(np.float32)), (32, 32), eps=eps)
  assert error(op(to_tensor(image)), np.tensordot(phases(points, (32, 32)), image, 2)) < 5*eps

def test_batch():
  points, image, samples = np.random.default_rng(0).uniform(-np.pi, np.pi, (300, 2)), rand(2, 3, 16, 20), rand(2, 3, 300)
  op = Nufft(Tensor(points.astype(np.float32)), (16, 20), eps=1e-5)
  phase = phases(points, (16, 20))
  assert error(op(to_tensor(image)), np.einsum("pxy,abxy->abp", phase, image)) < 5e-5
  assert error(op.adjoint(to_tensor(samples)), np.einsum("pxy,abp->abxy", phase.conj(), samples)) < 5e-5

def test_is_adjoint():
  # <A x, y> = <x, A^H y>
  points, image, samples = np.random.default_rng(0).uniform(-np.pi, np.pi, (400, 2)), rand(20, 24), rand(400)
  op = Nufft(Tensor(points.astype(np.float32)), (20, 24), eps=1e-5)
  left = np.vdot(samples, to_numpy(op(to_tensor(image))))
  right = np.vdot(to_numpy(op.adjoint(to_tensor(samples))), image)
  assert abs(left - right) < 1e-4 * abs(left)

def test_points_on_the_edge():
  # the kernel wraps around the grid
  points, image = np.array([[-np.pi, -np.pi], [np.pi - 1e-4, 0.0], [0.0, np.pi - 1e-4], [3.1, -3.1]]), rand(16, 16)
  op = Nufft(Tensor(points.astype(np.float32)), (16, 16), eps=1e-4)
  assert error(op(to_tensor(image)), np.tensordot(phases(points, (16, 16)), image, 2)) < 5e-4

def test_jit_with_new_points():
  # the points are an input: a new trajectory costs no recompilation
  @TinyJit
  def both(points:Tensor, image:Tensor, samples:Tensor) -> tuple[Tensor, Tensor]:
    op = Nufft(points, (16, 20), eps=1e-4)
    return op(image).realize(), op.adjoint(samples).realize()
  image, samples = rand(16, 20), rand(250)
  for seed in range(4):
    points = np.random.default_rng(seed).uniform(-np.pi, np.pi, (250, 2))
    forward, adjoint = both(Tensor(points.astype(np.float32)).realize(), to_tensor(image).realize(), to_tensor(samples).realize())
    assert error(forward, np.tensordot(phases(points, (16, 20)), image, 2)) < 5e-4
    assert error(adjoint, np.tensordot(samples, phases(points, (16, 20)).conj(), 1)) < 5e-4

def test_errors():
  with pytest.raises(ValueError): Nufft(Tensor.zeros(10, 3), (8, 8))
  with pytest.raises(ValueError): Nufft(Tensor.zeros(10), (8,))
  with pytest.raises(ValueError): Nufft(Tensor.zeros(10, 1), ())
  with pytest.raises(ValueError): Nufft(Tensor.zeros(10, 1), (0,))
  with pytest.raises(ValueError): Nufft(Tensor.zeros(10, 1), (8,), eps=0)
  with pytest.raises(ValueError): Nufft(Tensor.zeros(10, 1), (8,), eps=2)
  op = Nufft(Tensor.zeros(10, 2), (8, 6))
  with pytest.raises(ValueError): op(Tensor.zeros(1, 6, 2))       # would broadcast
  with pytest.raises(ValueError): op(Tensor.zeros(8, 6))
  with pytest.raises(ValueError): op.adjoint(Tensor.zeros(11, 2))  # would read past the tables
  with pytest.raises(ValueError): op.adjoint(Tensor.zeros(10, 3))

def test_gradient():
  # each transform is the gradient of the other
  points, image, samples = np.random.default_rng(0).uniform(-np.pi, np.pi, (300, 2)), rand(16, 20), rand(300)
  op, phase = Nufft(Tensor(points.astype(np.float32)), (16, 20), eps=1e-5), phases(points, (16, 20))
  x, y = to_tensor(image), to_tensor(samples)
  grad, = (op(x) * y).sum().gradient(x)  # sum(A x * y) is Re <y, A x>, its gradient is A^H y
  assert error(grad, np.tensordot(samples, phase.conj(), 1)) < 5e-5
  grad, = (op.adjoint(y) * x).sum().gradient(y)
  assert error(grad, np.tensordot(phase, image, 2)) < 5e-5

def test_gradient_of_a_loss():
  points, image = np.random.default_rng(0).uniform(-np.pi, np.pi, (200, 1)), rand(32)
  op, phase = Nufft(Tensor(points.astype(np.float32)), (32,), eps=1e-5), phases(points, (32,))
  x = to_tensor(image)
  grad, = op(x).square().sum().gradient(x)  # d/dx |A x|^2 = 2 A^H A x
  assert error(grad, 2 * phase.conj().T @ (phase @ image)) < 1e-4

def test_sets_of_points():
  # one set of points per frame, the same for all coils: (coils, frames, y, x) <-> (coils, frames, n_points)
  rng = np.random.default_rng(0)
  points, image, samples = rng.uniform(-np.pi, np.pi, (3, 150, 2)), rand(2, 3, 12, 16), rand(2, 3, 150)
  op = Nufft(Tensor(points.astype(np.float32)), (12, 16), eps=1e-5)
  phase = np.stack([phases(p, (12, 16)) for p in points])
  assert error(op(to_tensor(image)), np.einsum("fpyx,cfyx->cfp", phase, image)) < 5e-5
  assert error(op.adjoint(to_tensor(samples)), np.einsum("fpyx,cfp->cfyx", phase.conj(), samples)) < 5e-5

def test_points_wrap_around():
  points, image = np.random.default_rng(0).uniform(-np.pi, np.pi, (100, 2)), rand(16, 16)
  shifted = points + 2*np.pi*np.array([3, -2])
  op = Nufft(Tensor(shifted.astype(np.float32)), (16, 16), eps=1e-4)
  assert error(op(to_tensor(image)), np.tensordot(phases(points, (16, 16)), image, 2)) < 5e-4

def test_dtypes():
  points, image, samples = np.random.default_rng(0).uniform(-np.pi, np.pi, (100, 1)), rand(16), rand(100)
  phase = phases(points, (16,))
  # integer data is cast to the dtype of the tables, half precision is computed in single precision and cast back
  ints = np.round(image.real*4) + 1j*np.round(image.imag*4)
  op = Nufft(Tensor(points.astype(np.float32)), (16,), eps=1e-4)
  assert error(op(Tensor(np.stack([ints.real, ints.imag], -1).astype(np.int32))), phase @ ints) < 5e-4
  ints = np.round(samples.real*4) + 1j*np.round(samples.imag*4)
  assert error(op.adjoint(Tensor(np.stack([ints.real, ints.imag], -1).astype(np.int32))), ints @ phase.conj()) < 5e-4
  half = op(to_tensor(image).cast(dtypes.float16))
  assert half.dtype == dtypes.float16 and error(half, phase @ image) < 5e-3
  assert error(Nufft(Tensor(points.astype(np.float16)), (16,), eps=1e-2)(to_tensor(image)), phase @ image) < 5e-2

def test_empty():
  op = Nufft(Tensor.zeros(10, 2), (8, 6))
  assert op(Tensor.zeros(0, 8, 6, 2)).shape == (0, 10, 2)
  assert op.adjoint(Tensor.zeros(0, 10, 2)).shape == (0, 8, 6, 2)
  op = Nufft(Tensor.zeros(0, 2), (8, 6))
  assert op(Tensor.zeros(3, 8, 6, 2)).shape == (3, 0, 2)
  assert np.all(op.adjoint(Tensor.zeros(3, 0, 2)).numpy() == 0)
