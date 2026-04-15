import numpy as np
import pytest
from tinygrad import Device, Tensor, TinyJit, dtypes

from tinyfft import fft, fftn, ifft, ifftn
from tinyfft.fft import _plan_stages


def to_tensor(x:np.ndarray) -> Tensor:
  return Tensor(np.ascontiguousarray(x).view(x.real.dtype).reshape(*x.shape, 2))

def to_numpy(x:Tensor) -> np.ndarray:
  out = x.numpy()
  return out[..., 0] + 1j*out[..., 1]

def rand(*shape:int, dtype=np.complex64) -> np.ndarray:
  rng = np.random.default_rng(0)
  return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)).astype(dtype)

def assert_close(actual:Tensor, expected:np.ndarray, rtol:float=2e-6):
  assert actual.shape == (*expected.shape, 2)
  assert np.linalg.norm(to_numpy(actual) - expected) <= rtol * np.linalg.norm(expected)

# powers of two, mixed radix, a single stage, fused stages, primes (bluestein) and composites with large prime factors
SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 25, 30, 32, 49, 60, 64, 100, 128, 210, 256, 360, 1000, 1024, 11, 13, 17, 22, 97, 221, 1009]

@pytest.mark.parametrize("n", SIZES)
def test_fft(n:int):
  x = rand(3, n)
  assert_close(fft(to_tensor(x)), np.fft.fft(x.astype(np.complex128)))

@pytest.mark.parametrize("n", SIZES)
def test_ifft(n:int):
  x = rand(3, n)
  assert_close(ifft(to_tensor(x)), np.fft.ifft(x.astype(np.complex128)))

@pytest.mark.parametrize("n", [8192, 2**15, 3*2**14, 10**5, 2**17])
def test_fft_long(n:int):
  # more than one stage
  x = rand(n)
  assert_close(fft(to_tensor(x)), np.fft.fft(x.astype(np.complex128)))

def test_fft_long_batched():
  x = rand(2, 8192, 3)
  assert_close(fft(to_tensor(x), dim=1), np.fft.fft(x.astype(np.complex128), axis=1))

@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
def test_fft_dim(dim:int):
  x = rand(6, 8, 10)
  assert_close(fft(to_tensor(x), dim=dim), np.fft.fft(x.astype(np.complex128), axis=dim))
  assert_close(ifft(to_tensor(x), dim=dim), np.fft.ifft(x.astype(np.complex128), axis=dim))

@pytest.mark.parametrize("shape,dim", [((16, 16), None), ((8, 12, 10), None), ((4, 16, 20), (1, 2)), ((4, 16, 20), (-1, 0)),
                                       ((3, 5, 7, 9), (2,)), ((2, 3, 4, 5, 6), None), ((32, 11), None), ((64, 64, 64), None)])
def test_fftn(shape:tuple[int, ...], dim:tuple[int, ...]|None):
  x = rand(*shape)
  assert_close(fftn(to_tensor(x), dim=dim), np.fft.fftn(x.astype(np.complex128), axes=dim))
  assert_close(ifftn(to_tensor(x), dim=dim), np.fft.ifftn(x.astype(np.complex128), axes=dim))

@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_norm(norm:str):
  x = rand(12, 10)
  assert_close(fftn(to_tensor(x), norm=norm), np.fft.fftn(x.astype(np.complex128), norm=norm))
  assert_close(ifftn(to_tensor(x), norm=norm), np.fft.ifftn(x.astype(np.complex128), norm=norm))
  assert_close(fft(to_tensor(x), norm=norm), np.fft.fft(x.astype(np.complex128), norm=norm))
  assert_close(ifft(to_tensor(x), norm=norm), np.fft.ifft(x.astype(np.complex128), norm=norm))

@pytest.mark.parametrize("shape", [(64,), (30, 21), (8, 8, 8), (1009,)])
def test_roundtrip(shape:tuple[int, ...]):
  x = rand(*shape)
  assert_close(ifftn(fftn(to_tensor(x))), x)

def test_float64():
  if Device.DEFAULT.split(":")[0] in ("METAL", "WEBGPU"): pytest.skip("no float64 on this device")
  x = rand(4, 360, dtype=np.complex128)
  out = fft(to_tensor(x))
  assert out.dtype == dtypes.float64
  assert_close(out, np.fft.fft(x), rtol=1e-13)

def test_int_input():
  x = np.arange(32).reshape(16, 2)
  out = fft(Tensor(x))
  assert out.dtype == dtypes.default_float
  assert_close(out, np.fft.fft(x[:, 0] + 1j*x[:, 1]))

def test_non_contiguous_input():
  x = rand(8, 12, 16)
  assert_close(fftn(to_tensor(x).permute(2, 0, 1, 3)[::2]), np.fft.fftn(x.astype(np.complex128).transpose(2, 0, 1)[::2]))

def test_lazy():
  # nothing but the twiddle tables is computed until the result is realized
  out = fftn(Tensor.randn(16, 16, 2))
  assert not out.uop.is_realized

def test_jit():
  f = TinyJit(lambda x: fftn(x).realize())
  for i in range(4):
    x = rand(24, 32) * (i+1)
    assert_close(f(to_tensor(x).realize()), np.fft.fftn(x.astype(np.complex128)))

def test_gradient():
  # the transform is linear and unitary up to scaling: d/dx sum(|fft(x)|^2) = 2*n*x
  x = to_tensor(rand(4, 30))
  grad, = fft(x).square().sum().gradient(x)
  np.testing.assert_allclose(grad.numpy(), 2*30*x.numpy(), rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
@pytest.mark.parametrize("inverse", [False, True])
def test_gradient_adjoint(inverse:bool, norm:str):
  # the gradient of sum(w*f(x)) is the adjoint of f applied to w: the transform in the other direction with the same scale
  x, w, size = to_tensor(rand(6, 20)), rand(6, 20) + 0.5, 120
  grad, = ((ifftn if inverse else fftn)(x, norm=norm) * to_tensor(w)).sum().gradient(x)
  scale = {"backward": 1/size if inverse else 1, "ortho": size**-0.5, "forward": 1 if inverse else 1/size}[norm]
  other = np.fft.fftn(w.astype(np.complex128)) if inverse else np.fft.ifftn(w.astype(np.complex128)) * size
  assert_close(grad, other * scale)

def test_second_gradient():
  x = to_tensor(rand(4, 16))
  grad, = fft(x).square().sum().gradient(x)
  grad2, = grad.square().sum().gradient(x)
  np.testing.assert_allclose(grad2.numpy(), 2*(2*16)**2*x.numpy(), rtol=1e-4, atol=1e-3)

def test_errors():
  with pytest.raises(ValueError): fft(Tensor.zeros(4, 3))
  with pytest.raises(IndexError): fft(Tensor.zeros(4, 2), dim=1)
  with pytest.raises(ValueError): fftn(Tensor.zeros(4, 4, 2), dim=(0, -2))
  with pytest.raises(ValueError): fft(Tensor.zeros(4, 2), norm="unitary")  # type: ignore[arg-type]
  with pytest.raises(ValueError): fft(Tensor.zeros(0, 2))
  with pytest.raises(TypeError): fft(Tensor.zeros(4, 4, 2), dim=(0, 1))  # type: ignore[arg-type]
  with pytest.raises(TypeError): fftn(Tensor.zeros(4, 4, 2), dim=0.5)  # type: ignore[arg-type]

def test_dim_types():
  x = rand(6, 8)
  assert_close(fft(to_tensor(x), dim=np.int64(0)), np.fft.fft(x.astype(np.complex128), axis=0))
  assert_close(fftn(to_tensor(x), dim=[0, 1]), np.fft.fftn(x.astype(np.complex128)))  # type: ignore[arg-type]

def test_empty_batch():
  assert fft(Tensor.zeros(0, 4, 2)).shape == (0, 4, 2)
  assert fftn(Tensor.zeros(3, 0, 4, 2), dim=(0, 2)).shape == (3, 0, 4, 2)

@pytest.mark.parametrize("n", [64, 257, 1000])
def test_half_precision(n:int):
  # computed in single precision, the result is rounded to half precision
  x = (rand(2, n) / 8).astype(np.complex64)
  half = Tensor(np.ascontiguousarray(x).view(np.float32).reshape(2, n, 2).astype(np.float16))
  out = fft(half, norm="ortho")
  assert out.dtype == dtypes.float16
  quantized = half.numpy().astype(np.float64)
  assert_close(out, np.fft.fft(quantized[..., 0] + 1j*quantized[..., 1], norm="ortho"), rtol=2e-3)

@pytest.mark.parametrize("n", [2, 6, 64, 360, 1024, 3000])
def test_stages_cover(n:int):
  stages = _plan_stages(n)
  assert stages is not None and int(np.prod([direct*int(np.prod(butterflies)) for direct, butterflies, _ in stages])) == n
  assert all(direct <= 16 and int(np.prod(butterflies)) <= 24 for direct, butterflies, _ in stages)
  assert stages[-1][2] == 1

def test_plan_stages():
  assert _plan_stages(11) is None
  assert _plan_stages(256) == ((16, (4, 4), 1),)
  assert _plan_stages(192) == ((12, (4, 4), 1),)
  assert _plan_stages(384) == ((16, (4, 2, 3), 1),)
  assert _plan_stages(210) == ((10, (3, 7), 1),)
  assert _plan_stages(4096) == ((16, (4, 4), 16), (1, (4, 4), 1))
