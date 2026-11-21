<img src="assets/icon.svg" width="64" height="64" alt="">

# tinyfft

FFT and nonuniform FFT for [tinygrad](https://github.com/tinygrad/tinygrad), written in tinygrad.

tinygrad has no complex dtype. A complex tensor here is a real tensor with a trailing dimension of size 2, the layout of
`torch.view_as_real(x)`. `dim` arguments do not count that dimension.

## FFT

```python
from tinygrad import Tensor
from tinyfft import fft, ifft, fftn, ifftn

x = Tensor.randn(4, 256, 256, 2)
k = fftn(x, dim=(-2, -1))
y = ifftn(k, dim=(-2, -1))
```

`dim` and `norm` work as in numpy and torch. Any length works: factors 2, 3, 5 and 7 are transformed directly, other
lengths go through Bluestein's algorithm. The transforms are lazy, run inside a `TinyJit` and have an explicit gradient.
Half precision input is computed in single precision.

## Nonuniform FFT

```python
from tinyfft import Nufft

op = Nufft(points, (256, 256), eps=1e-6)   # points: (m, d) in radians
samples = op(image)                        # (..., 256, 256, 2) -> (..., m, 2)
image = op.adjoint(samples)                # (..., m, 2) -> (..., 256, 256, 2)
```

`op(image)` evaluates `sum_k image[k] * exp(-1j * k @ points[j])` with centered `k` (type 2 in FINUFFT), `adjoint` is its
adjoint (type 1). Neither is scaled, and each is the gradient of the other. `eps` is the relative accuracy.

`points` can have dimensions in front: `(frames, m, d)` is one set of points per frame, for data of shape
`(coils, frames, y, x, 2)`.

For a trajectory that changes between calls, build the `Nufft` inside a `TinyJit` and pass the points as an input. Nothing
recompiles then.

The adjoint adds to the grid with atomic operations, which exist for CUDA, OpenCL and the CPU. Other devices can only
run the forward transform.

## Speed

RTX 4080, complex64, tinygrad 0.14. The ratio is tinyfft over the reference.

| FFT, `bench/bench.py` | tinyfft | torch | ratio |
|---|---|---|---|
| 256x256 | 0.134 ms | 0.059 ms | 2.3x |
| 256x256x256 | 2.02 ms | 1.27 ms | 1.6x |
| 8 times 256x256x256 | 15.3 ms | 10.6 ms | 1.4x |

| NUFFT, `bench/bench_nufft.py`, eps 1e-3 | tinyfft | cufinufft | ratio |
|---|---|---|---|
| 2D spiral, 256x256, 131k points, forward | 0.18 ms | 1.16 ms | 0.2x |
| 2D spiral, adjoint | 0.25 ms | 0.20 ms | 1.3x |
| 3D kooshball, 128x128x128, 1.9M points, forward | 6.1 ms | 2.7 ms | 2.3x |
| 3D kooshball, adjoint | 8.3 ms | 3.9 ms | 2.1x |

A jitted tinygrad call takes 0.11 ms before it does anything, which is most of the 256x256 time. The first call compiles
the kernels and takes 2 to 18 seconds. In 3D, points sorted by grid cell make the forward transform about twice as fast.
The kernels target GPUs; on a CPU the FFT was 15 to 35 times slower than single threaded torch.

## Development

```sh
uv sync                        # add --group bench for torch, finufft and cufinufft
DEV=CPU uv run pytest -n 12
uv run ruff check . && uv run mypy
DEV=CUDA uv run python bench/bench.py 256x256
```

tinygrad's CPU device needs `clang`; `CC=/path/to/clang` selects one. Its CUDA device needs an nvrtc that knows the GPU:
`NVRTC_PATH=.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12` is the one from the `bench` group.
The design notes are in the module docstrings of [`fft.py`](src/tinyfft/fft.py) and [`nufft.py`](src/tinyfft/nufft.py).
