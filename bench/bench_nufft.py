"""
Compares tinyfft's Nufft with cufinufft on trajectories from MRI: `DEV=CUDA python bench/bench_nufft.py spiral2d koosh3d`.

Both get the trajectory once and then transform repeatedly: cufinufft with a plan and `setpts`, tinyfft with a `Nufft` in a
TinyJit. Forward is type 2 (image to samples), adjoint is type 1. Errors are relative to FINUFFT in double precision.
cufinufft from pip needs the CUDA libraries of the environment on the path:
`LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:.venv/lib/python3.12/site-packages/nvidia/cufft/lib`
"""
import argparse
import time
from collections.abc import Callable

import cufinufft
import finufft
import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit

from tinyfft import Nufft


def spiral2d(n:int, interleaves:int=32, samples:int=4096, turns:float=24) -> np.ndarray:
  """Variable density spiral interleaves out to the edge of k-space, `(interleaves*samples, 2)` in radians."""
  t = np.linspace(0, 1, samples, endpoint=False)
  angle = 2*np.pi*turns*t[None, :] + 2*np.pi*np.arange(interleaves)[:, None]/interleaves
  radius = np.pi * t[None, :]**1.5
  return np.stack([radius*np.cos(angle), radius*np.sin(angle)], axis=-1).reshape(-1, 2)

def koosh3d(n:int, spokes:int=30000) -> np.ndarray:
  """Spokes through the center of k-space with directions spread evenly over the sphere, `(spokes*n, 3)` in radians."""
  i = np.arange(spokes) + 0.5
  polar, azimuth = np.arccos(1 - i/spokes), np.pi*(1 + 5**0.5)*i   # half sphere: a spoke covers both directions
  direction = np.stack([np.sin(polar)*np.cos(azimuth), np.sin(polar)*np.sin(azimuth), np.cos(polar)], axis=-1)
  return (direction[:, None, :] * np.linspace(-np.pi, np.pi, n, endpoint=False)[None, :, None]).reshape(-1, 3)

CASES: dict[str, tuple[Callable[[int], np.ndarray], tuple[int, ...]]] = {
  "spiral2d": (spiral2d, (256, 256)), "spiral2d-384": (spiral2d, (384, 384)), "koosh3d-128": (lambda n: koosh3d(n, 15000), (128, 128, 128)),
  "koosh3d": (koosh3d, (192, 192, 192)),
}

def best(fxn:Callable[[], object], sync:Callable[[], object], reps:int) -> tuple[float, float]:
  """Time of the first call and of the fastest later one in ms."""
  times = []
  for _ in range(reps):
    sync()
    st = time.perf_counter()
    fxn()
    sync()
    times.append(time.perf_counter() - st)
  return times[0]*1e3, min(times[1:])*1e3

def to_complex(x:Tensor) -> np.ndarray:
  out = x.numpy()
  return out[..., 0] + 1j*out[..., 1]

def run(case:str, eps:float, reps:int) -> None:
  trajectory, shape = CASES[case]
  points = trajectory(shape[0]).astype(np.float32)
  m, d = points.shape
  rng = np.random.default_rng(0)
  image = (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)).astype(np.complex64)
  samples = (rng.standard_normal(m) + 1j*rng.standard_normal(m)).astype(np.complex64)
  coords = [points[:, i].astype(np.float64) for i in range(d)]
  type2, type1 = [(finufft.nufft2d2, finufft.nufft2d1), (finufft.nufft3d2, finufft.nufft3d1)][d-2]
  ref_forward = type2(*coords, image.astype(np.complex128), eps=1e-9, isign=-1, modeord=0)
  ref_adjoint = type1(*coords, samples.astype(np.complex128), shape, eps=1e-9, isign=1, modeord=0)
  def error(out:np.ndarray, ref:np.ndarray) -> float: return float(np.linalg.norm(out - ref) / np.linalg.norm(ref))

  # cufinufft goes first, tinygrad makes its own cuda context current
  sync = torch.cuda.synchronize
  xt, it, st_ = [torch.from_numpy(c.astype(np.float32)).cuda() for c in coords], torch.from_numpy(image).cuda(), torch.from_numpy(samples).cuda()
  rows = []
  for name, kind, sign, arg, ref in (("forward", 2, -1, it, ref_forward), ("adjoint", 1, 1, st_, ref_adjoint)):
    sync()
    start = time.perf_counter()
    plan = cufinufft.Plan(kind, shape, eps=eps, isign=sign, modeord=0, dtype="complex64")
    plan.setpts(*xt)
    sync()
    setup = (time.perf_counter() - start)*1e3
    _, t = best(lambda: plan.execute(arg), sync, reps)  # noqa: B023
    rows.append([name, setup, t, error(plan.execute(arg).cpu().numpy(), ref)])
    plan = None  # frees its GPU memory
  del xt, it, st_
  torch.cuda.empty_cache()

  sync = Device[Device.DEFAULT].synchronize
  pts = Tensor(points).realize()
  for row, (arg, ref) in zip(rows, ((image, ref_forward), (samples, ref_adjoint)), strict=True):
    sync()
    start = time.perf_counter()
    op = Nufft(pts, shape, eps=eps)
    sync()
    setup = (time.perf_counter() - start)*1e3
    x = Tensor(arg.view(np.float32).reshape(*arg.shape, 2)).realize()
    jit = TinyJit(lambda x: (op if row[0] == "forward" else op.adjoint)(x).realize())  # noqa: B023
    first, t = best(lambda: jit(x), sync, reps)  # noqa: B023
    row += [setup, first, t, error(to_complex(jit(x)), ref)]
  for name, c_setup, c_time, c_err, t_setup, t_first, t_time, t_err in rows:
    print(f"{case:>12} {m/1e6:6.2f}M {eps:7.0e} {name:>8} | cufinufft setup {c_setup:7.1f}ms run {c_time:8.3f}ms err {c_err:7.1e} | "
          f"tinyfft setup {t_setup:7.1f}ms first {t_first/1e3:5.1f}s run {t_time:8.3f}ms err {t_err:7.1e} | {t_time/c_time:5.1f}x")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("cases", nargs="*", default=["spiral2d", "koosh3d-128", "koosh3d"])
  parser.add_argument("--eps", type=float, nargs="*", default=[1e-3, 1e-5])
  parser.add_argument("--reps", type=int, default=10)
  args = parser.parse_args()
  for case in args.cases:
    for eps in args.eps: run(case, eps, args.reps)
