"""
Compares tinyfft with torch.fft: `DEV=CUDA python bench/bench.py 256x256 64*256x256`.

A case is the shape of a complex64 tensor that is transformed over all dimensions, a `64*` in front adds a batch dimension.
tinyfft runs in a TinyJit and the result is realized. The first call includes computing the twiddles and compiling the kernels.
No `.contiguous()` on the result: in tinygrad 0.14 that is a copy even when the buffer already is contiguous, as it is on GPUs.
"""
import argparse
import time
from collections.abc import Callable

import numpy as np
import torch
from tinygrad import Device, Tensor, TinyJit

from tinyfft import fftn

CASES = ["256x256", "64*256x256", "1024x1024", "256x256x256", "4096*256", "1048576", "1000x1000", "1009"]

def best(fxn:Callable[[], object], sync:Callable[[], object], reps:int) -> tuple[float, float]:
  # time of the first call and of the fastest later one in ms
  times = []
  for _ in range(reps):
    sync()
    st = time.perf_counter()
    fxn()
    sync()
    times.append(time.perf_counter() - st)
  return times[0]*1e3, min(times[2:])*1e3

def run(case:str, cuda:bool, reps:int, floor:float) -> str:
  batch, _, dims = case.rpartition("*")
  shape = tuple(int(s) for s in dims.split("x"))
  dim = tuple(range(-len(shape), 0))
  if batch: shape = (int(batch), *shape)
  rng = np.random.default_rng(0)
  data = (rng.standard_normal(shape) + 1j*rng.standard_normal(shape)).astype(np.complex64)
  ref = np.fft.fftn(data.astype(np.complex128), axes=dim)

  def error(out:np.ndarray) -> float: return float(np.linalg.norm(out - ref) / np.linalg.norm(ref))

  # torch goes first, tinygrad makes its own cuda context current
  xt = torch.from_numpy(data).to("cuda" if cuda else "cpu")
  _, t_torch = best(lambda: torch.fft.fftn(xt, dim=dim), torch.cuda.synchronize if cuda else lambda: None, reps)
  err_torch = error(torch.fft.fftn(xt, dim=dim).cpu().numpy())

  x = Tensor(data.view(np.float32).reshape(*shape, 2)).realize()
  jit = TinyJit(lambda x: fftn(x, dim=dim).realize())
  first, t_tiny = best(lambda: jit(x), Device[Device.DEFAULT].synchronize, reps)
  out = jit(x).numpy()
  err_tiny = error(out[..., 0] + 1j*out[..., 1])
  return (f"{case:>14} {first:9.0f}ms {t_tiny:8.3f}ms {t_torch:8.3f}ms {t_tiny/t_torch:6.1f}x {(t_tiny-floor)/t_torch:10.1f}x "
          f"{err_tiny:8.1e} {err_torch:9.1e}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("cases", nargs="*", default=CASES)
  parser.add_argument("--reps", type=int, default=30)
  parser.add_argument("--threads", type=int, default=1, help="torch cpu threads, tinygrad runs on one")
  args = parser.parse_args()

  cuda = Device.DEFAULT.split(":")[0] in ("CUDA", "NV")
  torch.set_num_threads(args.threads)
  print(f"tinygrad {Device.DEFAULT} vs torch {'cuda' if cuda else f'cpu, {args.threads} thread(s)'}")
  # the time of a jitted call that does next to nothing, tinyfft cannot be faster than this
  small = Tensor.zeros(2).contiguous().realize()
  noop = TinyJit(lambda x: (x+1).realize())
  floor = best(lambda: noop(small), Device[Device.DEFAULT].synchronize, 200)[1]
  print(f"an empty jitted tinygrad call takes {floor:.3f}ms, 'above call' is the ratio without it")
  print(f"{'case':>14} {'first call':>11} {'tinyfft':>10} {'torch':>10} {'ratio':>7} {'above call':>11} {'error':>8} {'torch err':>9}")
  for case in args.cases: print(run(case, cuda, args.reps, floor))
