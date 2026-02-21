| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.427 | 0.213 | 0.182 | 2.00x | 2.34x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.496 | 0.291 | 0.307 | 1.71x | 1.62x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.371 | 0.196 | 0.195 | 1.89x | 1.90x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.434 | 0.296 | 0.285 | 1.47x | 1.52x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.673 | 0.679 | 0.676 | 2.46x | 2.47x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.993 | 0.570 | 0.557 | 3.49x | 3.58x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.502 | 0.698 | 0.691 | 2.15x | 2.17x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 1.963 | 0.574 | 0.581 | 3.42x | 3.38x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.870 | 0.308 | 0.309 | 2.82x | 2.82x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.073 | 0.352 | 0.366 | 3.05x | 2.93x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.857 | 0.296 | 0.287 | 2.89x | 2.98x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 0.993 | 0.368 | 0.357 | 2.70x | 2.78x |
