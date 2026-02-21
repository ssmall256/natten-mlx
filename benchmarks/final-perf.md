| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.552 | 0.194 | 0.196 | 2.84x | 2.82x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.566 | 0.293 | 0.290 | 1.93x | 1.95x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 2.478 | 0.714 | 0.657 | 3.47x | 3.77x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.255 | 0.611 | 0.556 | 3.69x | 4.06x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.868 | 0.338 | 0.303 | 2.57x | 2.86x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.048 | 0.368 | 0.360 | 2.85x | 2.91x |
