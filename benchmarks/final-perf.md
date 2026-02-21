| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.445 | 0.211 | 0.208 | 2.11x | 2.14x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.502 | 0.687 | 0.689 | 0.73x | 0.73x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.570 | 0.714 | 0.699 | 2.20x | 2.25x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.903 | 2.311 | 2.324 | 0.82x | 0.82x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.851 | 0.272 | 0.279 | 3.13x | 3.05x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.999 | 2.010 | 2.003 | 0.50x | 0.50x |
