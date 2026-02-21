| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.566 | 0.210 | 0.207 | 2.70x | 2.74x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.634 | 0.313 | 0.297 | 2.03x | 2.13x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.932 | 0.663 | 0.671 | 2.92x | 2.88x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.841 | 0.605 | 0.560 | 4.69x | 5.08x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.960 | 0.312 | 0.298 | 3.08x | 3.23x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.059 | 0.371 | 0.354 | 2.85x | 2.99x |
