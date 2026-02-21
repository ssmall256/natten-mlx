| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.602 | 0.214 | 0.204 | 2.81x | 2.96x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.702 | 0.342 | 0.320 | 2.05x | 2.20x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.743 | 0.701 | 0.696 | 2.49x | 2.50x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.031 | 0.776 | 0.734 | 2.62x | 2.77x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.886 | 0.321 | 0.321 | 2.76x | 2.76x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.994 | 0.401 | 0.403 | 2.48x | 2.47x |
