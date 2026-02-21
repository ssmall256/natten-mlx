| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.488 | 0.194 | 0.193 | 2.51x | 2.53x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.611 | 0.663 | 0.670 | 0.92x | 0.91x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.723 | 0.693 | 0.696 | 2.49x | 2.48x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.099 | 2.398 | 2.406 | 0.88x | 0.87x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.842 | 0.313 | 0.314 | 2.69x | 2.68x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.979 | 1.937 | 1.981 | 0.51x | 0.49x |
