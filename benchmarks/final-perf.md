| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.615 | 0.211 | 0.201 | 2.92x | 3.06x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.483 | 0.302 | 0.303 | 1.60x | 1.59x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.691 | 0.667 | 0.664 | 2.53x | 2.55x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.875 | 0.558 | 0.561 | 3.36x | 3.34x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.862 | 0.305 | 0.302 | 2.82x | 2.85x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.959 | 0.370 | 0.374 | 2.59x | 2.57x |
