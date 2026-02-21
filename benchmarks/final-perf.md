| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.448 | 0.203 | 0.211 | 2.21x | 2.12x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.512 | 0.543 | 0.519 | 0.94x | 0.99x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.670 | 0.661 | 0.673 | 2.53x | 2.48x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.980 | 1.858 | 1.861 | 1.07x | 1.06x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.856 | 0.309 | 0.298 | 2.77x | 2.87x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.997 | 0.951 | 0.954 | 1.05x | 1.04x |
