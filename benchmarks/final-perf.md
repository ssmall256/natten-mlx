| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.427 | 0.211 | 0.205 | 2.02x | 2.08x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.483 | 0.333 | 0.335 | 1.45x | 1.44x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.457 | 0.681 | 0.654 | 2.14x | 2.23x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.862 | 0.756 | 0.723 | 2.46x | 2.58x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.819 | 0.321 | 0.315 | 2.56x | 2.60x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.979 | 0.393 | 0.402 | 2.49x | 2.43x |
