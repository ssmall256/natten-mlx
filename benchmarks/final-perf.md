| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.790 | 0.189 | 0.192 | 4.17x | 4.12x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.602 | 0.571 | 0.547 | 1.06x | 1.10x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.620 | 0.696 | 0.692 | 2.33x | 2.34x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.934 | 1.910 | 1.910 | 1.01x | 1.01x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.859 | 0.304 | 0.318 | 2.83x | 2.70x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.988 | 0.992 | 0.991 | 1.00x | 1.00x |
