| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.738 | 0.214 | 0.215 | 3.44x | 3.44x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 1.006 | 0.694 | 0.716 | 1.45x | 1.40x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.739 | 0.682 | 0.686 | 2.55x | 2.54x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.994 | 2.387 | 2.399 | 0.84x | 0.83x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.890 | 0.280 | 0.293 | 3.18x | 3.04x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.011 | 2.030 | 2.120 | 0.50x | 0.48x |
