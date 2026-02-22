| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.840 | 0.195 | 0.163 | 4.30x | 5.16x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.934 | 0.292 | 0.223 | 3.20x | 4.19x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.655 | 0.184 | 0.153 | 3.56x | 4.28x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.834 | 0.300 | 0.236 | 2.78x | 3.53x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.703 | 0.282 | 0.239 | 6.04x | 7.13x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.062 | 0.544 | 0.496 | 3.79x | 4.16x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.472 | 0.273 | 0.259 | 5.39x | 5.69x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 1.883 | 0.535 | 0.495 | 3.52x | 3.80x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.904 | 0.198 | 0.210 | 4.56x | 4.31x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.035 | 0.352 | 0.316 | 2.94x | 3.27x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.851 | 0.195 | 0.217 | 4.35x | 3.92x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 1.000 | 0.346 | 0.307 | 2.89x | 3.25x |
