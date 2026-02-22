| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.838 | 0.173 | 0.174 | 4.85x | 4.80x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.919 | 0.279 | 0.649 | 3.29x | 1.42x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.703 | 0.172 | 0.175 | 4.09x | 4.01x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.874 | 0.272 | 0.591 | 3.21x | 1.48x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.722 | 0.260 | 0.294 | 6.63x | 5.86x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.905 | 0.469 | 0.905 | 4.07x | 2.10x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.421 | 0.260 | 0.392 | 5.47x | 3.62x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 1.870 | 0.479 | 1.808 | 3.90x | 1.03x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.834 | 0.184 | 0.286 | 4.53x | 2.91x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.981 | 0.316 | 0.697 | 3.10x | 1.41x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.862 | 0.171 | 0.177 | 5.05x | 4.86x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 0.946 | 0.326 | 0.765 | 2.90x | 1.24x |
