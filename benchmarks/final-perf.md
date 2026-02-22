| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.435 | 0.181 | 0.188 | 2.40x | 2.32x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.500 | 0.283 | 0.303 | 1.77x | 1.65x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.360 | 0.164 | 0.176 | 2.19x | 2.05x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.428 | 0.286 | 0.309 | 1.50x | 1.38x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.782 | 0.260 | 0.280 | 6.85x | 6.37x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.110 | 0.484 | 0.498 | 4.36x | 4.24x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.425 | 0.270 | 0.282 | 5.27x | 5.06x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 1.813 | 0.482 | 0.501 | 3.76x | 3.62x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.823 | 0.190 | 0.196 | 4.34x | 4.21x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.959 | 0.328 | 0.353 | 2.92x | 2.72x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.818 | 0.191 | 0.187 | 4.28x | 4.37x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 0.942 | 0.337 | 0.363 | 2.80x | 2.59x |
