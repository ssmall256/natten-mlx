| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal vs pure | nanobind vs pure | nanobind vs fast_metal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.761 | 0.198 | 0.159 | 3.84x | 4.78x | 1.24x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.922 | 0.334 | 0.197 | 2.76x | 4.68x | 1.70x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.646 | 0.190 | 0.149 | 3.39x | 4.34x | 1.28x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.384 | 0.326 | 0.189 | 1.18x | 2.03x | 1.72x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.749 | 0.358 | 0.247 | 4.88x | 7.09x | 1.45x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.135 | 0.733 | 0.334 | 2.91x | 6.39x | 2.19x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.606 | 0.404 | 0.244 | 3.97x | 6.58x | 1.66x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 2.006 | 0.754 | 0.317 | 2.66x | 6.33x | 2.38x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.921 | 0.242 | 0.157 | 3.80x | 5.87x | 1.54x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.068 | 0.438 | 0.222 | 2.44x | 4.80x | 1.97x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.930 | 0.213 | 0.163 | 4.36x | 5.69x | 1.31x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 1.098 | 0.413 | 0.211 | 2.66x | 5.21x | 1.96x |
