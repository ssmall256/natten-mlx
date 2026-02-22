| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal vs pure | nanobind vs pure | nanobind vs fast_metal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.514 | 0.179 | 0.151 | 2.87x | 3.39x | 1.18x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.504 | 0.292 | 0.460 | 1.72x | 1.09x | 0.64x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.328 | 0.166 | 0.140 | 1.98x | 2.34x | 1.18x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.383 | 0.294 | 0.394 | 1.30x | 0.97x | 0.75x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.487 | 0.280 | 0.213 | 5.31x | 6.99x | 1.32x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.512 | 0.502 | 0.451 | 5.01x | 5.57x | 1.11x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.733 | 0.274 | 0.218 | 6.33x | 7.94x | 1.25x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 1.954 | 0.484 | 0.460 | 4.04x | 4.25x | 1.05x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.871 | 0.205 | 0.185 | 4.25x | 4.71x | 1.11x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.019 | 0.341 | 0.448 | 2.99x | 2.28x | 0.76x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.840 | 0.192 | 0.157 | 4.37x | 5.34x | 1.22x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 0.994 | 0.342 | 0.652 | 2.90x | 1.52x | 0.52x |
