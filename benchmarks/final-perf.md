| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal vs pure | nanobind vs pure | nanobind vs fast_metal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.767 | 0.198 | 0.153 | 3.86x | 5.01x | 1.30x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.442 | 0.315 | 0.194 | 1.40x | 2.29x | 1.63x |
| `na1d_k7_s1_d1_causal` | `forward` | 0.301 | 0.175 | 0.136 | 1.71x | 2.21x | 1.29x |
| `na1d_k7_s1_d1_causal` | `backward` | 0.342 | 0.312 | 0.193 | 1.09x | 1.77x | 1.62x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.592 | 0.284 | 0.244 | 5.60x | 6.52x | 1.16x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 2.858 | 0.536 | 0.432 | 5.33x | 6.61x | 1.24x |
| `na2d_k7x7_s1_d1_causal_h` | `forward` | 1.868 | 0.286 | 0.244 | 6.53x | 7.66x | 1.17x |
| `na2d_k7x7_s1_d1_causal_h` | `backward` | 2.040 | 0.487 | 0.433 | 4.19x | 4.71x | 1.12x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.910 | 0.191 | 0.198 | 4.76x | 4.58x | 0.96x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 1.038 | 0.353 | 0.245 | 2.94x | 4.23x | 1.44x |
| `na3d_k3x3x3_s1_d1_causal_d` | `forward` | 0.865 | 0.184 | 0.195 | 4.71x | 4.43x | 0.94x |
| `na3d_k3x3x3_s1_d1_causal_d` | `backward` | 1.023 | 0.343 | 0.252 | 2.98x | 4.06x | 1.36x |
