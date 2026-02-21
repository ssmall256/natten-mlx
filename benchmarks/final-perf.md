| Case | Direction | pure (ms) | fast_metal (ms) | nanobind (ms) | fast_metal speedup vs pure | nanobind speedup vs pure |
|---|---:|---:|---:|---:|---:|---:|
| `na1d_k7_s1_d1_noncausal` | `forward` | 0.806 | 0.203 | 0.183 | 3.97x | 4.42x |
| `na1d_k7_s1_d1_noncausal` | `backward` | 0.866 | 0.317 | 0.315 | 2.73x | 2.75x |
| `na2d_k7x7_s1_d1_noncausal` | `forward` | 1.850 | 0.680 | 0.669 | 2.72x | 2.76x |
| `na2d_k7x7_s1_d1_noncausal` | `backward` | 1.830 | 0.743 | 0.729 | 2.46x | 2.51x |
| `na3d_k3x3x3_s1_d1_noncausal` | `forward` | 0.844 | 0.304 | 0.306 | 2.78x | 2.76x |
| `na3d_k3x3x3_s1_d1_noncausal` | `backward` | 0.985 | 0.389 | 0.387 | 2.53x | 2.55x |
