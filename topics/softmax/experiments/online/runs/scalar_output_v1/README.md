# Scalar output diagnostic v1

D = abs(error_chain) - abs(error_balanced).
Numerical interval midpoint approximations; no population CI or engineering threshold.
Rows below show FP32 at the largest saved block count. All cells/dtypes are in summary.json.

| n | spread | V probe | arm | FMA | mean D | changed outputs / families |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 8 | alternating_sign | fixed | None | 2.0487E-8 | 8/8 |
| 2048 | 8 | alternating_sign | online | False | 1.7165E-8 | 8/8 |
| 2048 | 8 | alternating_sign | online | True | 1.7855E-8 | 8/8 |
| 2048 | 8 | first_block | fixed | None | 6.5645E-11 | 8/8 |
| 2048 | 8 | first_block | online | False | 7.1418E-11 | 8/8 |
| 2048 | 8 | first_block | online | True | 6.8986E-11 | 7/8 |
| 2048 | 8 | max_block | fixed | None | 1.7812E-9 | 8/8 |
| 2048 | 8 | max_block | online | False | 1.7165E-9 | 8/8 |
| 2048 | 8 | max_block | online | True | 1.8232E-9 | 8/8 |
| 2048 | 8 | one | fixed | None | 0.0000E+4 | 0/8 |
| 2048 | 8 | one | online | False | 0.0000E+4 | 0/8 |
| 2048 | 8 | one | online | True | 0.0000E+4 | 0/8 |
| 2048 | 8 | zero | fixed | None | 0.0000E+4 | 0/8 |
| 2048 | 8 | zero | online | False | 0.0000E+4 | 0/8 |
| 2048 | 8 | zero | online | True | 0.0000E+4 | 0/8 |
| 2048 | 25 | alternating_sign | fixed | None | 1.1627E-7 | 8/8 |
| 2048 | 25 | alternating_sign | online | False | 1.1314E-7 | 8/8 |
| 2048 | 25 | alternating_sign | online | True | 1.1514E-7 | 8/8 |
| 2048 | 25 | first_block | fixed | None | 4.1445E-12 | 8/8 |
| 2048 | 25 | first_block | online | False | 3.6220E-12 | 8/8 |
| 2048 | 25 | first_block | online | True | 3.6220E-12 | 8/8 |
| 2048 | 25 | max_block | fixed | None | 1.2949E-8 | 8/8 |
| 2048 | 25 | max_block | online | False | 1.3872E-8 | 8/8 |
| 2048 | 25 | max_block | online | True | 1.3888E-8 | 8/8 |
| 2048 | 25 | one | fixed | None | 0.0000E+4 | 0/8 |
| 2048 | 25 | one | online | False | 0.0000E+4 | 0/8 |
| 2048 | 25 | one | online | True | 0.0000E+4 | 0/8 |
| 2048 | 25 | zero | fixed | None | 0.0000E+4 | 0/8 |
| 2048 | 25 | zero | online | False | 0.0000E+4 | 0/8 |
| 2048 | 25 | zero | online | True | 0.0000E+4 | 0/8 |

Original denominator roots, A and D replay exactly. Constant V=0/1 controls are exact in every arm/dtype.
The fixed arm uses the same globally prepared c_hat as the denominator ablation, then sums c_hat*V with ordinary FP32 adds.
Raw references, V bits, signed errors, division residuals and counterfactuals are in results.json.
Storage casts occur after FP32 division; no claim about actual GPU instructions or task impact.
