## The checker's own pass rates and margin

A plan admitted but left with no verdict, at roomAppendices on its orthogonality precondition (section 9.4), is counted here as a precondition stop, so "reached a verdict" below subtracts it from "admitted" rather than treating the two as equal (see this function's docstring).

### Denominator: graded plans

| source | passed | graded plans | pass rate | 95 per cent interval |
|---|---|---|---|---|
| RPLAN ground truth | 3,650 | 4,000 | 91.2 | 90.3 to 92.1 |
| MSD ground truth, units path | 10,079 | 14,323 | 70.4 | 69.6 to 71.1 |
| House-GAN++ | 2,325 | 4,000 | 58.1 | 56.6 to 59.6 |
| House-GAN | 1,945 | 4,000 | 48.6 | 47.1 to 50.2 |
| HouseDiffusion | 44 | 3,991 | 1.1 | 0.8 to 1.5 |

### Denominator: admitted plans

| source | passed | admitted plans | pass rate | 95 per cent interval |
|---|---|---|---|---|
| RPLAN ground truth | 3,650 | 3,998 | 91.3 | 90.4 to 92.1 |
| MSD ground truth, units path | 10,079 | 14,206 | 70.9 | 70.2 to 71.7 |
| House-GAN++ | 2,325 | 3,557 | 65.4 | 63.8 to 66.9 |
| House-GAN | 1,945 | 3,556 | 54.7 | 53.1 to 56.3 |
| HouseDiffusion | 44 | 1,189 | 3.7 | 2.8 to 4.9 |

### Denominator: reached a verdict

| source | passed | reached a verdict | pass rate | 95 per cent interval |
|---|---|---|---|---|
| RPLAN ground truth | 3,650 | 3,998 | 91.3 | 90.4 to 92.1 |
| MSD ground truth, units path | 10,079 | 11,331 | 89.0 | 88.4 to 89.5 |
| House-GAN++ | 2,325 | 3,557 | 65.4 | 63.8 to 66.9 |
| House-GAN | 1,945 | 3,556 | 54.7 | 53.1 to 56.3 |
| HouseDiffusion | 44 | 1,071 | 4.1 | 3.1 to 5.5 |

| denominator | pair | margin | bootstrap, fixed pair | Newcombe, fixed pair | bootstrap, re-selected |
|---|---|---|---|---|---|
| graded plans | MSD ground truth, units path over House-GAN++ | +12.2 | +10.5 to +14.0 | +10.5 to +14.0 | +10.5 to +13.9 |
| admitted plans | MSD ground truth, units path over House-GAN++ | +5.6 | +3.9 to +7.3 | +3.9 to +7.3 | +3.9 to +7.3 |
| reached a verdict | MSD ground truth, units path over House-GAN++ | +23.6 | +21.9 to +25.2 | +21.9 to +25.3 | +21.9 to +25.3 |
