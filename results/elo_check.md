# Elo check (single split; the walk-forward backtest is stage 5)

Train: 2016-17 to 2018-19 (1140 matches). Test: 2019-20 to 2025-26 (2660 matches).

## Fitted on train only

| k | hfa | draw |
|---|---|---|
| 37.053 | 73.068 | 0.556 |

## Test scores

| forecaster | metric | value | ci_lo | ci_hi | n | spans_zero |
|---|---|---|---|---|---|---|
| elo | log_loss | 0.9917 | 0.9700 | 1.0130 | 2660 | False |
| elo | brier | 0.5908 | 0.5761 | 0.6056 | 2660 | False |
| elo | ece | 0.0280 | nan | nan | 2660 | nan |
| close_shin | log_loss | 0.9641 | 0.9461 | 0.9830 | 2660 | False |
| close_shin | brier | 0.5718 | 0.5593 | 0.5850 | 2660 | False |
| close_shin | ece | 0.0049 | nan | nan | 2660 | nan |
| base_rate | log_loss | 1.0723 | 1.0605 | 1.0842 | 2660 | False |
| base_rate | brier | 0.6496 | 0.6416 | 0.6578 | 2660 | False |
| base_rate | ece | 0.0269 | nan | nan | 2660 | nan |
| elo - close_shin | log_loss_diff | 0.0276 | 0.0189 | 0.0364 | 2660 | False |

## Final ratings, top and bottom five

| team | rating |
|---|---|
| Arsenal | 1824.7 |
| Man City | 1788.2 |
| Man United | 1730.9 |
| Bournemouth | 1690.0 |
| Aston Villa | 1668.7 |
| Watford | 1328.2 |
| Norwich | 1316.7 |
| Sheffield United | 1304.8 |
| Southampton | 1287.5 |
| Huddersfield | 1272.4 |
