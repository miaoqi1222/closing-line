# Walk-forward backtest: Elo vs closing line

Out-of-sample matches: 2660 (seasons 2019-20 to 2025-26); training window expands one season at a time from 3 seasons.
Bets: 2339, struck at Pinnacle pre-closing odds and settled there.

## Proper scores and calibration

| forecaster | log_loss | brier | ece | ece_h | ece_d | ece_a |
|---|---|---|---|---|---|---|
| base_rate | 1.0710 | 0.6486 | 0.0150 | 0.0225 | 0.0103 | 0.0122 |
| close | 0.9641 | 0.5718 | 0.0049 | 0.0189 | 0.0051 | 0.0141 |
| elo | 0.9884 | 0.5888 | 0.0209 | 0.0283 | 0.0150 | 0.0248 |

## Paired differences vs the closing line (negative = better than the close)

| forecaster | metric | value | ci_lo | ci_hi | n | spans_zero |
|---|---|---|---|---|---|---|
| elo - close | log_loss_diff | 0.0243 | 0.0162 | 0.0320 | 2660 | False |
| elo - close | brier_diff | 0.0170 | 0.0116 | 0.0223 | 2660 | False |
| base_rate - close | log_loss_diff | 0.1069 | 0.0904 | 0.1227 | 2660 | False |
| base_rate - close | brier_diff | 0.0768 | 0.0654 | 0.0876 | 2660 | False |

## Betting

| forecaster | metric | value | ci_lo | ci_hi | n | spans_zero |
|---|---|---|---|---|---|---|
| elo | n_bets | 2339.0000 | nan | nan | 2339 | False |
| elo | clv_mean | -0.0038 | -0.0076 | 0.0000 | 2339 | True |
| elo | ev_at_close_mean | -0.0385 | -0.0423 | -0.0349 | 2339 | False |
| elo | hit_rate | 0.3647 | 0.3459 | 0.3835 | 2339 | False |
| elo | roi_flat | 0.0053 | -0.0670 | 0.0803 | 2339 | True |
| elo | roi_kelly | -0.0079 | -0.0746 | 0.0582 | 2339 | True |
| elo | pnl_flat_total | 0.1241 | nan | nan | 2339 | False |
| elo | pnl_kelly_total | -0.2579 | nan | nan | 2339 | False |

## Log loss by season

| season | base_rate | close | elo |
|---|---|---|---|
| 2019-20 | 1.0657 | 0.9731 | 0.9814 |
| 2020-21 | 1.0905 | 0.9979 | 1.0439 |
| 2021-22 | 1.0696 | 0.9363 | 0.9551 |
| 2022-23 | 1.0508 | 0.9620 | 0.9786 |
| 2023-24 | 1.0535 | 0.8984 | 0.9284 |
| 2024-25 | 1.0818 | 0.9666 | 0.9877 |
| 2025-26 | 1.0851 | 1.0147 | 1.0438 |
