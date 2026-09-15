# Harness check: closing line vs itself and vs base rate

Matches scored: 3250 (seasons 2017-18 to 2025-26; 2016-17 is used only to seed the base-rate prior).
Bets are struck at ps pre-closing odds and settled at that price.

## Proper scores and calibration

| forecaster | log_loss | brier | ece | ece_h | ece_d | ece_a |
|---|---|---|---|---|---|---|
| base_rate | 1.0663 | 0.6453 | 0.0130 | 0.0196 | 0.0035 | 0.0160 |
| close_additive | 0.9483 | 0.5608 | 0.0065 | 0.0200 | 0.0085 | 0.0120 |
| close_multiplicative | 0.9484 | 0.5609 | 0.0058 | 0.0188 | 0.0090 | 0.0110 |
| close_shin | 0.9482 | 0.5608 | 0.0068 | 0.0195 | 0.0088 | 0.0102 |
| open_shin | 0.9520 | 0.5635 | 0.0071 | 0.0216 | 0.0074 | 0.0139 |

## Paired differences vs close_shin (negative = better than the closing line)

| forecaster | metric | value | ci_lo | ci_hi | n | spans_zero |
|---|---|---|---|---|---|---|
| close_multiplicative - close_shin | log_loss_diff | 0.0001 | -0.0004 | 0.0006 | 3250 | True |
| close_multiplicative - close_shin | brier_diff | 0.0001 | -0.0002 | 0.0003 | 3250 | True |
| close_additive - close_shin | log_loss_diff | 0.0000 | -0.0002 | 0.0002 | 3250 | True |
| close_additive - close_shin | brier_diff | -0.0000 | -0.0001 | 0.0001 | 3250 | True |
| open_shin - close_shin | log_loss_diff | 0.0038 | 0.0012 | 0.0061 | 3250 | False |
| open_shin - close_shin | brier_diff | 0.0026 | 0.0010 | 0.0042 | 3250 | False |
| base_rate - close_shin | log_loss_diff | 0.1181 | 0.1019 | 0.1333 | 3250 | False |
| base_rate - close_shin | brier_diff | 0.0844 | 0.0734 | 0.0948 | 3250 | False |

## Betting metrics

| forecaster | metric | value | ci_lo | ci_hi | n | spans_zero |
|---|---|---|---|---|---|---|
| close_shin | n_bets | 2433.0000 | nan | nan | 2433 | False |
| close_shin | bet_rate | 0.7486 | nan | nan | 3250 | False |
| close_shin | clv_mean | 0.0847 | 0.0820 | 0.0874 | 2433 | False |
| close_shin | ev_at_close_mean | 0.0539 | 0.0515 | 0.0562 | 2433 | False |
| close_shin | hit_rate | 0.3938 | 0.3740 | 0.4139 | 2433 | False |
| close_shin | roi_flat | 0.0506 | -0.0142 | 0.1172 | 2433 | True |
| close_shin | roi_kelly | 0.1112 | 0.0454 | 0.1755 | 2433 | False |
| close_shin | pnl_flat_total | 1.2319 | nan | nan | 2433 | False |
| close_shin | pnl_kelly_total | 1.8908 | nan | nan | 2433 | False |
| close_shin | mean_odds_taken | 3.7222 | nan | nan | 2433 | False |
| close_multiplicative | n_bets | 2451.0000 | nan | nan | 2451 | False |
| close_multiplicative | bet_rate | 0.7542 | nan | nan | 3250 | False |
| close_multiplicative | clv_mean | 0.0864 | 0.0838 | 0.0894 | 2451 | False |
| close_multiplicative | ev_at_close_mean | 0.0511 | 0.0488 | 0.0537 | 2451 | False |
| close_multiplicative | hit_rate | 0.3529 | 0.3354 | 0.3717 | 2451 | False |
| close_multiplicative | roi_flat | 0.0712 | -0.0001 | 0.1515 | 2451 | True |
| close_multiplicative | roi_kelly | 0.1115 | 0.0417 | 0.1880 | 2451 | False |
| close_multiplicative | pnl_flat_total | 1.7451 | nan | nan | 2451 | False |
| close_multiplicative | pnl_kelly_total | 1.6939 | nan | nan | 2451 | False |
| close_multiplicative | mean_odds_taken | 4.3592 | nan | nan | 2451 | False |
| close_additive | n_bets | 2437.0000 | nan | nan | 2437 | False |
| close_additive | bet_rate | 0.7498 | nan | nan | 3250 | False |
| close_additive | clv_mean | 0.0832 | 0.0805 | 0.0861 | 2437 | False |
| close_additive | ev_at_close_mean | 0.0536 | 0.0514 | 0.0561 | 2437 | False |
| close_additive | hit_rate | 0.4091 | 0.3906 | 0.4300 | 2437 | False |
| close_additive | roi_flat | 0.0679 | 0.0038 | 0.1382 | 2437 | False |
| close_additive | roi_kelly | 0.1101 | 0.0479 | 0.1760 | 2437 | False |
| close_additive | pnl_flat_total | 1.6555 | nan | nan | 2437 | False |
| close_additive | pnl_kelly_total | 1.9624 | nan | nan | 2437 | False |
| close_additive | mean_odds_taken | 3.5498 | nan | nan | 2437 | False |
| open_shin | n_bets | 0.0000 | nan | nan | 0 | False |
| open_shin | bet_rate | 0.0000 | nan | nan | 3250 | False |
| base_rate | n_bets | 3248.0000 | nan | nan | 3248 | False |
| base_rate | bet_rate | 0.9994 | nan | nan | 3250 | False |
| base_rate | clv_mean | -0.0034 | -0.0074 | 0.0005 | 3248 | True |
| base_rate | ev_at_close_mean | -0.0493 | -0.0535 | -0.0454 | 3248 | False |
| base_rate | hit_rate | 0.2294 | 0.2152 | 0.2445 | 3248 | False |
| base_rate | roi_flat | -0.0189 | -0.0915 | 0.0553 | 3248 | True |
| base_rate | roi_kelly | -0.0230 | -0.0990 | 0.0543 | 3248 | True |
| base_rate | pnl_flat_total | -0.6123 | nan | nan | 3248 | False |
| base_rate | pnl_kelly_total | -1.4181 | nan | nan | 3248 | False |
| base_rate | mean_odds_taken | 5.9243 | nan | nan | 3248 | False |

## Closing line reliability (pooled over outcomes)

| bin_lo | bin_hi | n | mean_pred | mean_obs |
|---|---|---|---|---|
| 0.0000 | 0.1000 | 553 | 0.0689 | 0.0705 |
| 0.1000 | 0.2000 | 1645 | 0.1553 | 0.1593 |
| 0.2000 | 0.3000 | 3390 | 0.2564 | 0.2519 |
| 0.3000 | 0.4000 | 1394 | 0.3414 | 0.3415 |
| 0.4000 | 0.5000 | 956 | 0.4465 | 0.4383 |
| 0.5000 | 0.6000 | 721 | 0.5512 | 0.5631 |
| 0.6000 | 0.7000 | 481 | 0.6457 | 0.6632 |
| 0.7000 | 0.8000 | 397 | 0.7456 | 0.7254 |
| 0.8000 | 0.9000 | 194 | 0.8408 | 0.8866 |
| 0.9000 | 1.0000 | 19 | 0.9138 | 0.7895 |

## Pre-closing to closing line movement (benchmark book)

| outcome | n | mean_log_move | mean_abs_log_move | p95_abs_log_move | mean_abs_prob_move | share_unchanged |
|---|---|---|---|---|---|---|
| home | 3250 | -0.0088 | 0.0568 | 0.1626 | 0.0208 | 0.0412 |
| draw | 3250 | 0.0021 | 0.0447 | 0.1312 | 0.0100 | 0.0249 |
| away | 3250 | -0.0054 | 0.0722 | 0.1993 | 0.0189 | 0.0215 |
