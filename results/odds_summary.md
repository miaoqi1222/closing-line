# Odds summary

Benchmark: `ps` closing odds, devigged with Shin's method; consensus of the other individual books where the benchmark book is absent.

## Overround, pooled across seasons (fraction, e.g. 0.025 = 2.5%)

| book | stage | seasons | n | mean | min_season_mean | max_season_mean |
|---|---|---|---|---|---|---|
| b365 | close | 7 | 2660 | 0.0548 | 0.0539 | 0.0559 |
| b365 | open | 10 | 3800 | 0.0468 | 0.0278 | 0.0553 |
| bw | close | 7 | 2507 | 0.0537 | 0.0445 | 0.0586 |
| bw | open | 10 | 3657 | 0.0532 | 0.0497 | 0.0578 |
| iw | close | 5 | 1715 | 0.0516 | 0.0499 | 0.0535 |
| iw | open | 8 | 2858 | 0.0534 | 0.0497 | 0.0591 |
| ps | close | 10 | 3630 | 0.0253 | 0.0205 | 0.0299 |
| ps | open | 10 | 3630 | 0.0274 | 0.0205 | 0.0360 |
| vc | close | 5 | 1900 | 0.0474 | 0.0389 | 0.0612 |
| vc | open | 8 | 3040 | 0.0470 | 0.0261 | 0.0690 |
| wh | close | 6 | 2189 | 0.0582 | 0.0482 | 0.0732 |
| wh | open | 9 | 3329 | 0.0556 | 0.0469 | 0.0761 |

Per-season detail with percentiles: `odds_overround.csv`.

## Devig method disagreement on closing odds (absolute probability difference)

| book | pair | n | mean_abs_diff_h | mean_abs_diff_d | mean_abs_diff_a | max_abs_diff | mean_abs_diff_favourite | mean_abs_diff_longshot |
|---|---|---|---|---|---|---|---|---|
| ps | multiplicative vs shin | 3630 | 0.0036 | 0.0019 | 0.0030 | 0.0176 | 0.0042 | 0.0026 |
| ps | multiplicative vs additive | 3630 | 0.0049 | 0.0025 | 0.0041 | 0.0236 | 0.0057 | 0.0035 |
| ps | additive vs shin | 3630 | 0.0012 | 0.0006 | 0.0010 | 0.0060 | 0.0014 | 0.0009 |
| b365 | multiplicative vs shin | 2660 | 0.0072 | 0.0039 | 0.0060 | 0.0259 | 0.0085 | 0.0051 |
| b365 | multiplicative vs additive | 2660 | 0.0097 | 0.0052 | 0.0081 | 0.0349 | 0.0114 | 0.0069 |
| b365 | additive vs shin | 2660 | 0.0025 | 0.0013 | 0.0021 | 0.0090 | 0.0029 | 0.0018 |
| bw | multiplicative vs shin | 2507 | 0.0066 | 0.0036 | 0.0056 | 0.0243 | 0.0079 | 0.0047 |
| bw | multiplicative vs additive | 2507 | 0.0090 | 0.0049 | 0.0075 | 0.0327 | 0.0106 | 0.0065 |
| bw | additive vs shin | 2507 | 0.0023 | 0.0012 | 0.0020 | 0.0085 | 0.0027 | 0.0017 |
| iw | multiplicative vs shin | 1715 | 0.0064 | 0.0034 | 0.0054 | 0.0236 | 0.0076 | 0.0046 |
| iw | multiplicative vs additive | 1715 | 0.0087 | 0.0045 | 0.0073 | 0.0318 | 0.0102 | 0.0062 |
| iw | additive vs shin | 1715 | 0.0022 | 0.0011 | 0.0019 | 0.0082 | 0.0026 | 0.0017 |
| wh | multiplicative vs shin | 2189 | 0.0077 | 0.0041 | 0.0065 | 0.0456 | 0.0091 | 0.0055 |
| wh | multiplicative vs additive | 2189 | 0.0105 | 0.0054 | 0.0088 | 0.0622 | 0.0123 | 0.0075 |
| wh | additive vs shin | 2189 | 0.0027 | 0.0013 | 0.0023 | 0.0166 | 0.0032 | 0.0020 |
| vc | multiplicative vs shin | 1900 | 0.0064 | 0.0033 | 0.0054 | 0.0326 | 0.0075 | 0.0046 |
| vc | multiplicative vs additive | 1900 | 0.0086 | 0.0043 | 0.0073 | 0.0441 | 0.0101 | 0.0062 |
| vc | additive vs shin | 1900 | 0.0022 | 0.0011 | 0.0019 | 0.0115 | 0.0026 | 0.0017 |

## Shin minus multiplicative, Pinnacle closing, by favourite strength

| fav_bin | n | shin_minus_mult_favourite | shin_minus_mult_longshot |
|---|---|---|---|
| [0.3, 0.4) | 552 | 0.0007 | -0.0007 |
| [0.4, 0.5) | 1064 | 0.0020 | -0.0013 |
| [0.5, 0.6) | 822 | 0.0042 | -0.0025 |
| [0.6, 0.7) | 547 | 0.0062 | -0.0038 |
| [0.7, 0.8) | 447 | 0.0086 | -0.0050 |
| [0.8, 0.9) | 188 | 0.0108 | -0.0059 |
| [0.9, 1.0) | 10 | 0.0136 | -0.0068 |

## Benchmark composition

| season | mkt_source | n | mean_n_books |
|---|---|---|---|
| 2016-17 | ps | 380 | 1.00 |
| 2017-18 | ps | 380 | 1.00 |
| 2018-19 | ps | 380 | 1.00 |
| 2019-20 | ps | 380 | 1.00 |
| 2020-21 | ps | 380 | 1.00 |
| 2021-22 | ps | 380 | 1.00 |
| 2022-23 | ps | 380 | 1.00 |
| 2023-24 | ps | 380 | 1.00 |
| 2024-25 | ps | 380 | 1.00 |
| 2025-26 | consensus | 170 | 6.49 |
| 2025-26 | ps | 210 | 1.00 |
