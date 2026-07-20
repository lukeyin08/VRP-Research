| comparison                                       |   mean_daily_diff |   ann_diff |    t_stat |   p_value |
|:-------------------------------------------------|------------------:|-----------:|----------:|----------:|
| model_linear_5d - always_short (net, HAC lag 22) |           5e-05   |    0.01173 |   0.80534 |   0.42062 |
| model_binary - always_short (net, HAC lag 22)    |          -3e-05   |   -0.00833 |  -1.18276 |   0.2369  |
| deflated Sharpe model_linear_5d (N=66 trials)    |           0.64347 |    0.6365  | nan       |   0.48959 |
| deflated Sharpe always_short (N=66 trials)       |           0.46597 |    0.62028 | nan       |   0.72317 |