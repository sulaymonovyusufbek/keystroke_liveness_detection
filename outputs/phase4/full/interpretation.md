Uncertainty from the test population is small: 98% of the open-set APCER @τ10 CIs span ≤ 1 pp (widest 1.0 pp), and
AUC CIs span ≤ 0.023. Seed variation is of a similar or larger size. The across-seed sd of APCER is at most 1.9 pp at τ10 / 1.0 pp at τ5
for the D4 variants, 2.1 / 3.6 pp for D5 leave-one-out, and 4.3 pp at τ10 (trained on A3, tested on A5b) / 6.4 pp at τ5
(trained on A3, tested on A2) for single D5 cross-attack cells. The across-seed sd of BPCER is ≤ 0.2 pp. Section (e)
lists APCER @τ10 for the cross-attack cells. Re-setting τ on resampled validation
users widens the D4-primary CIs about 2–5×, but no headline value moves outside them. The two cross-attack averages
have narrow, non-overlapping CIs: D3 61.4% [61.3, 61.5] vs D5 37.7% [37.5, 37.8]. On the identical-key pairs,
D5's BPCER CIs (19–33%) do not overlap D4-primary's (6.8–7.6%). D6, a key-conditioned lookup model without a
recurrent network, is far weaker than D4-primary on A2, A3, A4 and A5b in every variant (best: D6-gauss-speed, AUC
0.75–0.89 vs 0.968–0.991), and is also weaker than the timing-only D4-nokey on A2–A4. Key conditioning in a context
table, even with a per-session speed correction, therefore does not reproduce D4's detection rates. (Note: A2's
timings are themselves drawn from attacker-pool digraph distributions, the same kind of statistic that D6 tabulates.)
