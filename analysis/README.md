# Real-data descriptive tests

`run_real_data_tests.py` reads local Annual Survey of Industries panel blocks
A, B, E, H, and I for 1998-2017. It writes only aggregate results to
`results.json`; plant identifiers and plant-level records are never exported.

Run from the repository root:

```bash
python analysis/run_real_data_tests.py \
  --source-dir /path/to/asi/panel/blocks \
  --output analysis/results.json
```

The script estimates:

- an industry-year adjusted age-size gradient, with standard errors clustered
  by plant identifier;
- whether within-plant changes in directly imported input share predict
  next-year changes in major indigenous-input breadth;
- the two-year recovery rate after large plant-level indigenous-input breadth
  losses; and
- panel-linkage diagnostics needed to judge whether entry and exit can be used.

Input-breadth comparison windows never cross ASI schedule regimes: the early
forms requested five major indigenous inputs, while later forms requested ten.

These are provisional descriptive tests. They do not identify causal effects of
credit, importing, supplier destruction, or regional manufacturing hysteresis.
The current files lack the validated state crosswalk, bank exposure, demand
shocks, deflators, and informal-sector data required for those claims.
