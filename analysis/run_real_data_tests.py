#!/usr/bin/env python
"""Run aggregate-only, descriptive thesis tests on local ASI panel blocks.

The script deliberately does not export plant identifiers or plant-level rows.
It tests only hypotheses supported by the available ASI variables and labels
the resulting associations as descriptive rather than causal.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy import stats


DOMESTIC_TOTAL_CODES = {"99930", "9993000"}
IMPORT_TOTAL_CODES = {"99940", "9994000"}
DOMESTIC_COMPONENT_CODES = {"99901", "99920", "9990100", "9992000"}
CODE_REGIMES = (
    (1998, 2003, "NIC-1998"),
    (2004, 2007, "NIC-2004"),
    (2008, 2017, "NIC-2008"),
)
INPUT_SCHEDULE_REGIMES = (
    (1998, 2002, "five-major-input-form"),
    (2003, 2007, "ten-major-input-form-pre-2008"),
    (2008, 2017, "ten-major-input-form-2008-plus"),
)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def code_string(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    )


def read_dta(path: Path, columns: list[str]) -> pd.DataFrame:
    frame, _ = pyreadstat.read_dta(path, usecols=columns)
    return frame


def collapse_single_value(
    frame: pd.DataFrame,
    value: str,
    output: str,
) -> tuple[pd.DataFrame, int]:
    """Keep an ID-year value only when repeated rows agree."""
    selected = frame[["ID", "Year", value]].dropna(subset=["ID", "Year"])
    grouped = selected.groupby(["ID", "Year"], sort=False)[value]
    result = grouped.agg(["first", "nunique"]).reset_index()
    conflicts = int((result["nunique"] > 1).sum())
    result.loc[result["nunique"] > 1, "first"] = np.nan
    return result[["ID", "Year", "first"]].rename(columns={"first": output}), conflicts


def cluster_robust_slope(
    frame: pd.DataFrame,
    *,
    outcome: str,
    regressor: str,
    fixed_effect: str,
    cluster: str,
) -> dict[str, float | int]:
    """One-regressor OLS after fixed-effect demeaning, clustered by plant."""
    sample = frame[[outcome, regressor, fixed_effect, cluster]].dropna().copy()
    cell_size = sample.groupby(fixed_effect)[outcome].transform("size")
    sample = sample.loc[cell_size >= 2].copy()
    sample["y"] = sample[outcome] - sample.groupby(fixed_effect)[outcome].transform("mean")
    sample["x"] = sample[regressor] - sample.groupby(fixed_effect)[regressor].transform("mean")
    denominator = float(np.square(sample["x"]).sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"No within-cell variation in {regressor}")

    beta = float((sample["x"] * sample["y"]).sum() / denominator)
    sample["residual"] = sample["y"] - beta * sample["x"]
    sample["score"] = sample["x"] * sample["residual"]
    cluster_scores = sample.groupby(cluster, sort=False)["score"].sum()
    n_obs = int(len(sample))
    n_clusters = int(cluster_scores.size)
    fixed_effect_cells = int(sample[fixed_effect].nunique())
    n_parameters = fixed_effect_cells + 1
    correction = (n_clusters / (n_clusters - 1)) * (
        (n_obs - 1) / max(n_obs - n_parameters, 1)
    )
    variance = correction * float(np.square(cluster_scores).sum()) / denominator**2
    standard_error = math.sqrt(max(variance, 0.0))
    degrees_freedom = max(n_clusters - 1, 1)
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    t_stat = beta / standard_error if standard_error > 0 else math.nan
    p_value = (
        float(2 * stats.t.sf(abs(t_stat), degrees_freedom))
        if math.isfinite(t_stat)
        else math.nan
    )
    return {
        "estimate": beta,
        "std_error": standard_error,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": beta - critical * standard_error,
        "ci_high": beta + critical * standard_error,
        "observations": n_obs,
        "clusters": n_clusters,
        "fixed_effect_cells": fixed_effect_cells,
    }


def regime(year: int) -> str | None:
    for start, end, label in CODE_REGIMES:
        if start <= year <= end:
            return label
    return None


def input_schedule_regime(year: int) -> str | None:
    for start, end, label in INPUT_SCHEDULE_REGIMES:
        if start <= year <= end:
            return label
    return None


def load_year(panel_dir: Path, year_file: Path) -> tuple[pd.DataFrame, dict[str, int | float]]:
    suffix = year_file.name.removeprefix("A_")
    paths = {letter: panel_dir / f"{letter}_{suffix}" for letter in "ABEHI"}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required ASI blocks: " + ", ".join(missing))

    a = read_dta(
        paths["A"],
        ["Year", "ID", "SchemeCode", "IndCodeReturn", "Multiple"],
    )
    a["ID"] = code_string(a["ID"])
    a["industry_code"] = code_string(a["IndCodeReturn"])
    a["scheme_code"] = code_string(a["SchemeCode"])
    a["weight"] = numeric(a["Multiple"])
    duplicate_mask = a.duplicated(["ID", "Year"], keep=False)
    a_unique = a.loc[~duplicate_mask, ["ID", "Year", "industry_code", "scheme_code", "weight"]].copy()

    b = read_dta(paths["B"], ["Year", "ID", "YearInitialProduction"])
    b["ID"] = code_string(b["ID"])
    b["initial_production_year"] = numeric(b["YearInitialProduction"])
    initial_year, initial_year_conflicts = collapse_single_value(
        b, "initial_production_year", "initial_production_year"
    )

    e = read_dta(paths["E"], ["Year", "ID", "SI", "Persons"])
    e["ID"] = code_string(e["ID"])
    e["SI"] = numeric(e["SI"])
    e["employment"] = numeric(e["Persons"])
    employment, employment_conflicts = collapse_single_value(
        e.loc[e["SI"].eq(9)], "employment", "employment"
    )

    h = read_dta(paths["H"], ["Year", "ID", "SI", "asicc", "PurchVal"])
    h["ID"] = code_string(h["ID"])
    h["SI"] = numeric(h["SI"])
    h["item_code"] = code_string(h["asicc"])
    h["purchase_value"] = numeric(h["PurchVal"])
    domestic_total, domestic_total_conflicts = collapse_single_value(
        h.loc[h["item_code"].isin(DOMESTIC_TOTAL_CODES)],
        "purchase_value",
        "domestic_input_value",
    )
    domestic_items = h.loc[
        h["SI"].between(1, 10)
        & h["purchase_value"].gt(0)
        & ~h["item_code"].fillna("").str.startswith("99"),
        ["ID", "Year", "item_code"],
    ]
    domestic_breadth = (
        domestic_items.groupby(["ID", "Year"], sort=False)["item_code"]
        .nunique()
        .rename("domestic_input_breadth")
        .reset_index()
    )
    h_totals = (
        h.loc[h["item_code"].isin(DOMESTIC_COMPONENT_CODES)]
        .groupby(["ID", "Year"], sort=False)["purchase_value"]
        .sum(min_count=2)
        .rename("domestic_component_sum")
        .reset_index()
    )
    domestic_total = domestic_total.merge(h_totals, on=["ID", "Year"], how="left")
    domestic_error = (
        (domestic_total["domestic_input_value"] - domestic_total["domestic_component_sum"])
        .abs()
        .div(domestic_total["domestic_input_value"].abs().clip(lower=1))
    )
    domestic_total["domestic_total_consistent"] = (
        domestic_total["domestic_input_value"].ge(0)
        & (domestic_error.isna() | domestic_error.le(0.005))
    )

    i = read_dta(paths["I"], ["Year", "ID", "SI", "asicc", "PurchVal"])
    i["ID"] = code_string(i["ID"])
    i["SI"] = numeric(i["SI"])
    i["item_code"] = code_string(i["asicc"])
    i["purchase_value"] = numeric(i["PurchVal"])
    import_total, import_total_conflicts = collapse_single_value(
        i.loc[i["item_code"].isin(IMPORT_TOTAL_CODES)],
        "purchase_value",
        "imported_input_value",
    )
    imported_items = i.loc[
        i["SI"].between(1, 5)
        & i["purchase_value"].gt(0)
        & ~i["item_code"].fillna("").str.startswith("99"),
        ["ID", "Year", "item_code"],
    ]
    imported_breadth = (
        imported_items.groupby(["ID", "Year"], sort=False)["item_code"]
        .nunique()
        .rename("imported_input_breadth")
        .reset_index()
    )
    i_components = (
        i.loc[i["SI"].between(1, 6)]
        .groupby(["ID", "Year"], sort=False)["purchase_value"]
        .sum(min_count=1)
        .rename("import_component_sum")
        .reset_index()
    )
    import_total = import_total.merge(i_components, on=["ID", "Year"], how="left")
    import_error = (
        (import_total["imported_input_value"] - import_total["import_component_sum"])
        .abs()
        .div(import_total["imported_input_value"].abs().clip(lower=1))
    )
    import_total["import_total_consistent"] = import_error.le(0.005)
    import_block_presence = i[["ID", "Year"]].drop_duplicates().assign(
        import_block_present=True
    )
    import_total = import_block_presence.merge(
        import_total, on=["ID", "Year"], how="left", validate="one_to_one"
    )

    panel = (
        a_unique.merge(initial_year, on=["ID", "Year"], how="left", validate="one_to_one")
        .merge(employment, on=["ID", "Year"], how="left", validate="one_to_one")
        .merge(
            domestic_total[
                ["ID", "Year", "domestic_input_value", "domestic_total_consistent"]
            ],
            on=["ID", "Year"],
            how="left",
            validate="one_to_one",
        )
        .merge(domestic_breadth, on=["ID", "Year"], how="left", validate="one_to_one")
        .merge(
            import_total[
                [
                    "ID",
                    "Year",
                    "imported_input_value",
                    "import_total_consistent",
                    "import_block_present",
                ]
            ],
            on=["ID", "Year"],
            how="left",
            validate="one_to_one",
        )
        .merge(imported_breadth, on=["ID", "Year"], how="left", validate="one_to_one")
    )
    panel.loc[
        ~panel["domestic_total_consistent"].eq(True), "domestic_input_value"
    ] = np.nan
    panel["domestic_input_breadth"] = panel["domestic_input_breadth"].fillna(0)
    no_import_block = panel["import_block_present"].isna()
    panel.loc[no_import_block, "imported_input_value"] = 0
    panel.loc[
        panel["import_block_present"].eq(True)
        & ~panel["import_total_consistent"].eq(True),
        "imported_input_value",
    ] = np.nan
    panel["imported_input_breadth"] = panel["imported_input_breadth"].fillna(0)

    diagnostics = {
        "raw_a_rows": int(len(a)),
        "duplicate_a_rows": int(duplicate_mask.sum()),
        "nonpositive_weights": int(a["weight"].le(0).sum()),
        "missing_weights": int(a["weight"].isna().sum()),
        "initial_year_conflicts": initial_year_conflicts,
        "employment_conflicts": employment_conflicts,
        "domestic_total_conflicts": domestic_total_conflicts,
        "import_total_conflicts": import_total_conflicts,
        "domestic_total_consistency_n": int(domestic_error.notna().sum()),
        "domestic_total_consistency_pass": int(domestic_error.le(0.005).sum()),
        "import_total_consistency_n": int(import_error.notna().sum()),
        "import_total_consistency_pass": int(import_error.le(0.005).sum()),
    }
    return panel, diagnostics


def persistence_test(panel: pd.DataFrame, diagnostics: list[dict[str, int | float]]) -> dict:
    ordered = panel.sort_values(["ID", "Year"]).copy()
    observations = ordered.groupby("ID")["Year"].nunique()
    gaps = ordered.groupby("ID")["Year"].apply(
        lambda years: bool((np.diff(np.sort(years.unique())) > 1).any())
    )
    repeated = observations.ge(2)
    valid_initial = ordered.loc[
        ordered["initial_production_year"].notna()
        & ordered["initial_production_year"].le(ordered["Year"])
        & ordered["initial_production_year"].ge(ordered["Year"] - 150)
    ]
    initial_counts = valid_initial.groupby("ID")["initial_production_year"].agg(
        observations="count", distinct="nunique"
    )
    initial_repeat = initial_counts["observations"].ge(2)

    ordered["previous_year"] = ordered.groupby("ID")["Year"].shift()
    ordered["previous_industry"] = ordered.groupby("ID")["industry_code"].shift()
    ordered["regime"] = ordered["Year"].map(regime)
    ordered["previous_regime"] = ordered.groupby("ID")["regime"].shift()
    ordered["consecutive"] = ordered["Year"].sub(ordered["previous_year"]).eq(1)
    comparable = ordered.loc[
        ordered["consecutive"]
        & ordered["regime"].eq(ordered["previous_regime"])
        & ordered["industry_code"].notna()
        & ordered["previous_industry"].notna()
    ].copy()
    comparable["same_exact_industry"] = comparable["industry_code"].eq(
        comparable["previous_industry"]
    )
    comparable["same_two_digit_industry"] = comparable["industry_code"].str[:2].eq(
        comparable["previous_industry"].str[:2]
    )

    return {
        "plant_year_rows": int(sum(item["raw_a_rows"] for item in diagnostics)),
        "analysis_rows_after_duplicate_exclusion": int(len(panel)),
        "unique_ids": int(observations.size),
        "ids_two_plus": int(repeated.sum()),
        "ids_with_gap_among_repeated": int((gaps & repeated).sum()),
        "gap_share_among_repeated": float((gaps & repeated).sum() / repeated.sum()),
        "ids_with_repeated_valid_initial_year": int(initial_repeat.sum()),
        "stable_initial_year_share": float(
            initial_counts.loc[initial_repeat, "distinct"].eq(1).mean()
        ),
        "duplicate_id_year_rows": int(sum(item["duplicate_a_rows"] for item in diagnostics)),
        "consecutive_same_regime_pairs": int(len(comparable)),
        "exact_industry_stability_rate": float(comparable["same_exact_industry"].mean()),
        "two_digit_industry_stability_rate": float(
            comparable["same_two_digit_industry"].mean()
        ),
        "interpretation": (
            "Repeated IDs and high within-code-regime industry stability support panel "
            "feasibility, but gaps and duplicate ID-years prevent validated entry/exit claims."
        ),
    }


def scale_up_test(panel: pd.DataFrame) -> dict:
    sample = panel.loc[
        panel["employment"].gt(0)
        & panel["initial_production_year"].notna()
        & panel["industry_code"].notna()
    ].copy()
    sample["age"] = sample["Year"] - sample["initial_production_year"]
    sample = sample.loc[sample["age"].between(0, 150)].copy()
    sample["log_employment"] = np.log(sample["employment"])
    sample["log1p_age"] = np.log1p(sample["age"])
    sample["industry_two_digit"] = sample["industry_code"].str[:2]
    sample["industry_year"] = (
        sample["Year"].astype("int64").astype(str) + ":" + sample["industry_two_digit"]
    )
    initial_audit = sample.groupby("ID")["initial_production_year"].agg(
        observations="count", distinct="nunique"
    )
    stable_repeated_ids = initial_audit.index[
        initial_audit["observations"].ge(2) & initial_audit["distinct"].eq(1)
    ]
    model = cluster_robust_slope(
        sample,
        outcome="log_employment",
        regressor="log1p_age",
        fixed_effect="industry_year",
        cluster="ID",
    )
    stable_initial_year_model = cluster_robust_slope(
        sample.loc[sample["ID"].isin(stable_repeated_ids)],
        outcome="log_employment",
        regressor="log1p_age",
        fixed_effect="industry_year",
        cluster="ID",
    )
    employment_low, employment_high = sample["employment"].quantile([0.01, 0.99])
    employment_trimmed_model = cluster_robust_slope(
        sample.loc[sample["employment"].between(employment_low, employment_high)],
        outcome="log_employment",
        regressor="log1p_age",
        fixed_effect="industry_year",
        cluster="ID",
    )
    predicted_5_to_20 = math.exp(
        float(model["estimate"]) * (math.log1p(20) - math.log1p(5))
    ) - 1

    bins = [-1, 5, 10, 20, 150]
    labels = ["0-5", "6-10", "11-20", "21+"]
    sample["age_group"] = pd.cut(sample["age"], bins=bins, labels=labels)
    age_groups = []
    for label, group in sample.groupby("age_group", observed=True):
        age_groups.append(
            {
                "age_group": str(label),
                "observations": int(len(group)),
                "median_employment": float(group["employment"].median()),
                "geometric_mean_employment": float(math.exp(group["log_employment"].mean())),
            }
        )

    supported = model["estimate"] > 0 and model["p_value"] < 0.05
    return {
        "hypothesis": "Older plants are larger, consistent with scale-up over the life cycle.",
        "estimand": (
            "Within industry-year elasticity of employment with respect to one plus plant age."
        ),
        "model": model,
        "stable_initial_year_model": stable_initial_year_model,
        "employment_trimmed_1_99pct_model": employment_trimmed_model,
        "predicted_employment_difference_age_5_to_20": predicted_5_to_20,
        "age_groups": age_groups,
        "result": "descriptive_support" if supported else "descriptive_no_support",
        "interpretation": (
            "A positive coefficient supports an age-size gradient, not causal learning or "
            "survival effects; selective exit and cohort composition remain confounders."
        ),
    }


def foreign_sourcing_test(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    sample = panel.loc[
        panel["domestic_input_value"].gt(0)
        & panel["imported_input_value"].ge(0)
        & panel["industry_code"].notna()
    ].copy()
    denominator = sample["domestic_input_value"] + sample["imported_input_value"]
    sample["import_share"] = sample["imported_input_value"].div(denominator)
    sample = sample.loc[sample["import_share"].between(0, 1)].sort_values(["ID", "Year"])
    grouped = sample.groupby("ID", sort=False)
    sample["previous_year"] = grouped["Year"].shift()
    sample["next_year"] = grouped["Year"].shift(-1)
    sample["previous_import_share"] = grouped["import_share"].shift()
    sample["next_domestic_breadth"] = grouped["domestic_input_breadth"].shift(-1)
    sample["input_schedule_regime"] = sample["Year"].map(input_schedule_regime)
    sample["previous_input_schedule_regime"] = grouped["input_schedule_regime"].shift()
    sample["next_input_schedule_regime"] = grouped["input_schedule_regime"].shift(-1)
    sample["delta_import_share"] = sample["import_share"] - sample["previous_import_share"]
    sample["next_delta_domestic_breadth"] = (
        sample["next_domestic_breadth"] - sample["domestic_input_breadth"]
    )
    sample["industry_year"] = (
        sample["Year"].astype("int64").astype(str) + ":" + sample["industry_code"].str[:2]
    )
    triples = sample.loc[
        sample["Year"].sub(sample["previous_year"]).eq(1)
        & sample["next_year"].sub(sample["Year"]).eq(1)
        & sample["input_schedule_regime"].eq(sample["previous_input_schedule_regime"])
        & sample["input_schedule_regime"].eq(sample["next_input_schedule_regime"])
    ].copy()
    model = cluster_robust_slope(
        triples,
        outcome="next_delta_domestic_breadth",
        regressor="delta_import_share",
        fixed_effect="industry_year",
        cluster="ID",
    )
    absolute_change_cutoff = float(triples["delta_import_share"].abs().quantile(0.99))
    trimmed = triples.loc[
        triples["delta_import_share"].abs().le(absolute_change_cutoff)
    ].copy()
    trimmed_model = cluster_robust_slope(
        trimmed,
        outcome="next_delta_domestic_breadth",
        regressor="delta_import_share",
        fixed_effect="industry_year",
        cluster="ID",
    )
    absolute_change_cutoff_95 = float(triples["delta_import_share"].abs().quantile(0.95))
    trimmed_95 = triples.loc[
        triples["delta_import_share"].abs().le(absolute_change_cutoff_95)
    ].copy()
    trimmed_95_model = cluster_robust_slope(
        trimmed_95,
        outcome="next_delta_domestic_breadth",
        regressor="delta_import_share",
        fixed_effect="industry_year",
        cluster="ID",
    )
    nic_2008_model = cluster_robust_slope(
        triples.loc[triples["Year"].ge(2009)],
        outcome="next_delta_domestic_breadth",
        regressor="delta_import_share",
        fixed_effect="industry_year",
        cluster="ID",
    )
    triples["import_entry"] = (
        triples["previous_import_share"].eq(0) & triples["import_share"].gt(0)
    ).astype(float)
    importer_entry_model = cluster_robust_slope(
        triples,
        outcome="next_delta_domestic_breadth",
        regressor="import_entry",
        fixed_effect="industry_year",
        cluster="ID",
    )
    effect_10pp = float(model["estimate"]) * 0.10
    supported = model["estimate"] < 0 and model["p_value"] < 0.05
    importers = sample.loc[sample["import_share"].gt(0), "import_share"]
    return (
        {
            "hypothesis": (
                "A shift toward directly imported inputs predicts a later contraction in "
                "the breadth of major indigenous inputs."
            ),
            "estimand": (
                "Change in next-year indigenous-input breadth associated with a within-plant "
                "change in imported-input share, net of two-digit-industry-by-year effects "
                "and restricted to the same ASI input-schedule regime."
            ),
            "model": model,
            "trimmed_99pct_model": trimmed_model,
            "trimmed_95pct_model": trimmed_95_model,
            "nic_2008_period_model": nic_2008_model,
            "importer_entry_model": importer_entry_model,
            "effect_of_10pp_import_share_increase": effect_10pp,
            "plant_years": int(len(sample)),
            "analysis_years": [int(sample["Year"].min()), int(sample["Year"].max())],
            "importing_plant_years": int(sample["import_share"].gt(0).sum()),
            "importing_plant_year_share": float(sample["import_share"].gt(0).mean()),
            "median_import_share_among_importers": float(importers.median()),
            "result": "descriptive_support" if supported else "descriptive_no_support",
            "interpretation": (
                "The lagged first-difference design removes time-invariant plant differences "
                "and same-year denominator mechanics, but import choice remains endogenous and "
                "the association is not causal evidence of supplier destruction."
            ),
        },
        sample,
    )


def hysteresis_diagnostic(input_panel: pd.DataFrame) -> dict:
    sample = input_panel.sort_values(["ID", "Year"]).copy()
    grouped = sample.groupby("ID", sort=False)
    sample["previous_year"] = grouped["Year"].shift()
    sample["next_year"] = grouped["Year"].shift(-1)
    sample["next_two_year"] = grouped["Year"].shift(-2)
    sample["previous_breadth"] = grouped["domestic_input_breadth"].shift()
    sample["next_breadth"] = grouped["domestic_input_breadth"].shift(-1)
    sample["next_two_breadth"] = grouped["domestic_input_breadth"].shift(-2)
    sample["input_schedule_regime"] = sample["Year"].map(input_schedule_regime)
    sample["previous_input_schedule_regime"] = grouped["input_schedule_regime"].shift()
    sample["next_input_schedule_regime"] = grouped["input_schedule_regime"].shift(-1)
    sample["next_two_input_schedule_regime"] = grouped["input_schedule_regime"].shift(-2)
    complete_window = (
        sample["Year"].sub(sample["previous_year"]).eq(1)
        & sample["next_year"].sub(sample["Year"]).eq(1)
        & sample["next_two_year"].sub(sample["Year"]).eq(2)
        & sample["input_schedule_regime"].eq(sample["previous_input_schedule_regime"])
        & sample["input_schedule_regime"].eq(sample["next_input_schedule_regime"])
        & sample["input_schedule_regime"].eq(sample["next_two_input_schedule_regime"])
    )
    threshold_results = []
    for threshold in (1, 2, 3):
        events = sample.loc[
            complete_window
            & sample["previous_breadth"]
            .sub(sample["domestic_input_breadth"])
            .ge(threshold)
        ].copy()
        events["recovered_one_year"] = events["next_breadth"].ge(
            events["previous_breadth"]
        )
        events["recovered_two_year"] = (
            events[["next_breadth", "next_two_breadth"]]
            .max(axis=1)
            .ge(events["previous_breadth"])
        )
        threshold_results.append(
            {
                "breadth_loss_threshold": threshold,
                "events": int(len(events)),
                "recovered_within_one_year_share": float(
                    events["recovered_one_year"].mean()
                ),
                "recovered_within_two_years_share": float(
                    events["recovered_two_year"].mean()
                ),
                "not_recovered_within_two_years_share": float(
                    (~events["recovered_two_year"]).mean()
                ),
            }
        )
    preferred = threshold_results[1]
    return {
        "hypothesis": "Large indigenous-input breadth losses may persist rather than reverse quickly.",
        "event_definition": (
            "A fall of at least two major indigenous input codes with two subsequent consecutive "
            "observations available inside the same ASI input-schedule regime."
        ),
        "events": preferred["events"],
        "recovered_within_one_year_share": preferred[
            "recovered_within_one_year_share"
        ],
        "recovered_within_two_years_share": preferred[
            "recovered_within_two_years_share"
        ],
        "not_recovered_within_two_years_share": preferred[
            "not_recovered_within_two_years_share"
        ],
        "threshold_sensitivity": threshold_results,
        "result": "plant_level_persistence_diagnostic",
        "interpretation": (
            "This is a plant-level persistence diagnostic, not the thesis's regional supplier-"
            "ecosystem hysteresis test; sampling, reporting changes, and ID gaps remain threats."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    a_files = [
        path
        for path in sorted(args.source_dir.glob("A_*.dta"))
        if "conflicted copy" not in path.name
    ]
    if len(a_files) != 20:
        raise ValueError(f"Expected 20 annual A blocks, found {len(a_files)}")

    panels: list[pd.DataFrame] = []
    diagnostics: list[dict[str, int | float]] = []
    for a_file in a_files:
        panel, year_diagnostics = load_year(args.source_dir, a_file)
        panels.append(panel)
        diagnostics.append(year_diagnostics)
        print(f"Loaded {a_file.name}: {len(panel):,} analysis rows")

    full_panel = pd.concat(panels, ignore_index=True)
    persistence = persistence_test(full_panel, diagnostics)
    scale_up = scale_up_test(full_panel)
    foreign_sourcing, input_panel = foreign_sourcing_test(full_panel)
    hysteresis = hysteresis_diagnostic(input_panel)

    domestic_consistency_n = sum(
        int(item["domestic_total_consistency_n"]) for item in diagnostics
    )
    import_consistency_n = sum(
        int(item["import_total_consistency_n"]) for item in diagnostics
    )
    quality = {
        "source_files_used": len(a_files) * 5,
        "source_bytes_used": int(
            sum(
                (args.source_dir / f"{letter}_{a_file.name.removeprefix('A_')}").stat().st_size
                for a_file in a_files
                for letter in "ABEHI"
            )
        ),
        "nonpositive_weights": int(sum(item["nonpositive_weights"] for item in diagnostics)),
        "missing_weights": int(sum(item["missing_weights"] for item in diagnostics)),
        "domestic_total_consistency_rate": float(
            sum(int(item["domestic_total_consistency_pass"]) for item in diagnostics)
            / domestic_consistency_n
        ),
        "import_total_consistency_rate": float(
            sum(int(item["import_total_consistency_pass"]) for item in diagnostics)
            / import_consistency_n
        ),
        "initial_year_conflicts": int(
            sum(item["initial_year_conflicts"] for item in diagnostics)
        ),
        "employment_conflicts": int(sum(item["employment_conflicts"] for item in diagnostics)),
        "domestic_total_conflicts": int(
            sum(item["domestic_total_conflicts"] for item in diagnostics)
        ),
        "import_total_conflicts": int(
            sum(item["import_total_conflicts"] for item in diagnostics)
        ),
        "weights_used_in_estimation": False,
        "input_schedule_regime_restriction": True,
        "plant_identifiers_exported": False,
    }

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "source": "Annual Survey of Industries panel blocks A, B, E, H, and I",
            "years": [1998, 2017],
            "unit": "plant-year",
            "status": "provisional real-data descriptive analysis",
        },
        "evidence_boundary": {
            "causal_claims_supported": False,
            "untestable_with_current_files": [
                "Production-oriented local bank credit builds regional supplier ecosystems",
                "Non-production credit is weaker than production credit",
                "Foreign sourcing causally destroys regional domestic supplier capacity",
                "Regional supplier loss creates manufacturing hysteresis",
                "Formal-informal reallocation explains state manufacturing gaps",
            ],
        },
        "tests": {
            "panel_persistence": persistence,
            "scale_up": scale_up,
            "foreign_sourcing": foreign_sourcing,
            "hysteresis_diagnostic": hysteresis,
        },
        "quality": quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote aggregate-only results to {args.output}")


if __name__ == "__main__":
    main()
