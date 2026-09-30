"""Descriptive statistics and data-quality checks for the September 30 milestone."""

import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "synthetic_microdata_2021-2024_20260921.zip"
OUT = ROOT / "analysis/results"


def number(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"scope": "Synthetic supplied tables only; descriptive/mechanical checks, not population estimates", "cases": {}}
    loaded = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()

        def read(path, columns=None):
            matches = [n for n in names if n == path or n.endswith("/" + path)]
            assert len(matches) == 1, (path, matches)
            with archive.open(matches[0]) as source:
                frame = pd.read_csv(source, dtype=str, keep_default_na=False, na_values=[""], usecols=columns)
            with archive.open(matches[0]) as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            loaded[path] = {"rows": len(frame), "selected_columns": list(frame.columns), "sha256": digest}
            print(f"Read {path}: {len(frame):,} rows", flush=True)
            return frame

        roster = read("d008/2024/kontr_k.csv")
        dwelling = read("d006/2024/rio_2024.csv")
        subject = read("d002/2024/subject.csv")
        assert not roster.duplicated(["NOMER", "NOMP"]).any()
        assert not dwelling.duplicated("NOMER").any()
        assert not subject.duplicated("NOMER").any()
        household_sizes = roster.groupby("NOMER").size()
        base = dwelling[["NOMER", "J_PL", "OB_PL", "KOL_K", "ZEM"]].copy()
        base["roster_rows"] = base.NOMER.map(household_sizes)
        base["J_PL"] = number(base.J_PL)
        base["OB_PL"] = number(base.OB_PL)
        area = base.J_PL.notna()
        result["cases"]["household_vs_person_denominator"] = {
            "households_with_dwelling": len(base), "roster_households": len(household_sizes),
            "unmatched_households": int(base.roster_rows.isna().sum()),
            "household_mean_J_PL": float(base.loc[area, "J_PL"].mean()),
            "household_median_J_PL": float(base.loc[area, "J_PL"].median()),
            "person_join_mean_of_household_J_PL": float(np.average(base.loc[area, "J_PL"], weights=base.loc[area, "roster_rows"])),
            "J_PL_above_OB_PL": int((base.J_PL > base.OB_PL).sum()),
            "note": "The second mean weights household dwelling area by roster size; it is not mean per-person area. No records removed, no survey weights.",
        }
        result["cases"]["room_code_normalization"] = {
            "raw_values": sorted(base.KOL_K.dropna().unique().tolist()),
            "raw_n_unique": int(base.KOL_K.nunique()),
            "numeric_n_unique": int(number(base.KOL_K).nunique()),
            "numeric_parse_failures": int((base.KOL_K.notna() & number(base.KOL_K).isna()).sum()),
            "note": "Numerical normalization is appropriate for documented room counts, not automatically for ID/classification codes.",
        }
        land_fields = [c for c in dwelling if c.startswith("NOMP") and c != "NOMP"]
        has_person_slot = dwelling[land_fields].notna().any(axis=1)
        has_positive_person_slot = dwelling[land_fields].apply(number).gt(0).any(axis=1)
        result["cases"]["land_question_routing"] = {
            "ZEM_counts": dwelling.ZEM.value_counts(dropna=False).to_dict(),
            "ZEM_no_rows": int((dwelling.ZEM == "2").sum()),
            "ZEM_no_with_any_person_slot": int(((dwelling.ZEM == "2") & has_person_slot).sum()),
            "ZEM_yes_with_no_person_slot": int(((dwelling.ZEM == "1") & ~has_person_slot).sum()),
            "ZEM_no_with_positive_person_slot": int(((dwelling.ZEM == "2") & has_positive_person_slot).sum()),
            "note": "D006 instructions route ZEM=2 to question 9. Positive person references in skipped question 8 conflict with that documented route. This is a synthetic-data rule conflict, not evidence of real respondent errors; no correction is inferred.",
        }
        satisfaction = number(subject.GR1)
        scored = satisfaction.between(1, 10)
        result["cases"]["satisfaction_special_code"] = {
            "file": "d002/2024/subject.csv", "field": "GR1",
            "rows": len(subject), "missing": int(satisfaction.isna().sum()),
            "code_89_count": int((satisfaction == 89).sum()),
            "valid_scale_1_10_count": int(scored.sum()),
            "other_nonmissing_values": sorted(satisfaction[satisfaction.notna() & ~scored & (satisfaction != 89)].unique().tolist()),
            "naive_mean_including_89": float(satisfaction.mean()),
            "mean_among_scale_answers": float(satisfaction[scored].mean()),
            "median_among_scale_answers": float(satisfaction[scored].median()),
            "satisfied_8_10_share_among_scale_answers": float(satisfaction[scored].ge(8).mean()),
            "note": "Unweighted synthetic respondent answers. 89 is not a score. Ordinal-scale means are shown only to demonstrate the coding pitfall.",
        }

        purchases = read("d004/2024/1kv/kv_vopr1.csv", ["NOMER", "KODNU", "STOIMK"])
        utilities = read("d004/2024/1kv/kv_vopr2.csv", ["NOMER", "KODNU", "STOIMK"])
        nr = utilities.groupby("NOMER").size()
        repeats = purchases.NOMER.map(nr).fillna(0).astype(int)
        amounts = number(purchases.STOIMK)
        common = repeats.gt(0)
        preserved = float(amounts[common].sum())
        inflated = float((amounts * repeats).sum())
        household_purchases = purchases.assign(amount=amounts).groupby("NOMER").amount.sum(min_count=1)
        household_utilities = utilities.assign(amount=number(utilities.STOIMK)).groupby("NOMER").amount.sum(min_count=1)
        safe = household_purchases.to_frame("purchases").join(household_utilities.to_frame("utilities"), how="inner", validate="one_to_one")
        assert np.isclose(safe.purchases.sum(), preserved)
        result["cases"]["many_to_many_join"] = {
            "left_rows": len(purchases), "right_rows": len(utilities),
            "naive_inner_join_rows": int(repeats.sum()),
            "left_rows_without_match": int((~common).sum()),
            "naive_join_left_STOIMK_sum": inflated,
            "left_STOIMK_sum_on_matched_households": preserved,
            "amount_inflation_factor": inflated / preserved,
            "safe_household_join_rows": len(safe),
            "safe_join_left_sum": float(safe.purchases.sum()),
            "note": "Exact join multiplicities computed without materializing the erroneous join. This covers only non-food purchases and utilities, not all household consumption.",
        }

        labour_path = "t001/2024/baza.csv"
        labour = read(labour_path, ["TE", "K", "RSPD", "kv", "mes", "ZAN_RABOTA", "ZAN_VREMYA_1", "ZAN_VREMYA_2"])
        expected_quarter = ((number(labour.mes) - 1) // 3) + 1
        quarter = number(labour.kv)
        complete = quarter.notna() & expected_quarter.notna()
        groups = labour.groupby("ZAN_RABOTA", dropna=False).agg(
            rows=("RSPD", "size"), followup_nonmissing=("ZAN_VREMYA_2", lambda x: x.notna().sum())
        )
        result["cases"]["labour_force_limits"] = {
            "rows": len(labour), "complete_month_quarter": int(complete.sum()),
            "month_quarter_mismatch": int((quarter[complete] != expected_quarter[complete]).sum()),
            "missing_quarter": int(quarter.isna().sum()),
            "distinct_RSPD": int(labour.RSPD.nunique()),
            "distinct_TE_K_RSPD_kv_mes": len(labour.drop_duplicates(["TE", "K", "RSPD", "kv", "mes"])),
            "followup_by_raw_work_code": groups.reset_index().to_dict("records"),
            "note": "Branch associations are descriptive, not a validated routing rule. No unemployment estimate is attempted.",
        }

        # All paths/columns in the existing exact audit, without treating code values as measurements.
        profiles = pd.read_csv(ROOT / "audit/column_profiles.csv", keep_default_na=False)
        result["cases"]["metadata_coverage"] = {
            "table_column_instances": len(profiles),
            "generic_description_instances": int(profiles.description.str.contains("полная расшифровка требует|полный кодбук ответов отсутствует", regex=True).sum()),
            "potential_weight_columns_by_name": profiles.loc[profiles.column.str.contains(r"weight|ves|wgt|(?:^|_)wt(?:$|_)|koef|вес", case=False, regex=True), ["path", "column"]].to_dict("records"),
            "note": "Name search cannot prove that no weights exist. Generic descriptions reflect limits of the audit, not necessarily absent source documentation.",
        }
    result["inputs"] = loaded
    result["elapsed_seconds"] = time.time() - started
    (OUT / "descriptive_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["cases"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
