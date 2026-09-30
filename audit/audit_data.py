"""Read-only source audit; generated evidence lives in audit/.

Run with: uv run --with polars --with xlrd --with python-docx --with pypdf
    python audit/audit_data.py prepare|profile
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit"
STEM = "synthetic_microdata_2021-2024_20260921"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def csvout(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def prepare():
    import xlrd
    from docx import Document
    dest = OUT / "documents"
    dest.mkdir(parents=True, exist_ok=True)
    inventory, docs, cached = [], [], {}
    with tarfile.open(ROOT / (STEM + ".tar")) as tar, zipfile.ZipFile(ROOT / (STEM + ".zip")) as z:
        zm = {x.filename.split("/", 1)[1]: x for x in z.infolist() if not x.is_dir()}
        tm = {m.name.split("/", 1)[1]: m for m in tar if m.isfile()}
        index = list(csv.DictReader(io.StringIO(tar.extractfile(tm["documentation/INDEX.csv"]).read().decode())))
        kinds = {"documentation/" + r["path"]: r["kind"] for r in index}
        sources = [("tar", n, tar.extractfile(m)) for n, m in tm.items() if not (n.endswith(".csv") and not n.startswith("documentation/"))]
        sources += [("zip", n, z.open(m)) for n, m in zm.items() if n not in tm]
        for container, name, stream in sources:
            with stream:
                data = stream.read()
            digest, ext = sha(data), Path(name).suffix.lower()
            record = {"container": container, "path": name, "bytes": len(data), "sha256": digest,
                      "kind": kinds.get(name, "guide/catalog/metadata")}
            if digest not in cached:
                info = {}
                if ext == ".xls":
                    book = xlrd.open_workbook(file_contents=data)
                    sheets = {s.name: [s.row_values(i) for i in range(s.nrows)] for s in book.sheets()}
                    info["sheets"] = {k: {"rows": len(v), "columns": max(map(len, v), default=0)} for k, v in sheets.items()}
                    dump(dest / (digest[:16] + ".json"), sheets)
                    text = "\n".join("## " + k + "\n" + "\n".join("\t".join(str(x).replace("\n", " / ") for x in row) for row in rows) for k, rows in sheets.items())
                elif ext == ".doc":
                    p = subprocess.run(["textutil", "-convert", "txt", "-format", "doc", "-stdin", "-stdout"], input=data, capture_output=True, check=True)
                    text = p.stdout.decode("utf-8")
                elif ext == ".docx":
                    doc = Document(io.BytesIO(data))
                    # iter_inner_content retains paragraph/table ordering.
                    blocks = []
                    for block in doc.iter_inner_content():
                        if hasattr(block, "rows"):
                            blocks.extend("\t".join(c.text for c in row.cells) for row in block.rows)
                        else:
                            blocks.append(block.text)
                    text = "\n".join(blocks)
                    info["inline_images"] = len(doc.inline_shapes)
                elif ext == ".pdf":
                    p = subprocess.run(["pdftotext", "-layout", "-", "-"], input=data, capture_output=True, check=True)
                    text = p.stdout.decode("utf-8")
                    info["pages"] = text.count("\f")
                else:
                    text = data.decode("utf-8-sig")
                textpath = dest / (digest[:16] + ".txt")
                textpath.write_text(text, encoding="utf-8")
                cached[digest] = {"text_path": str(textpath.relative_to(ROOT)), "text_chars": len(text), **info}
            record.update(cached[digest])
            docs.append(record)
        for n in sorted(set(tm) | set(zm)):
            inventory.append({"path": n, "in_tar": n in tm, "in_zip": n in zm,
                              "bytes": tm[n].size if n in tm else zm[n].file_size,
                              "kind": kinds.get(n, "data" if n.endswith(".csv") and n != "MANIFEST.csv" else "guide/metadata")})
        manifest = json.loads(z.read(zm["MANIFEST.json"]))
        dump(OUT / "manifest.json", manifest)
        csvout(OUT / "file_inventory.csv", inventory)
        dump(OUT / "documents.json", docs)
        print(json.dumps({"files_union": len(inventory), "documents": len(docs), "unique_document_contents": len(cached), "documentation_kinds": dict(collections.Counter(kinds.values()))}, ensure_ascii=False), flush=True)
    source_hashes = {}
    for p in [ROOT / (STEM + ".zip"), ROOT / (STEM + ".tar")]:
        with p.open("rb") as f:
            source_hashes[p.name] = {"sha256": hashlib.file_digest(f, "sha256").hexdigest(), "bytes": p.stat().st_size}
    dump(OUT / "source_hashes.json", source_hashes)


def finite(x):
    return None if isinstance(x, float) and not math.isfinite(x) else x


def profile_column(s, path):
    n, null = len(s), s.null_count()
    v = s.drop_nulls()
    counts = v.value_counts().sort(["count", s.name], descending=[True, False])
    distinct = len(counts)
    num = v.cast(pl.Float64, strict=False)
    parsed = len(v) - num.null_count()
    integer = bool(len(v)) and bool(v.str.contains(r"^[+-]?\d+$").all())
    leading = int(v.str.contains(r"^0\d").sum())
    rec = {"path": path, "column": s.name, "storage_type": "CSV text",
           "inferred_type": "empty" if not len(v) else "integer" if integer else "numeric" if parsed == len(v) else "string",
           "rows": n, "nulls": null, "null_pct": round(100 * null / n, 6), "n_unique": distinct,
           "leading_zero_rows": leading, "whitespace_rows": int((v != v.str.strip_chars()).sum()),
           "empty_strings": int((v == "").sum()),
           "placeholder_rows": int(v.str.to_lowercase().is_in(["na", "n/a", "nan", "null", "none", "tbd", "?"]).sum()),
           "top5": counts.head(5).to_dicts(), "bottom5": counts.tail(5).to_dicts(),
           "all_values_if_le30": sorted(v.unique().to_list()) if distinct <= 30 else None,
           "min_length": v.str.len_chars().min(), "max_length": v.str.len_chars().max()}
    if len(v) and parsed == len(v):
        rec.update({"min": finite(num.min()), "max": finite(num.max()), "mean": finite(num.mean()), "std": finite(num.std()),
                    "zero_rows": int((num == 0).sum()), "negative_rows": int((num < 0).sum()),
                    "nonfinite_rows": int((~num.is_finite()).sum())})
        for q, label in [(0.01, "p01"), (0.05, "p05"), (0.25, "p25"), (0.5, "p50"), (0.75, "p75"), (0.95, "p95"), (0.99, "p99")]:
            rec[label] = finite(num.quantile(q, interpolation="linear"))
        if rec["p25"] is not None and rec["p75"] is not None:
            iqr = rec["p75"] - rec["p25"]
            rec["iqr_outlier_rows"] = int(((num < rec["p25"] - 1.5 * iqr) | (num > rec["p75"] + 1.5 * iqr)).sum())
    elif len(v):
        rec.update({"min": v.min(), "max": v.max()})
    return rec


def profile():
    manifest = json.loads((OUT / "manifest.json").read_text())
    records, profiles, archive_hashes, special = [], [], [], []
    start = time.time()
    with tarfile.open(ROOT / (STEM + ".tar")) as tar, zipfile.ZipFile(ROOT / (STEM + ".zip")) as z:
        tm = {m.name.split("/", 1)[1]: m for m in tar if m.isfile()}
        for i, meta in enumerate(manifest):
            name = meta["path"]
            raw = tar.extractfile(tm[name]).read()
            digest = sha(raw)
            with z.open(STEM + "/" + name) as f:
                zdigest = hashlib.file_digest(f, "sha256").hexdigest()
            archive_hashes.append({"path": name, "tar_sha256": digest, "zip_sha256": zdigest, "equal": digest == zdigest})
            # Read every value as text first: no loss of leading zeros or code semantics.
            df = pl.read_csv(raw, infer_schema=False, encoding="utf8", raise_if_empty=True)
            del raw
            rec = {"path": name, "form": meta["form"], "year": int(meta["year"]),
                   "rows": df.height, "columns_count": df.width, "columns": df.columns,
                   "manifest_rows_match": df.height == meta["rows_synthetic"],
                   "manifest_columns_match": df.width == meta["columns_synthetic"],
                   "manifest_bytes_match": tm[name].size == meta["bytes_csv"],
                   "duplicate_rows_excess": df.height - df.n_unique(),
                   "all_null_rows": int(df.select(pl.all_horizontal(pl.all().is_null()).sum()).item())}
            candidates = []
            if "ID" in df.columns:
                candidates = [["ID"], ["ID", "KCP"], [c for c in df.columns if c not in {"VAL_NUM", "GR1"}]]
            elif "NOMER" in df.columns:
                candidates = [["NOMER"]]
                if "NOMP" in df.columns:
                    candidates.append(["NOMER", "NOMP"])
                if "KODNU" in df.columns:
                    candidates.append(["NOMER", "KODNU"])
            rec["candidate_keys"] = [{"columns": key, "distinct": df.n_unique(subset=key),
                                      "null_rows": int(df.select(pl.any_horizontal(pl.col(key).is_null()).sum()).item())} for key in candidates]
            rec["period_checks"] = {}
            for c in ["GOD", "GODO", "RYEAR"]:
                if c in df.columns:
                    rec["period_checks"][c + "_mismatch"] = int((df[c].cast(pl.Int64, strict=False) != int(meta["year"])).sum())
            quarter = re.search(r"/(\d)kv/", name)
            for c in ["KVARTAL", "kv"]:
                if c in df.columns:
                    num = df[c].cast(pl.Int64, strict=False)
                    rec["period_checks"][c + "_outside_1_4"] = int((~num.is_between(1, 4)).sum())
                    if quarter:
                        rec["period_checks"][c + "_path_mismatch"] = int((num != int(quarter[1])).sum())
            for c in ["RMONTH", "MES_ROJD", "mes", "DH_DATROZH_2"]:
                if c in df.columns:
                    num = df[c].cast(pl.Int64, strict=False)
                    rec["period_checks"][c + "_outside_1_12"] = int((~num.is_between(1, 12)).sum())
                    if c == "RMONTH" and quarter:
                        rec["period_checks"][c + "_quarter_end_mismatch"] = int((num != int(quarter[1]) * 3).sum())
            for c in ["GOD_ROJD", "DH_DATROZH_3"]:
                if c in df.columns:
                    num = df[c].cast(pl.Int64, strict=False)
                    rec["period_checks"][c + "_future"] = int((num > int(meta["year"])).sum())
                    rec["period_checks"][c + "_age_above_120"] = int((num < int(meta["year"]) - 120).sum())
            profiles.extend(profile_column(df[c], name) for c in df.columns)
            if "KCP" in df.columns:
                val = "VAL_NUM" if "VAL_NUM" in df.columns else "GR1"
                q = df.with_columns(pl.col(val).cast(pl.Float64)).group_by("KCP").agg(
                    pl.len().alias("n"), pl.col(val).min().alias("min"), pl.col(val).max().alias("max"),
                    pl.col(val).median().alias("median"), (pl.col(val) < 0).sum().alias("negative"))
                special.extend({"path": name, **row} for row in q.to_dicts())
            records.append(rec)
            dump(OUT / "table_profiles.json", records)
            # Checkpoint so even an interrupted long run retains its exact results.
            csvout(OUT / "column_profiles.csv", profiles)
            if i % 8 == 0 or meta["form"] in {"1t_god", "1t_kv", "2t"}:
                print(f"{i+1}/{len(manifest)} {name}: {df.height:,} rows; {time.time()-start:.1f}s", flush=True)
            del df
    dump(OUT / "column_profiles.json", profiles)
    csvout(OUT / "table_profiles.csv", [{k: v for k, v in r.items() if k != "columns"} for r in records])
    csvout(OUT / "archive_comparison.csv", archive_hashes)
    csvout(OUT / "enterprise_indicator_ranges.csv", special)
    print("COMPLETE", sum(r["rows"] for r in records), "rows", len(profiles), "column profiles", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "profile"])
    mode = parser.parse_args().mode
    if mode == "profile":
        import polars as pl
    globals()[mode]()
