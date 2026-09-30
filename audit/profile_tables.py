"""Exact full-data profiles using installed pandas; no training or source writes."""
import collections
import hashlib
import io
import json
import re
import tarfile
import time
import zipfile

import numpy as np
import pandas as pd

from audit_data import ROOT, OUT, STEM, csvout, dump, sha


def column_profile(s, path):
    counts = s.value_counts(dropna=True)
    counts = counts[counts > 0]
    freq = counts.to_numpy(dtype=np.int64)
    labels = counts.index.astype(str).to_numpy()
    order = sorted(range(len(freq)), key=lambda i: (-int(freq[i]), labels[i]))
    def examples(ix):
        return [{s.name: str(labels[i]), "count": int(freq[i])} for i in ix]
    numeric = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(dtype=float)
    integer = all(re.fullmatch(r"[+-]?\d+", x) for x in labels)
    n, valid = len(s), int(freq.sum())
    rec = {"path": path, "column": s.name, "storage_type": "CSV text",
           "inferred_type": "empty" if not valid else "integer" if integer else "numeric" if not np.isnan(numeric).any() else "string",
           "rows": n, "nulls": n-valid, "null_pct": round(100*(n-valid)/n, 6), "n_unique": len(counts),
           "leading_zero_rows": sum(int(f) for x, f in zip(labels, freq) if re.match(r"^0\d", x)),
           "whitespace_rows": sum(int(f) for x, f in zip(labels, freq) if x != x.strip()),
           "empty_strings": sum(int(f) for x, f in zip(labels, freq) if x == ""),
           "placeholder_rows": sum(int(f) for x, f in zip(labels, freq) if x.lower() in {"na","n/a","nan","null","none","tbd","?"}),
           "top5": examples(order[:5]), "bottom5": examples(order[-5:]),
           "all_values_if_le30": sorted(labels.tolist()) if len(counts) <= 30 else None,
           "min_length": min(map(len, labels), default=None), "max_length": max(map(len, labels), default=None)}
    if valid and not np.isnan(numeric).any() and np.isfinite(numeric).all():
        idx = np.argsort(numeric)
        vals, weights = numeric[idx], freq[idx]
        cdf = np.cumsum(weights)
        mean = float(np.average(vals, weights=weights))
        rec.update({"min": float(vals[0]), "max": float(vals[-1]), "mean": mean,
                    "std": float(np.sqrt(np.sum(weights*(vals-mean)**2)/(valid-1))) if valid > 1 else None,
                    "zero_rows": int(freq[numeric == 0].sum()), "negative_rows": int(freq[numeric < 0].sum()),
                    "nonfinite_rows": 0})
        for q, label in [(0.01,"p01"),(0.05,"p05"),(0.25,"p25"),(0.5,"p50"),(0.75,"p75"),(0.95,"p95"),(0.99,"p99")]:
            pos = (valid-1)*q
            low, high = int(np.floor(pos)), int(np.ceil(pos))
            a, b = vals[np.searchsorted(cdf, low+1)], vals[np.searchsorted(cdf, high+1)]
            rec[label] = float(a+(b-a)*(pos-low))
        iqr = rec["p75"]-rec["p25"]
        rec["iqr_outlier_rows"] = int(freq[(numeric < rec["p25"]-1.5*iqr) | (numeric > rec["p75"]+1.5*iqr)].sum())
    elif valid:
        rec.update({"min": str(min(labels)), "max": str(max(labels)),
                    "nonfinite_rows": int(freq[~np.isfinite(numeric) & ~np.isnan(numeric)].sum())})
    return rec


def main():
    manifest = json.loads((OUT/"manifest.json").read_text())
    tables, columns, hashes, indicators = [], [], [], []
    start = time.time()
    with tarfile.open(ROOT/(STEM+".tar")) as tar, zipfile.ZipFile(ROOT/(STEM+".zip")) as z:
        tm = {m.name.split("/",1)[1]:m for m in tar if m.isfile()}
        for i, meta in enumerate(manifest):
            path = meta["path"]
            raw = tar.extractfile(tm[path]).read()
            digest = sha(raw)
            with z.open(STEM+"/"+path) as f:
                zdigest = hashlib.file_digest(f,"sha256").hexdigest()
            hashes.append({"path":path,"tar_sha256":digest,"zip_sha256":zdigest,"equal":digest==zdigest})
            df = pd.read_csv(io.BytesIO(raw),dtype="category",keep_default_na=False,na_values=[""],encoding="utf-8")
            del raw
            rec = {"path":path,"form":meta["form"],"year":int(meta["year"]),"rows":len(df),"columns_count":len(df.columns),"columns":df.columns.tolist(),
                   "manifest_rows_match":len(df)==meta["rows_synthetic"],"manifest_columns_match":len(df.columns)==meta["columns_synthetic"],
                   "manifest_bytes_match":tm[path].size==meta["bytes_csv"],"duplicate_rows_excess":int(df.duplicated().sum()),
                   "all_null_rows":int(df.isna().all(axis=1).sum())}
            candidates=[]
            if "ID" in df:
                candidates=[["ID"],["ID","KCP"],[c for c in df if c not in {"GR1","VAL_NUM"}]]
            elif "NOMER" in df:
                candidates=[["NOMER"]]
                candidates += [["NOMER",c] for c in ["NOMP","KODNU"] if c in df]
            elif "RSPD" in df:
                candidates=[["RSPD"],[c for c in ["TE","K","RSPD","kv","mes"] if c in df]]
            rec["candidate_keys"]=[{"columns":key,"distinct":int(len(df)-df.duplicated(subset=key).sum()),"null_rows":int(df[key].isna().any(axis=1).sum())} for key in candidates]
            current=[column_profile(df[c],path) for c in df]
            columns.extend(current)
            rec["period_checks"]={}
            quarter=re.search(r"/(\d)kv/",path)
            for c in df:
                if c not in {"GOD","GODO","RYEAR","KVARTAL","kv","RMONTH","MES_ROJD","mes","DH_DATROZH_2","GOD_ROJD","DH_DATROZH_3","SM","SPPN1","SPPN2"}:continue
                cnt=df[c].value_counts(dropna=True)
                vals=pd.to_numeric(pd.Series(cnt.index.astype(str)),errors="coerce").to_numpy()
                weights=cnt.to_numpy()
                check={}
                if c in {"GOD","GODO","RYEAR"}:check["path_mismatch"]=vals!=int(meta["year"])
                if c in {"KVARTAL","kv"}:
                    check["outside_1_4"]=(vals<1)|(vals>4)
                    if quarter:check["path_mismatch"]=vals!=int(quarter[1])
                if c in {"RMONTH","MES_ROJD","mes","DH_DATROZH_2","SM"}:
                    check["outside_1_12"]=(vals<1)|(vals>12)
                    if c=="RMONTH" and quarter:check["quarter_end_mismatch"]=vals!=int(quarter[1])*3
                if c in {"GOD_ROJD","DH_DATROZH_3","SPPN1","SPPN2"}:
                    check["future_year"]=vals>int(meta["year"])
                    check["before_year_minus_120"]=(vals<int(meta["year"])-120)
                rec["period_checks"].update({c+"_"+label:int(weights[mask].sum()) for label,mask in check.items()})
            if "KCP" in df:
                val="VAL_NUM" if "VAL_NUM" in df else "GR1"
                for key, sub in df.groupby("KCP",observed=True):
                    v=pd.to_numeric(sub[val].astype(str),errors="coerce")
                    indicators.append({"path":path,"KCP":key,"n":len(sub),"nulls":int(v.isna().sum()),"min":float(v.min()),"max":float(v.max()),"median":float(v.median()),"negative":int((v<0).sum())})
            tables.append(rec)
            dump(OUT/"table_profiles.json",tables)
            csvout(OUT/"column_profiles.csv",columns)
            print(f"{i+1}/{len(manifest)} {path}: {len(df):,} rows, {time.time()-start:.1f}s",flush=True)
            del df
    dump(OUT/"column_profiles.json",columns)
    csvout(OUT/"table_profiles.csv",[{k:v for k,v in r.items() if k!="columns"} for r in tables])
    csvout(OUT/"archive_comparison.csv",hashes)
    csvout(OUT/"enterprise_indicator_ranges.csv",indicators)
    print("COMPLETE",sum(r["rows"] for r in tables),"rows",len(columns),"column profiles",flush=True)


if __name__=="__main__":
    main()
