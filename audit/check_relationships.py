"""Check keys, reference-code coverage, dates and domain rules without training."""
import collections
import itertools
import json
import re
import tarfile

import pandas as pd

from audit_data import ROOT, OUT, STEM, csvout, dump


def num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    manifest = json.loads((OUT/"manifest.json").read_text())
    docs = json.loads((OUT/"documents.json").read_text())
    checks, references, panels, domains = [], [], [], []
    reference_codes = collections.defaultdict(set)
    reference_quality = []
    for doc in docs:
        if doc["kind"] != "reference book":
            continue
        sheets = json.loads((ROOT/doc["text_path"]).with_suffix(".json").read_text())
        p = doc["path"].split("/")
        codes = []
        for sheet, rows in sheets.items():
            for row in rows:
                # Reference layouts differ: identify code-looking cells without dropping zeros.
                for cell in row:
                    text = str(int(cell)) if isinstance(cell, float) and cell.is_integer() else str(cell).strip().rstrip(",").strip()
                    if re.fullmatch(r"[A-Za-zА-Яа-я]?\d+(?:[.\-]\d+)*", text):
                        codes.append(text)
        reference_quality.append({"path":doc["path"],"code_like_cells":len(codes),"distinct_code_like_values":len(set(codes)),"method":"all numeric/dotted/hyphenated codes, optional letter prefix; Excel numeric integers normalized, surrounding whitespace and trailing comma removed; leading zeros preserved"})
        if p[1] == "d004":
            match = re.search(r"kv_vopr(\d+)",p[-1])
            if match:
                q = match[1]
                reference_codes[(p[2],"kv_vopr"+q)].update(codes)
                if q=="9": reference_codes[(p[2],"kv_vopr10")].update(codes)
        elif p[1]=="2t":
            c = "GSDPRS" if "ГСДПРС" in p[-1] else "SUO"
            reference_codes[(p[2],c)].update(codes)
    with tarfile.open(ROOT/(STEM+".tar")) as tar:
        tm = {m.name.split("/",1)[1]:m for m in tar if m.isfile()}
        def read(path, columns=None):
            with tar.extractfile(tm[path]) as f:
                return pd.read_csv(f,dtype=str,keep_default_na=False,usecols=columns)
        rosters = {str(y):read(f"d008/{y}/kontr_k.csv") for y in range(2021,2025)}
        household_keys = {y:set(df.NOMER) for y,df in rosters.items()}
        for year, roster in rosters.items():
            hh = roster.groupby("NOMER").agg(rows=("NOMP","size"),kol=("KOL_CHL","first"),kol_distinct=("KOL_CHL","nunique"),te=("TE","nunique"),k=("K","nunique"))
            domains.append({"path":f"d008/{year}/kontr_k.csv","households":len(hh),"roster_count_vs_KOL_CHL_mismatch":int((hh.rows!=num(hh.kol)).sum()),
                            "KOL_CHL_inconsistent_within_household":int((hh.kol_distinct>1).sum()),"TE_inconsistent":int((hh.te>1).sum()),"K_inconsistent":int((hh.k>1).sum()),
                            "duplicate_person_keys":int(roster.duplicated(["NOMER","NOMP"]).sum()),
                            "height_below_40_or_above_230":int(((num(roster.ROST)<40)|(num(roster.ROST)>230)).sum())})
        for a,b in itertools.combinations(rosters,2):
            panels.append({"left":"d008/"+a,"right":"d008/"+b,"key":"NOMER","left_distinct":len(household_keys[a]),"right_distinct":len(household_keys[b]),"intersection":len(household_keys[a]&household_keys[b])})
        for meta in manifest:
            if meta["form"] not in {"d002","d004","d006","d008","t001"}:
                continue
            path,year,form = meta["path"],meta["year"],meta["form"]
            df = rosters[year] if form=="d008" else read(path)
            if "NOMER" in df:
                roster = rosters[year]
                ref = roster[["NOMER","TE","K"]].drop_duplicates()
                assert not ref.duplicated("NOMER").any()
                joined = df.merge(ref,on="NOMER",how="left",suffixes=("","_ref"),validate="many_to_one",indicator=True)
                record = {"path":path,"rows":len(df),"unmatched_household_rows":int((joined._merge=="left_only").sum()),
                          "TE_mismatch":int((joined.TE!=joined.TE_ref).sum()),"K_mismatch":int((joined.K!=joined.K_ref).sum())}
                if "NOMP" in df:
                    cols=[c for c in ["NOMER","NOMP","POL","GOD_ROJD"] if c in roster]
                    person=df.merge(roster[cols],on=["NOMER","NOMP"],how="left",suffixes=("","_ref"),validate="many_to_one",indicator=True)
                    record["unmatched_person_rows"]=int((person._merge=="left_only").sum())
                    record["POL_mismatch"]=int(((person.POL!=person.POL_ref)&(person._merge=="both")).sum())
                    if "VOZ" in df and "GOD_ROJD" in person:
                        age = int(year)-num(person.GOD_ROJD)
                        record["VOZ_outside_birth_year_age_or_age_minus_1"]=int((num(person.VOZ).notna()&age.notna()&~((num(person.VOZ)==age)|(num(person.VOZ)==age-1))).sum())
                checks.append(record)
            if form=="d004":
                table=path.rsplit("/",1)[1][:-4]
                if "KODNU" in df and (year,table) in reference_codes:
                    valid=reference_codes[(year,table)]
                    missing=(df.KODNU!="")&~df.KODNU.isin(valid)
                    references.append({"path":path,"column":"KODNU","reference_year":year,"nonnull_rows":int((df.KODNU!="").sum()),"unmatched_rows":int(missing.sum()),"unmatched_codes":sorted(df.loc[missing,"KODNU"].unique().tolist()),"valid_codes_count":len(valid)})
                if table=="kv_vopr1":
                    domains.append({"path":path,"rows":len(df),"repeated_household_item_excess":int(df.duplicated(["NOMER","KODNU"]).sum()),"rows_in_repeated_household_item_groups":int(df.duplicated(["NOMER","KODNU"],keep=False).sum())})
            if form=="d006":
                record={"path":path,"living_area_above_total":int((num(df.J_PL)>num(df.OB_PL)).sum()),"living_area_nonpositive":int((num(df.J_PL)<=0).sum()),"rooms_nonpositive":int((num(df.KOL_K)<=0).sum()),"U_codes_outside_1_2":sum(int((~df[c].isin(["","1","2"])).sum()) for c in df if re.fullmatch(r"U\d+",c))}
                land_missing=0
                for j in range(1,9):
                    c=f"NOMP{j}"
                    active = df[c].ne("") & df[c].ne("0")
                    keys=set(zip(rosters[year].NOMER,num(rosters[year].NOMP)))
                    land_missing+=sum((a,b) not in keys for a,b in zip(df.loc[active,"NOMER"],num(df.loc[active,c])))
                record["unmatched_nonzero_land_person_refs"]=land_missing
                domains.append(record)
            if form=="t001":
                age=int(year)-num(df.DH_DATROZH_3)
                record={"path":path,"age_birth_year_inconsistent":int((num(df.vosr).notna()&age.notna()&~((num(df.vosr)==age)|(num(df.vosr)==age-1))).sum())}
                if "kv" in df:
                    record["month_quarter_inconsistent"]=int((num(df.kv).notna() & (((num(df.mes)-1)//3+1)!=num(df.kv))).sum())
                    record["quarter_missing"]=int(df.kv.eq("").sum())
                domains.append(record)
        # Same household, but subject and assessment refer to potentially different respondents.
        for year in rosters:
            a=read(f"d002/{year}/subject.csv")
            b=read(f"d002/{year}/ocenka.csv")
            j=a.merge(b,on="NOMER",suffixes=("_s","_o"),how="outer",validate="one_to_one",indicator=True)
            domains.append({"path":"d002/"+year+"/subject_vs_ocenka","unmatched_households":int((j._merge!="both").sum()),
                            "NOMP_mismatch":int((j.NOMP_s!=j.NOMP_o).sum()),"POL_mismatch":int((j.POL_s!=j.POL_o).sum()),"VOZ_mismatch":int((j.VOZ_s!=j.VOZ_o).sum())})
        # Prove exact-ID overlap across periods and forms; do not invent a suffix mapping.
        enterprise_ids={}
        for meta in manifest:
            if meta["form"] not in {"1t_god","1t_kv","2t"}:continue
            path=meta["path"]
            df=read(path,lambda c:c in {"ID","GSDPRS","SUO","SPPN1","SPPN2","SM"})
            enterprise_ids[path]=set(df.ID)
            for c in ["GSDPRS","SUO"]:
                if c in df and (meta["year"],c) in reference_codes:
                    valid=reference_codes[(meta["year"],c)]
                    bad=df[c].ne("")&~df[c].isin(valid)
                    references.append({"path":path,"column":c,"reference_year":meta["year"],"nonnull_rows":int(df[c].ne("").sum()),"unmatched_rows":int(bad.sum()),"unmatched_codes":sorted(df.loc[bad,c].unique().tolist()),"valid_codes_count":len(valid)})
            if "SPPN1" in df and "SPPN2" in df:
                domains.append({"path":path,"hire_year_before_birth_year":int((num(df.SPPN2)<num(df.SPPN1)).sum())})
            print("ID check",path,len(enterprise_ids[path]),flush=True)
        for a,b in itertools.combinations(enterprise_ids,2):
            panels.append({"left":a,"right":b,"key":"ID","left_distinct":len(enterprise_ids[a]),"right_distinct":len(enterprise_ids[b]),"intersection":len(enterprise_ids[a]&enterprise_ids[b])})
    csvout(OUT/"join_checks.csv",checks)
    csvout(OUT/"reference_checks.csv",references)
    csvout(OUT/"panel_checks.csv",panels)
    csvout(OUT/"domain_checks.csv",domains)
    csvout(OUT/"reference_inventory.csv",reference_quality)
    dump(OUT/"relationship_results.json",{"joins":checks,"references":references,"panels":panels,"domain":domains})
    print("RELATIONSHIP CHECKS COMPLETE",flush=True)


if __name__=="__main__":
    main()
