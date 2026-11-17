import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://api.fda.gov/drug/event.json"

AGE_CONVERSION = {
    "800": 10.0,
    "801": 1.0,
    "802": 1 / 12.0,
    "803": 1 / 52.0,
    "804": 1 / 365.0,
    "805": 1 / 8760.0,
}

def normalize_age(age_val, age_unit):
    if age_val is None:
        return None
    try:
        age = float(age_val)
        multiplier = AGE_CONVERSION.get(str(age_unit), 1.0)
        age_years = round(age * multiplier, 2)
        return age_years if 0 < age_years <= 115 else None
    except (ValueError, TypeError):
        return None

def extract_next_url(link_header, api_key):
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    if not match:
        return None
    next_url = match.group(1)
    if "api_key=" not in next_url:
        sep = "&" if "?" in next_url else "?"
        next_url = f"{next_url}{sep}api_key={api_key}"
    return next_url

def parse_report(report):
    patient = report.get("patient", {})
    
    age = normalize_age(
        patient.get("patientonsetage"),
        patient.get("patientonsetageunit")
    )
    sex = patient.get("patientsex")
    if age is None or str(sex) not in ("1", "2"):
        return None

    drugs_list = patient.get("drug", [])
    primary_suspects = []
    all_drugs = []
    indications = []

    for d in drugs_list:
        openfda_info = d.get("openfda", {})
        generic_names = openfda_info.get("generic_name", [])
        drug_name = generic_names[0].upper() if generic_names else d.get("medicinalproduct", "").upper().strip()
        
        if drug_name:
            all_drugs.append(drug_name)
            if str(d.get("drugcharacterization")) == "1":
                primary_suspects.append(drug_name)
                
        if d.get("drugindication"):
            indications.append(d.get("drugindication").upper().strip())

    if not primary_suspects:
        return None

    reactions = [
        r.get("reactionmeddrapt").upper().strip()
        for r in patient.get("reaction", [])
        if r.get("reactionmeddrapt")
    ]
    if not reactions:
        return None

    return {
        "safetyreportid": str(report.get("safetyreportid")),
        "receivedate": report.get("receivedate"),
        "age": age,
        "sex": int(sex),
        "primary_suspect_drug": primary_suspects[0],
        "all_drugs": "|".join(sorted(set(all_drugs))),
        "polypharmacy_count": len(set(all_drugs)),
        "indications": "|".join(sorted(set(indications))) if indications else "UNKNOWN",
        "reactions": "|".join(sorted(set(reactions))),
        "is_serious": int(report.get("serious", 0)),
        "is_hospitalized": int(report.get("seriousnesshospitalization", 0)),
        "is_death": int(report.get("seriousnessdeath", 0)),
    }

def fetch_faers_dataset(target_records=25000, date_range="[20240101 TO 20260630]", api_key=None):
    api_key = api_key or os.getenv("OPENFDA_API_KEY")
    if not api_key:
        raise ValueError("OPENFDA_API_KEY missing!")

    params = {
        "api_key": api_key,
        "search": f"receivedate:{date_range} AND _exists_:patient.patientonsetage AND _exists_:patient.patientsex",
        "sort": "receivedate:desc",
        "limit": 1000
    }

    url = BASE_URL
    collected = []
    seen_ids = set()

    pbar = tqdm(total=target_records, desc="Fetching Clean FAERS ICSRs")

    while url and len(collected) < target_records:
        try:
            response = requests.get(url, params=params if url == BASE_URL else None, timeout=30)
            
            if response.status_code == 429:
                print("\nRate limit hit. Waiting 15s...")
                time.sleep(15)
                continue
            elif response.status_code != 200:
                print(f"\nAPI Error {response.status_code}: {response.text[:200]}")
                break

            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            for report in results:
                rep_id = str(report.get("safetyreportid"))
                if rep_id in seen_ids:
                    continue
                
                parsed = parse_report(report)
                if parsed:
                    seen_ids.add(rep_id)
                    collected.append(parsed)
                    pbar.update(1)
                    if len(collected) >= target_records:
                        break

            url = extract_next_url(response.headers.get("Link"), api_key)
            time.sleep(0.25)

        except requests.exceptions.RequestException as e:
            print(f"\nNetwork error: {e}. Retrying in 5s...")
            time.sleep(5)

    pbar.close()
    if not collected:
        return pd.DataFrame()

    df = pd.DataFrame(collected)
    df.drop_duplicates(subset=["age", "sex", "primary_suspect_drug", "reactions"], inplace=True)
    return df

if __name__ == "__main__":
    df_clean = fetch_faers_dataset(target_records=25000)
    output_path = "data/processed/faers_clean_25k.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_clean)} deduplicated records to {output_path}")
    print("\nTop 15 Reported Reactions:")
    print(df_clean["reactions"].str.split("|").explode().value_counts().head(15))
