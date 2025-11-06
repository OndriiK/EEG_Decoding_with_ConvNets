# run_combined.py
"""
Run BCIC and HGD and create the 'Combined' dataset evaluation by pooling subject accuracies.
"""

import json
import core
import dataset_bcic as bcic
import dataset_hgd as hgd
from datetime import datetime

BCIC_JSON = f"deepconvnet_{bcic.NAME}_results.json" # deepconvnet_bcic_iv_2a_results.json
HGD_JSON  = f"deepconvnet_{hgd.NAME}_results.json"  # deepconvnet_hgd_results.json
COMBINED_JSON = "deepconvnet_combined_results.json"
COMBINED_LOG  = "deepconvnet_combined_results.log"


def _run_and_load(ds_module, max_epochs):
    json_path = f"deepconvnet_{ds_module.NAME}_results.json"
    core.run_all(ds_module, max_epochs=max_epochs)
    with open(json_path, "r") as f:
        return json.load(f)


def _pooled_stats(values):
    import numpy as np
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()) if arr.size else float("nan"), float(arr.std()) if arr.size else float("nan")


def main(max_epochs=800):
    bcic_results = _run_and_load(bcic, max_epochs)
    hgd_results  = _run_and_load(hgd,  max_epochs)

    #(BCIC band, HGD band) -> combined label
    band_map = {
        "0-f_end": ("0-38",  "0-125"),
        "4-f_end": ("4-38",  "4-125"),
    }

    combined = {}
    for label, (bcic_key, hgd_key) in band_map.items():
        bcic_accs = bcic_results.get(bcic_key, {}).get("accuracies", [])
        hgd_accs  = hgd_results.get(hgd_key,  {}).get("accuracies", [])
        pooled = list(bcic_accs) + list(hgd_accs)
        mean, std = _pooled_stats(pooled)
        combined[label] = {
            "accuracies": pooled,
            "mean": mean,
            "std": std,
            "N": len(pooled),
            "parts": {
                "bcic_key": bcic_key,
                "bcic_N": len(bcic_accs),
                "hgd_key": hgd_key,
                "hgd_N": len(hgd_accs),
            },
        }

    # Save and print a short summary
    with open(COMBINED_JSON, "w") as f:
        json.dump(combined, f, indent=2)

    with open(COMBINED_LOG, "w") as f:
        f.write(f"Combined results - {datetime.now()}\n")
        f.write("="*72 + "\n\n")
        for label in ["0-f_end", "4-f_end"]:
            r = combined[label]
            f.write(f"{label}:\n")
            f.write(f"  Mean: {r['mean']:.3f}\n" if r['mean']==r['mean'] else "  Mean: nan\n")
            f.write(f"  Std:  {r['std']:.3f}\n"  if r['std']==r['std']   else "  Std: nan\n")
            f.write(f"  N:    {r['N']}\n")
            f.write(f"  From: BCIC[{r['parts']['bcic_key']}] + HGD[{r['parts']['hgd_key']}]\n\n")

    print("\n=== COMBINED SUMMARY ===")
    for label in ["0-f_end", "4-f_end"]:
        r = combined[label]
        mean = f"{r['mean']:.3f}" if r['mean']==r['mean'] else "nan"
        std  = f"{r['std']:.3f}"  if r['std']==r['std']   else "nan"
        print(f"{label}: Mean={mean}, Std={std}, N={r['N']} "
              f"(BCIC={r['parts']['bcic_key']}, HGD={r['parts']['hgd_key']})")
    print(f"\nSaved: {COMBINED_JSON} and {COMBINED_LOG}")


if __name__ == "__main__":
    main(max_epochs=800)
