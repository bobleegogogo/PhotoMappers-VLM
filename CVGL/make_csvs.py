#!/usr/bin/env python3
"""Join the top-10 retrieval CSVs written by predict.py onto the spreadsheet keys and write the
deliverables for the PhotoMappers study.  CPU only; rerun freely after predict.py.

  predictions_11737_top1.csv
      one row per test group (2,347) with, for both fusion systems (VGI-SVI-RSI and
      VGI-SVI-RSI + Sinkhorn), the top-1 retrieved group, whether it is the correct one, the
      great-circle distance between the retrieved and the true location, and the rank of the true
      group (blank if outside the top 10).  Carries excel_sheet/excel_row_in_file/GlobalId/
      AttachmentId/AttachmentName/DownloadURL/Incident and the true lat/lon, so it joins onto
      '20%test' of split_80%_11737.xlsx directly (use excel_row_in_file; GlobalId is not unique).

  predictions_11737_top10_long.csv
      long format, 10 rows per query per system: system, query, rank, retrieved, score,
      is_match, error_km.  Input for the VLM re-ranking / explanation experiments.

  breakdown_11737.csv
      top-1 accuracy and km error of both systems, split into the 1,756 test groups of the
      original 8,780-group batch and the 591 newly added ones.

Usage:
  python make_csvs.py                       # reads output/dinov3l_11737_*_top10.csv
  python make_csvs.py --tag my_run --suffix _my_run
"""
import argparse, csv, math, os, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out-dir", default=str(ROOT / "output"), help="where predict.py wrote its CSVs")
ap.add_argument("--mapping", default=str(ROOT / "output" / "mapping_11737.csv"))
ap.add_argument("--tag", default="dinov3l_11737", help="prefix of the top-10 CSVs to read")
ap.add_argument("--suffix", default="", help="suffix for the output filenames")
ap.add_argument("--label", default="DINOv3-L", help="system label")
A = ap.parse_args()
OUT, TAG, SFX = A.out_dir, A.tag, A.suffix
SYSTEMS = [("fusion", f"{A.label} (VGI-SVI-RSI)", f"{OUT}/{TAG}_fusion_top10.csv"),
           ("fusion_sinkhorn", f"{A.label} (VGI-SVI-RSI + Sinkhorn)",
            f"{OUT}/{TAG}_fusion_sinkhorn_top10.csv")]
KEYS = ["excel_sheet", "excel_row_in_file", "batch", "GlobalId", "ParentObjectId", "AttachmentId",
        "AttachmentName", "DownloadURL", "Incident"]


def haversine(a, b):
    R = 6371.0088
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return R * 2 * math.asin(math.sqrt(h))


mapping = {}
with open(A.mapping) as f:
    for r in csv.DictReader(f):
        r["latitude"], r["longitude"] = float(r["latitude"]), float(r["longitude"])
        mapping[int(r["gid"])] = r

ranks = {}          # (sys, query_gid) -> list of rows sorted by rank
for skey, _, path in SYSTEMS:
    with open(path) as f:
        for r in csv.DictReader(f):
            ranks.setdefault((skey, int(r["query_gid"])), []).append(r)
for v in ranks.values():
    v.sort(key=lambda r: int(r["rank"]))

queries = sorted({q for (s, q) in ranks})
print(f"{len(queries)} test queries x {len(SYSTEMS)} systems")

# --- long format -----------------------------------------------------------------------------
with open(f"{OUT}/predictions_11737{SFX}_top10_long.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "query_gid", "query_image", "query_lat", "query_lon", "rank",
                "retrieved_gid", "retrieved_image", "retrieved_lat", "retrieved_lon",
                "score", "is_match", "error_km"])
    for skey, label, _ in SYSTEMS:
        for q in queries:
            mq = mapping[q]
            for r in ranks[(skey, q)]:
                mr = mapping[int(r["retrieved_gid"])]
                d = haversine((mq["latitude"], mq["longitude"]), (mr["latitude"], mr["longitude"]))
                w.writerow([label, q, mq["image_name"], mq["latitude"], mq["longitude"],
                            r["rank"], r["retrieved_gid"], mr["image_name"], mr["latitude"],
                            mr["longitude"], r["score"], r["is_match"], round(d, 4)])

# --- top-1 wide ------------------------------------------------------------------------------
cols = ["gid", "image_name", "latitude", "longitude"] + KEYS
for skey, _, _ in SYSTEMS:
    cols += [f"{skey}_top1_gid", f"{skey}_top1_image", f"{skey}_top1_lat", f"{skey}_top1_lon",
             f"{skey}_top1_score", f"{skey}_correct", f"{skey}_error_km",
             f"{skey}_rank_of_true"]
hits = {s: 0 for s, _, _ in SYSTEMS}
with open(f"{OUT}/predictions_11737{SFX}_top1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for q in queries:
        mq = mapping[q]
        row = {"gid": q, "image_name": mq["image_name"], "latitude": mq["latitude"],
               "longitude": mq["longitude"], **{k: mq.get(k, "") for k in KEYS}}
        for skey, _, _ in SYSTEMS:
            rs = ranks[(skey, q)]
            top = rs[0]
            mr = mapping[int(top["retrieved_gid"])]
            true_rank = next((int(r["rank"]) for r in rs if int(r["retrieved_gid"]) == q), "")
            ok = int(top["is_match"])
            hits[skey] += ok
            row.update({f"{skey}_top1_gid": top["retrieved_gid"], f"{skey}_top1_image": mr["image_name"],
                        f"{skey}_top1_lat": mr["latitude"], f"{skey}_top1_lon": mr["longitude"],
                        f"{skey}_top1_score": top["score"], f"{skey}_correct": ok,
                        f"{skey}_error_km": round(haversine((mq["latitude"], mq["longitude"]),
                                                            (mr["latitude"], mr["longitude"])), 4),
                        f"{skey}_rank_of_true": true_rank})
        w.writerow(row)

for skey, label, _ in SYSTEMS:
    print(f"  {label:38} top-1 correct {hits[skey]}/{len(queries)} = "
          f"{100.0 * hits[skey] / len(queries):.2f}%")

# --- old-vs-new breakdown --------------------------------------------------------------------
with open(f"{OUT}/predictions_11737{SFX}_top1.csv") as f:
    pred = list(csv.DictReader(f))
with open(f"{OUT}/breakdown_11737{SFX}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "subset", "n", "top1_accuracy", "median_error_km", "mean_error_km",
                "pct_within_1km", "pct_within_25km"])
    print()
    for skey, label, _ in SYSTEMS:
        for subset in ("all", "GRSM test groups", "newly added"):
            sub = pred if subset == "all" else [
                r for r in pred if r["batch"] == ("old" if subset.startswith("GRSM") else "new")]
            e = sorted(float(r[f"{skey}_error_km"]) for r in sub)
            acc = 100.0 * sum(int(r[f"{skey}_correct"]) for r in sub) / len(sub)
            row = [label, subset, len(sub), round(acc, 2), round(statistics.median(e), 2),
                   round(sum(e) / len(e), 2),
                   round(100.0 * sum(x <= 1 for x in e) / len(e), 2),
                   round(100.0 * sum(x <= 25 for x in e) / len(e), 2)]
            w.writerow(row)
            print(f"  {label:36} {subset:17} n={len(sub):5} top-1={acc:6.2f}%  "
                  f"median={statistics.median(e):8.2f}km")

print(f"\nwrote {OUT}/predictions_11737{SFX}_top1.csv, {OUT}/predictions_11737{SFX}_top10_long.csv, "
      f"{OUT}/breakdown_11737{SFX}.csv")
