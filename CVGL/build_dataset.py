#!/usr/bin/env python3
"""Build the CVUSA-format trees for the PhotoMappers dataset (11,737 groups) and the
gid <-> spreadsheet mapping.

Source of truth is the split spreadsheet (default data/split_80%_11737.xlsx, sheets '80%train' =
9,390 rows and '20%test' = 2,347 rows).  Every row is one *group* = one VGI photo plus the RSI tile
and SVI panorama at its coordinate.  Images are coordinate-named ("<lat>, <lon>.jpg") and come in
two batches:

  batch "old" (8,780 rows)  <old-dir>/{RSI,SVI,VGI}/             name = f"{Y}, {X}"
  batch "new" (2,957 rows)  <new-dir>/{RSI,SVI,VGI}_2957/        name = f"{X}, {Y}"

NOTE the swap: in the 2,957 newly added rows the spreadsheet's X column holds the latitude and Y the
longitude, the opposite of the original 8,780 rows.  Which convention a row uses is decided by which
of the two candidate names actually exists in that row's batch folder; the resolution is exact (no
nearest-neighbour fallback) and every row resolves uniquely.

gids are assigned train-first, in sheet order:  train = 0..9389, test = 9390..11736.

Outputs (all symlinks, no image is copied):
  <out>/disaster_vgi/       streetview/panos/<gid>.jpg -> VGI   (query)
                            bingmap/<gid>.jpg          -> RSI   (reference)
  <out>/disaster_svi2vgi/   streetview/panos/<gid>.jpg -> SVI   (query)
                            bingmap/<gid>.jpg          -> VGI   (reference)
  each with splits/{train,val}-19zl.csv  (val = the test sheet)
  <mapping>                 gid -> spreadsheet row + true coordinates + file names

Usage:  python build_dataset.py [--dry-run]
"""
import argparse, csv, os, pathlib, sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
SHEETS = [("80%train", "train", "train-19zl.csv"), ("20%test", "test", "val-19zl.csv")]
TREES = ("disaster_vgi", "disaster_svi2vgi")
# spreadsheet columns kept in the mapping file so the table can be joined on whatever key is used
KEEP = ["URLName", "ParentObjectId", "AttachmentId", "GlobalId", "AttachmentName",
        "SaveRelativePath", "DownloadURL", "Incident", "Y", "X"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--xlsx", type=pathlib.Path, default=ROOT / "data" / "split_80%_11737.xlsx")
ap.add_argument("--old-dir", type=pathlib.Path, default=ROOT / "data" / "Dataset20260630",
                help="original 8,780-group batch with RSI/ SVI/ VGI/")
ap.add_argument("--new-dir", type=pathlib.Path, default=ROOT / "data" / "Dataset_new2957",
                help="2,957 newly added groups with RSI_2957/ SVI_2957/ VGI_2957/")
ap.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "cvformat",
                help="where the CVUSA-format symlink trees are written")
ap.add_argument("--mapping", type=pathlib.Path, default=ROOT / "output" / "mapping_11737.csv")
ap.add_argument("--dry-run", action="store_true", help="resolve and verify only, write nothing")
A = ap.parse_args()
OLD, NEW, OUT = A.old_dir.absolute(), A.new_dir.absolute(), A.out.absolute()


def modality_dir(batch, mod):
    return (OLD / mod) if batch == "old" else (NEW / f"{mod}_2957")


def resolve(row):
    """-> (batch, image_name) for one spreadsheet row, or (None, None) if neither exists."""
    yx, xy = f"{row.Y}, {row.X}", f"{row.X}, {row.Y}"
    if (OLD / "RSI" / f"{yx}.jpg").exists():
        return "old", yx
    if (NEW / "RSI_2957" / f"{xy}.jpg").exists():
        return "new", xy
    return None, None


def link(target, dst, dry):
    if dry:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(target)


def main():
    dry = A.dry_run
    xl = pd.ExcelFile(A.xlsx)
    gid, mapping, missing, unresolved = 0, [], [], []
    split_rows = {}

    for sheet, split, csvname in SHEETS:
        df = xl.parse(sheet)
        rows = []
        for i, row in enumerate(df.itertuples(index=False)):
            pad = f"{gid:07d}"
            batch, name = resolve(row)
            if batch is None:
                unresolved.append((gid, sheet, i + 2, row.Y, row.X))
                rows.append(f"bingmap/{pad}.jpg,streetview/panos/{pad}.jpg,")
                gid += 1
                continue
            # latitude/longitude in the true sense, independent of the column swap
            lat, lon = (row.Y, row.X) if batch == "old" else (row.X, row.Y)
            srcs = {m: modality_dir(batch, m) / f"{name}.jpg" for m in ("RSI", "SVI", "VGI")}
            absent = [m for m, p in srcs.items() if not p.exists()]
            if absent:
                missing.append((gid, sheet, i, name, ",".join(absent)))
            else:
                link(srcs["VGI"], OUT / "disaster_vgi" / "streetview" / "panos" / f"{pad}.jpg", dry)
                link(srcs["RSI"], OUT / "disaster_vgi" / "bingmap" / f"{pad}.jpg", dry)
                link(srcs["SVI"], OUT / "disaster_svi2vgi" / "streetview" / "panos" / f"{pad}.jpg", dry)
                link(srcs["VGI"], OUT / "disaster_svi2vgi" / "bingmap" / f"{pad}.jpg", dry)
            rows.append(f"bingmap/{pad}.jpg,streetview/panos/{pad}.jpg,")
            rec = {"gid": gid, "split": split, "excel_sheet": sheet, "excel_row_0based": i,
                   "excel_row_in_file": i + 2, "batch": batch, "image_name": f"{name}.jpg",
                   "latitude": lat, "longitude": lon,
                   "vgi_path": os.path.relpath(srcs["VGI"], ROOT),
                   "rsi_path": os.path.relpath(srcs["RSI"], ROOT),
                   "svi_path": os.path.relpath(srcs["SVI"], ROOT)}
            rec.update({k: getattr(row, k, "") for k in KEEP if k in df.columns})
            mapping.append(rec)
            gid += 1
        split_rows[csvname] = rows

    if not dry:
        for tree in TREES:
            sp = OUT / tree / "splits"
            sp.mkdir(parents=True, exist_ok=True)
            for csvname, rows in split_rows.items():
                (sp / csvname).write_text("\n".join(rows) + "\n")
        A.mapping.parent.mkdir(parents=True, exist_ok=True)
        with open(A.mapping, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mapping[0].keys()))
            w.writeheader()
            w.writerows(mapping)

    n_old = sum(m["batch"] == "old" for m in mapping)
    print(f"rows={len(mapping)}  old-batch={n_old}  new-batch={len(mapping) - n_old}")
    for csvname, rows in split_rows.items():
        print(f"  splits/{csvname}: {len(rows)} rows")
    if unresolved:
        print(f"UNRESOLVED: {len(unresolved)} rows have no image in either batch "
              f"(gid, sheet, xlsx row, Y, X) -- is the download complete?")
        for u in unresolved[:20]:
            print("   ", u)
    if missing or unresolved:
        print(f"MISSING images for {len(missing)} groups (gid, sheet, row, name, modalities):")
        for m in missing[:20]:
            print("   ", m)
        sys.exit(1)
    print("all three modalities present for every group")
    if not dry:
        for tree in TREES:
            q = len(list((OUT / tree / "streetview" / "panos").glob("*.jpg")))
            r = len(list((OUT / tree / "bingmap").glob("*.jpg")))
            print(f"  {tree}: query={q} reference={r}")
        print(f"mapping -> {A.mapping}")


if __name__ == "__main__":
    main()
