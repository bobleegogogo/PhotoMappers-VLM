#!/usr/bin/env python3
"""Evaluate the PhotoMappers cross-view geo-localization model on the test split and write the
recall table plus the per-system top-10 retrieval CSVs.

Four systems, all from one feature extraction:
  VGI-RSI                  VGI photo -> RSI satellite gallery        (branch trained with --task vgi_rsi)
  VGI-SVI                  VGI photo -> SVI street-view gallery      (branch trained with --task svi_vgi)
  VGI-SVI-RSI              late fusion: w * z(S_svi) + (1 - w) * z(S_rsi), w = 0.6
  VGI-SVI-RSI + Sinkhorn   the fused matrix after Sinkhorn doubly-stochastic normalisation
                           (temperature 0.3, 50 iterations), exploiting the 1:1 query<->gallery bijection

Query = VGI photo (384x384 for the RSI branch, 336x336 for the SVI branch).  Galleries = RSI tiles
(384x384) and SVI panoramas (336x672).  Gallery = the same test groups as the queries.

Outputs (in --out-dir):
  recall_<tag>.csv                                   R@1 / R@5 / R@10 / R@1% per system
  <tag>_{rsi_only,svi_only,fusion,fusion_sinkhorn}_top10.csv   10 ranks per query with scores

Usage:
  python predict.py                                  # bundled checkpoints, data/cvformat, output/
  python predict.py --vgi-rsi checkpoints/vgi_rsi --svi-vgi checkpoints/svi_vgi --tag my_run
"""
import argparse, csv, glob, os, pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from sample4geo.model import TimmModel
from sample4geo.dataset.cvusa import CVUSADatasetEval
from sample4geo.transforms import get_transforms_val
from sample4geo.trainer import predict

ROOT = pathlib.Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default=str(ROOT / "data" / "cvformat"), help="output of build_dataset.py")
ap.add_argument("--vgi-rsi", default=str(ROOT / "checkpoints" / "vgi_rsi"),
                help="VGI<->RSI weights file, or a run dir (best weights_e*_<R@1>.pth is used)")
ap.add_argument("--svi-vgi", default=str(ROOT / "checkpoints" / "svi_vgi"),
                help="SVI<->VGI weights file, or a run dir")
ap.add_argument("--mapping", default=str(ROOT / "output" / "mapping_11737.csv"))
ap.add_argument("--out-dir", default=str(ROOT / "output"))
ap.add_argument("--tag", default="dinov3l_11737", help="output filename prefix")
ap.add_argument("--label", default="DINOv3-L", help="system label in the printed/CSV table")
ap.add_argument("--model", default="vit_large_patch16_dinov3")
ap.add_argument("--fusion-w", type=float, default=0.6, help="weight of the SVI term in the fusion")
ap.add_argument("--sinkhorn-t", type=float, default=0.3)
ap.add_argument("--sinkhorn-iters", type=int, default=50)
ap.add_argument("--batch-size", type=int, default=64)
A = ap.parse_args()


class Cfg:
    device = torch.device("cuda")
    normalize_features = True
    verbose = False


def best_weights(path):
    """A weights file is used as is; for a run dir take the weights_e*_<R@1>.pth with the best R@1."""
    if os.path.isfile(path):
        return path
    ws = glob.glob(os.path.join(path, "weights_e*_*.pth"))
    if not ws:
        raise FileNotFoundError(f"no weights_e*_*.pth in {path}")
    return max(ws, key=lambda p: float(p.rsplit("_", 1)[1][:-4]))


def recall_at_k(dist, q_ids, g_ids, ranks=(1, 5, 10)):
    q_ids, g_ids = np.asarray(q_ids), np.asarray(g_ids)
    order = np.argsort(dist, axis=1)                       # ascending distance
    rank = np.full(dist.shape[0], 10**9, dtype=np.int64)
    for i in range(dist.shape[0]):
        hit = np.where(g_ids[order[i]] == q_ids[i])[0]
        if len(hit):
            rank[i] = hit[0]
    out = {f"R@{k}": round(100.0 * np.mean(rank < k), 2) for k in ranks}
    out["R@1%"] = round(100.0 * np.mean(rank < max(1, round(0.01 * len(g_ids)))), 2)
    return out


def sort_by_gid(f, i):
    i = i.cpu().numpy()
    return f.cpu().numpy().astype(np.float32)[np.argsort(i)], np.sort(i)


def zscore(M):
    return (M - M.mean()) / (M.std() + 1e-9)


def sinkhorn(S, temp, it):
    P = np.exp((S - S.max()) / temp).astype(np.float64)
    for _ in range(it):
        P /= P.sum(1, keepdims=True) + 1e-12
        P /= P.sum(0, keepdims=True) + 1e-12
    return P


def make_model(ckpt, sz, dyn):
    m = TimmModel(A.model, pretrained=False, img_size=sz, dynamic_img_size=dyn)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=True)
    return m.cuda().eval()


def feats(m, tree, item, sz, gh, gw):
    dc = m.get_config()
    sat_tf, grd_tf = get_transforms_val((sz, sz), (gh, gw), mean=dc["mean"], std=dc["std"])
    tf = grd_tf if item == "query" else sat_tf
    ds = CVUSADatasetEval(os.path.join(A.data, tree), "val", item, tf)
    return sort_by_gid(*predict(Cfg(), m, DataLoader(ds, batch_size=A.batch_size, num_workers=4)))


def load_names():
    """gid -> coordinate-named image file, from the mapping written by build_dataset.py."""
    with open(A.mapping) as f:
        return {int(r["gid"]): r["image_name"] for r in csv.DictReader(f)}


def write_csv(path, sim, gids, names, k=10):
    order = np.argsort(-sim, axis=1)[:, :k]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_gid", "query_image", "rank", "retrieved_gid", "retrieved_image",
                    "score", "is_match"])
        for i in range(sim.shape[0]):
            for r, j in enumerate(order[i]):
                w.writerow([int(gids[i]), names[int(gids[i])], r + 1, int(gids[j]),
                            names[int(gids[j])], round(float(sim[i, j]), 6),
                            int(gids[j] == gids[i])])


def main():
    os.makedirs(A.out_dir, exist_ok=True)
    names = load_names()
    ck_vr, ck_sv = best_weights(A.vgi_rsi), best_weights(A.svi_vgi)
    print(f"checkpoints:\n  VGI-RSI {ck_vr}\n  SVI-VGI {ck_sv}", flush=True)

    mVR = make_model(ck_vr, 384, False)
    vgi_vr, g = feats(mVR, "disaster_vgi", "query", 384, 384, 384)
    rsi, gr = feats(mVR, "disaster_vgi", "reference", 384, 384, 384)
    del mVR; torch.cuda.empty_cache()

    mSV = make_model(ck_sv, 336, True)
    vgi_sv, gs = feats(mSV, "disaster_svi2vgi", "reference", 336, 336, 672)
    svi, gsv = feats(mSV, "disaster_svi2vgi", "query", 336, 336, 672)
    del mSV; torch.cuda.empty_cache()

    assert (g == gr).all() and (g == gs).all() and (g == gsv).all(), "gid misalignment"
    print(f"{len(g)} test groups, gid {g.min()}..{g.max()}", flush=True)

    W, T, IT = A.fusion_w, A.sinkhorn_t, A.sinkhorn_iters
    sim_rsi, sim_svi = vgi_vr @ rsi.T, vgi_sv @ svi.T
    fusion = W * zscore(sim_svi) + (1 - W) * zscore(sim_rsi)
    sink = sinkhorn(fusion, T, IT)

    systems = [(f"{A.tag}_rsi_only", f"{A.label} (VGI-RSI)", sim_rsi),
               (f"{A.tag}_svi_only", f"{A.label} (VGI-SVI)", sim_svi),
               (f"{A.tag}_fusion", f"{A.label} (VGI-SVI-RSI), w={W}", fusion),
               (f"{A.tag}_fusion_sinkhorn",
                f"{A.label} (VGI-SVI-RSI + Sinkhorn), w={W} t={T}", sink)]

    rows = []
    for key, label, S in systems:
        write_csv(os.path.join(A.out_dir, f"{key}_top10.csv"), S, g, names)
        r = recall_at_k(-S, g, g)
        rows.append((label, r))
        print(f"{label:46} R@1={r['R@1']:6} R@5={r['R@5']:6} R@10={r['R@10']:6} "
              f"R@1%={r['R@1%']:6}", flush=True)

    with open(os.path.join(A.out_dir, f"recall_{A.tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "queries", "gallery", "R@1", "R@5", "R@10", "R@1pct"])
        for label, r in rows:
            w.writerow([label, len(g), len(g), r["R@1"], r["R@5"], r["R@10"], r["R@1%"]])
    print(f"\nwrote {len(systems)} top-10 CSVs + recall_{A.tag}.csv to {A.out_dir}/")


if __name__ == "__main__":
    main()
