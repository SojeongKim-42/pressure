# Sweep the within-finger normal-patch split threshold across several objects.
#
# For the cracker box the split angle (30 deg) was read off a bimodal valley in
# the within-finger normal-spread distribution. Smooth objects (can/bowl) have
# no such valley -- the surface normal varies continuously -- so the threshold
# is no longer data-given but a modeling tolerance: how much normal spread we
# tolerate inside one lumped contact patch before splitting it.
#
# This script sweeps the threshold T and, per object, reports the tradeoff:
#   - patches per finger-contact        (fragmentation; rises as T drops)
#   - patch spatial diameter [cm]       (physical patch size; a finger pad is
#                                        ~1.5-2 cm -- below that we are slicing a
#                                        single pad, which has no physical sense)
#   - within-patch P90 normal spread    (how well one normal represents a patch)
#   - within-patch resultant length R   (=|mean unit normal|; 1=coherent)
#   - p10 patch area [cm^2]             (small-area onset = pressure artifact)
#
# A fixed ANGLE maps to different physical patch SIZES on different curvatures,
# so comparing objects shows whether 20 deg generalizes or whether a physical
# patch-size criterion is the object-invariant choice.

import os

os.environ.setdefault("DEX_YCB_DIR", "/datasets/dexycb")

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import trimesh.proximity

from dex_ycb_toolkit.factory import get_dataset

from compute_contact import (_FINGER_NAMES, _PART_TO_FINGER, contact_directions,
                             split_by_normal, vertex_areas)
from scan_contact_scenes import HandModel, _SERIAL

_FINGER_IDS = [fi for fi, name in enumerate(_FINGER_NAMES) if name != "palm"]
_ANGLES = [10, 15, 20, 25, 30, 40, 50, 60, 90]


def patch_metrics(dirs, pos, areas, T, min_verts):
    """Split one finger's contact (dirs/pos/areas) at angle T; per-patch stats.

    Returns list of dicts for patches with >= min_verts vertices.
    """
    labels = split_by_normal(dirs, T)
    out = []
    for p in range(labels.max() + 1):
        grp = np.where(labels == p)[0]
        if len(grp) < min_verts:
            continue
        d = dirs[grp]
        mean = d.mean(axis=0)
        R = np.linalg.norm(mean)                 # resultant length in [0,1]
        mean_u = mean / R if R > 1e-9 else d[0]
        ang = np.degrees(np.arccos(np.clip(d @ mean_u, -1.0, 1.0)))
        pp = pos[grp]
        # patch spatial diameter = max pairwise distance (patches are small).
        diam = 0.0
        if len(pp) > 1:
            diff = pp[:, None, :] - pp[None, :, :]
            diam = float(np.sqrt((diff ** 2).sum(-1)).max())
        out.append({
            "n": len(grp),
            "area_cm2": float(areas[grp].sum()) * 1e4,
            "diam_cm": diam * 100.0,
            "p90_spread": float(np.percentile(ang, 90)),
            "R": float(R),
        })
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Sweep split threshold across objects")
    parser.add_argument("--name", default="s0_train")
    parser.add_argument("--ycb_ids", type=int, nargs="+",
                        default=[2, 3, 8, 1, 4, 9, 13, 14, 5],
                        help="objects to sweep (default: 3 box, 3 can, bowl/mug/bottle)")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--thresh", type=float, default=0.005)
    parser.add_argument("--min_verts", type=int, default=3)
    parser.add_argument("--max_seqs", type=int, default=0,
                        help="limit sequences per object (0 = all)")
    parser.add_argument("--out_dir",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "vis", "contact"))
    args = parser.parse_args()

    dataset = get_dataset(args.name)
    cam_idx = dataset._serials.index(_SERIAL)
    mapping = dataset._mapping
    hands = HandModel()

    # results[ycb_id][T] = list of per-patch dicts; plus per (obj) patches/finger.
    summary = {}     # ycb_id -> {T -> aggregated dict}
    obj_names = {}

    for ycb_id in args.ycb_ids:
        obj_name = dataset.ycb_classes[ycb_id]
        obj_names[ycb_id] = obj_name
        obj_mesh = trimesh.load(dataset.obj_file[ycb_id], process=False,
                                force="mesh")
        pq = trimesh.proximity.ProximityQuery(obj_mesh)
        seq_ids = [s for s in range(len(dataset._sequences))
                   if dataset._ycb_ids[s][dataset._ycb_grasp_ind[s]] == ycb_id]
        if args.max_seqs:
            seq_ids = seq_ids[:args.max_seqs]

        per_T_patches = {T: [] for T in _ANGLES}
        per_T_npatch = {T: [] for T in _ANGLES}   # patches per finger-contact
        n_frames = 0
        n_fc = 0                                   # finger-contacts processed
        t0 = time.time()
        for s in seq_ids:
            side = dataset._mano_side[s]
            betas = dataset._mano_betas[s]
            grasp_ind = dataset._ycb_grasp_ind[s]
            part = hands.part_labels(side)
            finger = _PART_TO_FINGER[part]
            faces = hands.layer(side).th_faces.numpy()

            sel_seq = np.where((mapping[:, 0] == s) &
                               (mapping[:, 1] == cam_idx))[0]
            sel_seq = sel_seq[np.argsort(mapping[sel_seq, 2])]
            for idx in sel_seq:
                f = int(mapping[idx, 2])
                if f % args.stride != 0:
                    continue
                sample = dataset[idx]
                label = np.load(sample["label_file"])
                if np.all(label["pose_m"] == 0.0):
                    continue
                pose_y = label["pose_y"][grasp_ind]
                if np.all(pose_y == 0.0):
                    continue
                R_y = pose_y[:, :3].astype(np.float64)
                t_y = pose_y[:, 3].astype(np.float64)

                hand_cam = hands.verts(side, betas, label["pose_m"])
                hand_mesh = trimesh.Trimesh(vertices=hand_cam, faces=faces,
                                            process=False)
                va = vertex_areas(hand_mesh)
                hand_obj = (hand_cam - t_y) @ R_y
                closest, dist, tri_id = pq.on_surface(hand_obj)
                behind = np.einsum("ij,ij->i", hand_obj - closest,
                                   obj_mesh.face_normals[tri_id]) < 0
                sd = np.where(behind, dist, -dist)
                contact = sd > -args.thresh
                if contact.sum() == 0:
                    continue
                n_frames += 1
                d_all = contact_directions(hand_obj, obj_mesh, closest, sd,
                                           tri_id)
                for fi in _FINGER_IDS:
                    vsel = np.where(contact & (finger == fi))[0]
                    if len(vsel) < args.min_verts:
                        continue
                    n_fc += 1
                    dirs = d_all[vsel]
                    pos = hand_cam[vsel]
                    areas = va[vsel]
                    for T in _ANGLES:
                        pm = patch_metrics(dirs, pos, areas, T, args.min_verts)
                        per_T_patches[T].extend(pm)
                        per_T_npatch[T].append(len(pm))
            print("  [%s] seq %d/%d, %d frames, %d finger-contacts  [%.0fs]" %
                  (obj_name, seq_ids.index(s) + 1, len(seq_ids), n_frames,
                   n_fc, time.time() - t0))

        summary[ycb_id] = {}
        for T in _ANGLES:
            P = per_T_patches[T]
            if not P:
                continue
            area = np.array([d["area_cm2"] for d in P])
            diam = np.array([d["diam_cm"] for d in P])
            spread = np.array([d["p90_spread"] for d in P])
            Rv = np.array([d["R"] for d in P])
            summary[ycb_id][T] = {
                "patches_per_finger": float(np.mean(per_T_npatch[T])),
                "n_patches": len(P),
                "med_area": float(np.median(area)),
                "p10_area": float(np.percentile(area, 10)),
                "med_diam": float(np.median(diam)),
                "med_spread": float(np.median(spread)),
                "mean_R": float(np.mean(Rv)),
            }
        print("[%s] done: %d frames, %d finger-contacts" %
              (obj_name, n_frames, n_fc))

    # ----- text tables -----
    for ycb_id in args.ycb_ids:
        if ycb_id not in summary or not summary[ycb_id]:
            continue
        print("\n=== %s ===" % obj_names[ycb_id])
        print("%5s %10s %10s %10s %11s %8s" %
              ("T", "patch/fing", "med_diam", "med_area", "med_spread", "mean_R"))
        print("%5s %10s %10s %10s %11s %8s" %
              ("[deg]", "", "[cm]", "[cm2]", "[deg]", ""))
        for T in _ANGLES:
            if T not in summary[ycb_id]:
                continue
            r = summary[ycb_id][T]
            print("%5d %10.2f %10.2f %10.2f %11.1f %8.3f" %
                  (T, r["patches_per_finger"], r["med_diam"], r["med_area"],
                   r["med_spread"], r["mean_R"]))

    # ----- plot: one line per object, 4 metric panels vs T -----
    os.makedirs(args.out_dir, exist_ok=True)
    cmap = plt.cm.tab10
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    metrics = [("patches_per_finger", "patches per finger-contact", None),
               ("med_diam", "median patch diameter [cm]", 2.0),
               ("med_spread", "median within-patch P90 spread [deg]", None),
               ("mean_R", "mean within-patch resultant R", 0.95)]
    for mi, (key, ylabel, hline) in enumerate(metrics):
        ax = axes[mi]
        for ci, ycb_id in enumerate(args.ycb_ids):
            if ycb_id not in summary or not summary[ycb_id]:
                continue
            Ts = [T for T in _ANGLES if T in summary[ycb_id]]
            ys = [summary[ycb_id][T][key] for T in Ts]
            ax.plot(Ts, ys, "o-", color=cmap(ci % 10),
                    label=obj_names[ycb_id])
        ax.axvline(20, color="k", ls=":", lw=1)
        if hline is not None:
            ax.axhline(hline, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("split threshold T [deg]")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Split-threshold sweep across objects "
                 "(dotted=20 deg; pad~2cm / R=0.95 guides)")
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "split_sweep.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("\nsaved %s" % out_png)

    out_npz = os.path.join(args.out_dir, "split_sweep.npz")
    np.savez(out_npz, summary=np.array(summary, dtype=object),
             obj_names=np.array(obj_names, dtype=object),
             angles=np.array(_ANGLES))
    print("saved %s" % out_npz)


if __name__ == "__main__":
    main()
