# Diagnostic: within-finger object-normal angular spread distribution.
#
# Goal: decide the angle threshold for sub-clustering a finger into separate
# contact patches (compute_contact.py step 4). A finger that touches a single
# flat box face has near-zero spread among its contact vertices' object
# normals; a finger wrapped over an edge spans two ~perpendicular faces and
# shows a large spread. By pooling this spread over many cracker-box grasp
# frames we can read off where the "single-face noise floor" ends and genuine
# multi-patch contacts begin, and pick the split threshold in that valley.
#
# Object face normals are used (not MANO vertex normals): the MANO pad is
# rounded so hand normals are noisy, while the box face normals are what the
# downstream force model actually consumes. Angles are frame-invariant so we
# compute them in the object canonical frame (ProximityQuery built once).
#
# NOTE: unlike scan_contact_scenes.py this does NOT apply the vertical/lift
# prefilter -- we WANT both clean single-face grips and messy wrap-around
# grips in the sample so the distribution is representative.

import os

os.environ.setdefault("DEX_YCB_DIR", "/datasets/dexycb")

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import trimesh.proximity

from dex_ycb_toolkit.factory import get_dataset

from compute_contact import (_FINGER_NAMES, _PART_TO_FINGER,
                             contact_directions)
from scan_contact_scenes import HandModel, _CRACKER_YCB_ID, _SERIAL

# Fingers only (skip palm): palm contact is not a finger contact patch.
_FINGER_IDS = [fi for fi, name in enumerate(_FINGER_NAMES) if name != "palm"]


def main():
    parser = argparse.ArgumentParser(
        description="Within-finger object-normal angular spread distribution")
    parser.add_argument("--name", default="s0_train")
    parser.add_argument("--stride", type=int, default=4,
                        help="frame stride per sequence")
    parser.add_argument("--thresh", type=float, default=0.005,
                        help="proximity threshold in meters")
    parser.add_argument("--min_verts", type=int, default=3,
                        help="min contact vertices per finger to include")
    parser.add_argument("--max_seqs", type=int, default=0,
                        help="limit sequences for a quick run (0 = all)")
    parser.add_argument("--out_dir",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "vis", "contact"))
    args = parser.parse_args()

    dataset = get_dataset(args.name)
    cam_idx = dataset._serials.index(_SERIAL)
    mapping = dataset._mapping

    obj_mesh = trimesh.load(dataset.obj_file[_CRACKER_YCB_ID], process=False,
                            force="mesh")
    pq = trimesh.proximity.ProximityQuery(obj_mesh)
    hands = HandModel()

    seq_ids = [s for s in range(len(dataset._sequences))
               if dataset._ycb_ids[s][dataset._ycb_grasp_ind[s]]
               == _CRACKER_YCB_ID]
    if args.max_seqs:
        seq_ids = seq_ids[:args.max_seqs]
    print("%d cracker-box sequences in %s (stride %d, thresh %.1f mm)" %
          (len(seq_ids), args.name, args.stride, args.thresh * 1000))

    per_vertex_dev = []   # angle of each contact vertex's obj-normal from finger mean
    per_finger_max = []   # per (frame,finger): max deviation (sensitive to 1 stray vertex)
    per_finger_p90 = []   # per (frame,finger): 90th-pct deviation (robust multi-patch signal)
    records = []          # (idx, frame, side, finger, n_verts, max_dev, p90_dev)
    n_frames = 0
    t0 = time.time()

    for s in seq_ids:
        side = dataset._mano_side[s]
        betas = dataset._mano_betas[s]
        grasp_ind = dataset._ycb_grasp_ind[s]
        part = hands.part_labels(side)
        finger = _PART_TO_FINGER[part]

        sel_seq = np.where((mapping[:, 0] == s) & (mapping[:, 1] == cam_idx))[0]
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
            R_y, t_y = pose_y[:, :3].astype(np.float64), pose_y[:, 3].astype(np.float64)

            hand_cam = hands.verts(side, betas, label["pose_m"])
            hand_obj = (hand_cam - t_y) @ R_y
            closest, dist, tri_id = pq.on_surface(hand_obj)
            behind = np.einsum("ij,ij->i", hand_obj - closest,
                               obj_mesh.face_normals[tri_id]) < 0
            sd = np.where(behind, dist, -dist)
            contact = sd > -args.thresh
            if contact.sum() == 0:
                continue
            n_frames += 1
            # 군집화에 쓰는 것과 동일한 '손→물체 contact direction' 기준으로 spread 측정.
            d_all = contact_directions(hand_obj, obj_mesh, closest, sd, tri_id)

            for fi in _FINGER_IDS:
                vsel = np.where(contact & (finger == fi))[0]
                if len(vsel) < args.min_verts:
                    continue
                N = d_all[vsel]
                mean = N.mean(axis=0)
                mean /= np.linalg.norm(mean)
                ang = np.degrees(np.arccos(np.clip(N @ mean, -1.0, 1.0)))
                per_vertex_dev.extend(ang.tolist())
                per_finger_max.append(float(ang.max()))
                per_finger_p90.append(float(np.percentile(ang, 90)))
                records.append((int(idx), f, side, _FINGER_NAMES[fi],
                                len(vsel), float(ang.max()),
                                float(np.percentile(ang, 90))))
        print("  seq %d/%d done, %d frames, %d finger-clusters  [%.0fs]" %
              (seq_ids.index(s) + 1, len(seq_ids), n_frames, len(per_finger_max),
               time.time() - t0))

    per_vertex_dev = np.array(per_vertex_dev)
    per_finger_max = np.array(per_finger_max)
    per_finger_p90 = np.array(per_finger_p90)
    print("\n%d contact frames, %d finger-clusters, %d contact vertices" %
          (n_frames, len(per_finger_max), len(per_vertex_dev)))

    # ----- numbers -----
    def pct_table(name, arr):
        qs = [50, 75, 90, 95, 99]
        print("\n%s deviation from finger-mean object-normal [deg]:" % name)
        print("  mean %.1f   min %.1f   max %.1f" %
              (arr.mean(), arr.min(), arr.max()))
        print("  " + "  ".join("p%d %.1f" % (q, np.percentile(arr, q)) for q in qs))

    pct_table("per-vertex", per_vertex_dev)
    pct_table("per-finger MAX", per_finger_max)
    pct_table("per-finger P90 (robust)", per_finger_p90)

    print("\nfraction of finger-clusters exceeding T  (MAX vs P90 signal):")
    print("  %6s %12s %12s" % ("T[deg]", "by MAX", "by P90"))
    for T in (5, 10, 15, 20, 30, 45, 60, 90):
        print("  %6d %11.1f%% %11.1f%%" %
              (T, 100.0 * (per_finger_max > T).mean(),
               100.0 * (per_finger_p90 > T).mean()))

    # Multi-patch suspects (largest within-finger spread).
    records.sort(key=lambda r: -r[5])
    print("\nTop 15 widest-spread finger-clusters (candidate multi-patch):")
    print("  %8s %5s %5s %-7s %6s %8s %8s" %
          ("idx", "frame", "side", "finger", "nverts", "maxdev", "p90dev"))
    for r in records[:15]:
        print("  %8d %5d %5s %-7s %6d %8.1f %8.1f" % r)

    # ----- visualization -----
    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    bins = np.arange(0, 122, 2)

    def panel(ax, data, title, xlabel, color, qs, logy=False):
        ax.hist(data, bins=bins, color=color)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        for q in qs:
            x = np.percentile(data, q)
            ax.axvline(x, color="k", ls="--", lw=0.8)
            ax.text(x, ax.get_ylim()[1], " p%d=%.0f" % (q, x),
                    rotation=90, va="top", fontsize=8)

    panel(axes[0], per_vertex_dev, "Per-vertex object-normal deviation",
          "per-vertex deviation from finger-mean [deg]", "#1f78b4",
          (90, 95, 99), logy=True)
    axes[0].set_ylabel("contact vertices (log)")
    panel(axes[1], per_finger_max,
          "Per-finger MAX spread (sensitive to stray verts)",
          "per-finger MAX deviation [deg]", "#e31a1c", (50, 75, 90))
    axes[1].set_ylabel("finger-clusters")
    panel(axes[2], per_finger_p90,
          "Per-finger P90 spread (robust split signal)",
          "per-finger P90 deviation [deg]", "#33a02c", (50, 75, 90))
    axes[2].set_ylabel("finger-clusters")

    fig.suptitle("Within-finger object-normal spread  (%d clusters, %d frames, %s)"
                 % (len(per_finger_max), n_frames, args.name))
    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "normal_spread.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("\nsaved %s" % out_png)

    out_npz = os.path.join(args.out_dir, "normal_spread.npz")
    np.savez(out_npz,
             per_vertex_dev=per_vertex_dev,
             per_finger_max=per_finger_max,
             per_finger_p90=per_finger_p90,
             records=np.array(records, dtype=object),
             thresh=args.thresh, stride=args.stride)
    print("saved %s" % out_npz)


if __name__ == "__main__":
    main()
