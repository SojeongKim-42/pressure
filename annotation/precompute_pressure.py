# Precompute & store physics-based pressure pseudo-labels for grasp scenes,
# WITHOUT rendering (Research Plan: Simplest Case -- the actual annotation step).
#
# For one object (--ycb_id) or every object in ycb_object_params.json (--ycb_id 0),
# this iterates the s0_train sequences that grasp it and, for each frame (--stride,
# default 1 = every frame) where hand+object are present, runs the pipeline once:
#   detect_contact -> cluster_stats (finger x normal-patch) -> per-contact
#   friction tangent -> min-effort SOCP (2D friction + torque about COM) ->
#   pressure_k = fn_k / area_k   (solve_pressure.solve_frame).
# Per-object mass and friction (mu) are read from ycb_object_params.json.
#
# EVERY hand+object-present frame gets a row, including ones that did not solve
# (no_contact / few_usable / infeasible) -- those are stored with 0 contacts and
# their status string, so the precompute step does NOT decide "is this a grasp";
# the renderer (render_pressure_video --from_data) decides what to draw from the
# stored status. Pass --solved_only to keep just the feasible-grasp frames.
#
# Only the VALUES are saved (no images): one compressed npz per object in
# pressure_labels/<model>.npz, as two flat tables -- a per-frame table and a
# per-contact table the frame table indexes into (see save_pressure_npz). This
# is the lookup store the renderer reads via render_pressure_video --from_data,
# so it can draw straight from the values instead of re-solving every frame.
#
# Usage:
#   python precompute_pressure.py --ycb_id 1        # one object (can)
#   python precompute_pressure.py --ycb_id 0        # all objects in the json
#   python precompute_pressure.py --ycb_id 13 13 14 # several objects

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("DEX_YCB_DIR", "/datasets/dexycb")

import argparse
import json
import time

import numpy as np
import trimesh

from dex_ycb_toolkit.factory import get_dataset

from compute_contact import gravity_in_camera, load_hand
from solve_pressure import _GRAVITY, object_com, solve_frame
from scan_contact_scenes import _SERIAL

_PARAMS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ycb_object_params.json")
_DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "pressure_labels")


# ycb_object_params.json -> {ycb_id: {model, name, mass_kg, mu, material}}.
def load_object_params(path=_PARAMS_JSON):
    with open(path) as f:
        data = json.load(f)
    return {o["ycb_id"]: o for o in data["objects"]}


# 저장된 pseudo-label npz를 읽어, idx로 한 프레임의 contact 값을 바로 꺼낼 수 있는
# 형태로 돌려준다. 반환: (meta dict, frames structured array, contacts dict,
# idx->frame-row 매핑). render_pressure_video --from_data 가 이걸 쓴다.
def load_pressure_npz(path):
    d = np.load(path, allow_pickle=True)
    meta = {k: d[k].item() for k in d.files if k.startswith("meta_")}
    frames = {k[2:]: d[k] for k in d.files if k.startswith("f_")}
    contacts = {k[2:]: d[k] for k in d.files if k.startswith("c_")}
    idx_to_row = {int(ix): i for i, ix in enumerate(frames["idx"])}
    return meta, frames, contacts, idx_to_row


# frame-row 하나에 해당하는 contact들을 dict(키=contact 필드, 값=배열)로 잘라 준다.
def frame_contacts(frames, contacts, row):
    start = int(frames["contact_start"][row])
    n = int(frames["n_contacts"][row])
    return {k: v[start:start + n] for k, v in contacts.items()}


def compute_object(dataset, ycb_id, params, args):
    """Solve every usable frame grasping `ycb_id`; return flat frame+contact tables."""
    obj_name = dataset.ycb_classes[ycb_id]
    mass = params["mass_kg"]
    mu = params["mu"]
    cam_idx = dataset._serials.index(_SERIAL)
    mapping = dataset._mapping
    obj_canon = trimesh.load(dataset.obj_file[ycb_id], process=False, force="mesh")

    seq_ids = [
        s
        for s in range(len(dataset._sequences))
        if dataset._ycb_ids[s][dataset._ycb_grasp_ind[s]] == ycb_id
    ]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]
    print(
        "\n=== ycb_id %d  %s  (m=%.3f kg, mu=%.2f, %s)  %d sequences ==="
        % (ycb_id, obj_name, mass, mu, params.get("material", "?"), len(seq_ids))
    )

    # per-frame columns
    F = {k: [] for k in ("idx", "seq", "frame", "side", "status", "n_support",
                         "fn_sum", "weight", "contact_start", "n_contacts")}
    # per-contact columns (flat across all frames of this object)
    C = {k: [] for k in ("label", "finger", "centroid", "area_m2", "normal",
                         "t1", "t2", "fn", "ft1", "ft2", "pressure_pa", "verts")}
    skip = {"no_contact": 0, "few_usable": 0, "infeasible": 0}
    n_solved = 0  # 접촉이 풀려 pressure 값이 있는 프레임 수
    n_stored = 0  # 저장된 frame 행 수 (solved + unsolved; --solved_only면 solved만)
    t0 = time.time()

    for s in seq_ids:
        seq_name = dataset._sequences[s]
        grasp_ind = dataset._ycb_grasp_ind[s]
        sel = np.where((mapping[:, 0] == s) & (mapping[:, 1] == cam_idx))[0]
        sel = sel[np.argsort(mapping[sel, 2])]
        n_seq = 0
        for idx in sel:
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

            hand_mesh, finger = load_hand(sample, label)
            obj_mesh = obj_canon.copy()
            obj_mesh.apply_transform(np.vstack((pose_y, [0, 0, 0, 1])).astype(np.float64))
            g_cam = gravity_in_camera(sample, dataset.data_dir)

            res, status = solve_frame(
                hand_mesh, obj_mesh, finger, g_cam, args.thresh, args.min_verts,
                mass, mu, args.friction, args.torque,
            )
            # 손·물체가 있는 프레임은 (풀리든 안 풀리든) frame 행을 항상 남긴다.
            # 풀이 실패(no_contact/few_usable/infeasible)는 contact 0개로 status만 기록 →
            # 그릴지 말지는 render(--from_data)가 판단. --solved_only면 풀린 것만 저장.
            if res is None:
                skip[status] = skip.get(status, 0) + 1
                if args.solved_only:
                    continue
                F["contact_start"].append(len(C["fn"]))
                F["n_contacts"].append(0)
                F["idx"].append(int(idx))
                F["seq"].append(int(s))
                F["frame"].append(f)
                F["side"].append(sample["mano_side"])
                F["status"].append(str(status))
                F["n_support"].append(0)
                F["fn_sum"].append(0.0)
                F["weight"].append(float(mass * _GRAVITY))
                n_stored += 1
                n_seq += 1
                continue
            kept, fn, ft1, ft2, normals, t1s, t2s, pressures, n_support = res

            F["contact_start"].append(len(C["fn"]))
            F["n_contacts"].append(len(kept))
            F["idx"].append(int(idx))
            F["seq"].append(int(s))
            F["frame"].append(f)
            F["side"].append(sample["mano_side"])
            F["status"].append(str(status))
            F["n_support"].append(int(n_support))
            F["fn_sum"].append(float(fn.sum()))
            F["weight"].append(float(mass * _GRAVITY))
            for ci, c in enumerate(kept):
                C["label"].append(c["label"])
                C["finger"].append(c["finger"])
                C["centroid"].append(c["centroid"])
                C["area_m2"].append(c["area_m2"])
                C["normal"].append(normals[ci])
                C["t1"].append(t1s[ci])
                C["t2"].append(t2s[ci])
                C["fn"].append(float(fn[ci]))
                C["ft1"].append(float(ft1[ci]))
                C["ft2"].append(float(ft2[ci]))
                C["pressure_pa"].append(float(pressures[ci]))
                C["verts"].append(np.asarray(c["verts"], dtype=np.int32))
            n_solved += 1
            n_stored += 1
            n_seq += 1
        print(
            "  seq %d/%d %s: %d frames  [%d stored / %d solved, skip %s, %.0fs]"
            % (seq_ids.index(s) + 1, len(seq_ids), seq_name, n_seq, n_stored,
               n_solved, skip, time.time() - t0)
        )
    return F, C, skip, n_solved, n_stored


def save_pressure_npz(out_path, ycb_id, obj_name, params, args, F, C, skip,
                      n_solved, n_stored):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arrs = {}
    # per-frame table (prefix f_), per-contact table (prefix c_).
    arrs["f_idx"] = np.array(F["idx"], dtype=np.int64)
    arrs["f_seq"] = np.array(F["seq"], dtype=np.int32)
    arrs["f_frame"] = np.array(F["frame"], dtype=np.int32)
    arrs["f_side"] = np.array(F["side"])
    arrs["f_status"] = np.array(F["status"])
    arrs["f_n_support"] = np.array(F["n_support"], dtype=np.int32)
    arrs["f_fn_sum"] = np.array(F["fn_sum"], dtype=np.float64)
    arrs["f_weight"] = np.array(F["weight"], dtype=np.float64)
    arrs["f_contact_start"] = np.array(F["contact_start"], dtype=np.int64)
    arrs["f_n_contacts"] = np.array(F["n_contacts"], dtype=np.int32)
    arrs["c_label"] = np.array(C["label"])
    arrs["c_finger"] = np.array(C["finger"])
    arrs["c_centroid"] = np.array(C["centroid"], dtype=np.float64).reshape(-1, 3)
    arrs["c_area_m2"] = np.array(C["area_m2"], dtype=np.float64)
    arrs["c_normal"] = np.array(C["normal"], dtype=np.float64).reshape(-1, 3)
    arrs["c_t1"] = np.array(C["t1"], dtype=np.float64).reshape(-1, 3)
    arrs["c_t2"] = np.array(C["t2"], dtype=np.float64).reshape(-1, 3)
    arrs["c_fn"] = np.array(C["fn"], dtype=np.float64)
    arrs["c_ft1"] = np.array(C["ft1"], dtype=np.float64)
    arrs["c_ft2"] = np.array(C["ft2"], dtype=np.float64)
    arrs["c_pressure_pa"] = np.array(C["pressure_pa"], dtype=np.float64)
    arrs["c_verts"] = np.array(C["verts"] + [None], dtype=object)[:-1]  # ragged
    # metadata (prefix meta_, scalars stored 0-d so load_pressure_npz can .item()).
    arrs["meta_ycb_id"] = np.array(ycb_id)
    arrs["meta_model"] = np.array(params["model"])
    arrs["meta_obj_name"] = np.array(obj_name)
    arrs["meta_name"] = np.array(params.get("name", obj_name))
    arrs["meta_material"] = np.array(params.get("material", ""))
    arrs["meta_mass_kg"] = np.array(params["mass_kg"])
    arrs["meta_mu"] = np.array(params["mu"])
    arrs["meta_friction"] = np.array(args.friction)
    arrs["meta_torque"] = np.array(bool(args.torque))
    arrs["meta_thresh"] = np.array(args.thresh)
    arrs["meta_min_verts"] = np.array(args.min_verts)
    arrs["meta_stride"] = np.array(args.stride)
    arrs["meta_split"] = np.array(get_split_angle())
    arrs["meta_dataset"] = np.array(args.name)
    arrs["meta_solved_only"] = np.array(bool(args.solved_only))
    arrs["meta_n_solved"] = np.array(n_solved)   # pressure 값이 있는 프레임 수
    arrs["meta_n_stored"] = np.array(n_stored)   # 저장된 frame 행 수(solved+unsolved)
    arrs["meta_skip"] = np.array(skip)  # dict -> 0-d object
    np.savez_compressed(out_path, **arrs)
    return out_path


# compute_contact._SPLIT_ANGLE_DEG (실제 cluster_stats 기본값)을 기록용으로 읽어 온다.
def get_split_angle():
    from compute_contact import _SPLIT_ANGLE_DEG
    return _SPLIT_ANGLE_DEG


def main():
    parser = argparse.ArgumentParser(
        description="Precompute & store contact/pressure pseudo-labels (no render)"
    )
    parser.add_argument("--name", default="s0_train")
    parser.add_argument(
        "--ycb_id", type=int, nargs="+", default=[1],
        help="object id(s) to process; pass 0 (alone) to do every object in the json",
    )
    parser.add_argument("--params", default=_PARAMS_JSON,
                        help="ycb_object_params.json with per-object mass_kg / mu")
    parser.add_argument("--stride", type=int, default=1,
                        help="frame stride per sequence (1 = every frame; raise to subsample)")
    parser.add_argument("--thresh", type=float, default=0.005)
    parser.add_argument("--min_verts", type=int, default=3)
    parser.add_argument("--friction", choices=["1d", "2d"], default="2d")
    parser.add_argument(
        "--no_torque", dest="torque", action="store_false",
        help="disable torque equilibrium (force-only).",
    )
    parser.set_defaults(torque=True)
    parser.add_argument(
        "--solved_only", action="store_true",
        help="store only frames with a feasible grasp solve (default: also store "
        "hand+object-present frames that did not solve, tagged with their status, "
        "so the renderer decides what to draw).",
    )
    parser.add_argument("--max_seqs", type=int, default=0,
                        help="limit sequences per object for a quick run (0 = all)")
    parser.add_argument("--out_dir", default=_DEFAULT_OUT_DIR)
    args = parser.parse_args()

    params_by_id = load_object_params(args.params)
    dataset = get_dataset(args.name)

    if args.ycb_id == [0]:
        ycb_ids = sorted(params_by_id)
    else:
        ycb_ids = args.ycb_id
    print("processing %d object(s): %s" % (len(ycb_ids), ycb_ids))

    written = []
    for ycb_id in ycb_ids:
        if ycb_id not in params_by_id:
            print("  [skip] ycb_id %d not in %s" % (ycb_id, args.params))
            continue
        params = params_by_id[ycb_id]
        F, C, skip, n_solved, n_stored = compute_object(dataset, ycb_id, params, args)
        if n_stored == 0:
            print("  no usable frames for %s; nothing saved" % params["model"])
            continue
        out_path = os.path.join(args.out_dir, "%s.npz" % params["model"])
        save_pressure_npz(out_path, ycb_id, dataset.ycb_classes[ycb_id], params,
                          args, F, C, skip, n_solved, n_stored)
        written.append((out_path, n_solved, n_stored, len(C["fn"])))
        print("  wrote %d frames (%d solved) / %d contacts -> %s (skip %s)"
              % (n_stored, n_solved, len(C["fn"]), out_path, skip))

    print("\n=== done: %d object file(s) ===" % len(written))
    for p, ns, nst, nc in written:
        print("  %-55s  %5d frames (%5d solved)  %6d contacts"
              % (os.path.basename(p), nst, ns, nc))


if __name__ == "__main__":
    main()
