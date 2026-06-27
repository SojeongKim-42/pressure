# Render a video of physics-based pressure prediction across many grasp scenes
# of one object (Research Plan: Simplest Case, step 2 -- generalized beyond the
# clean vertical-hold cracker case to any --ycb_id).
#
# For every s0_train sequence grasping the target object, each frame (at a
# stride) where the hand and object are present is processed with the pipeline:
#   detect_contact -> cluster_stats (finger x normal-patch) -> per-contact
#   friction tangent -> min-effort SOCP (2D friction + torque equilibrium about
#   the object COM, --no_torque for force-only) -> pressure_k.
# The hand is colored by predicted pressure (inferno, FIXED vmax so frames are
# comparable) and the solved contact force F_k is drawn as a green arrow
# (length proportional to |F_k|). Left = camera-view overlay, right = gravity
# turntable. Frames are concatenated into one mp4.
#
# Unlike the clean validation we do NOT require a vertical pose -- tilted holds
# are included. SUPPORT contacts (n_k nearly parallel to gravity, t_k undefined)
# are excluded per the plan; frames left with <2 usable contacts, or whose SOCP
# is infeasible, are skipped (counted and reported).
#
# NOTE: --mass and --mu default to cracker; pass per-object values for others
# (e.g. master_chef_can: --ycb_id 1 --mass 0.414 --mu 1.11).

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("DEX_YCB_DIR", "/datasets/dexycb")

import argparse
import time

import cv2
import matplotlib.cm
import numpy as np
import pyrender
import trimesh

from dex_ycb_toolkit.factory import get_dataset

from compute_contact import (cluster_stats, detect_contact, gravity_in_camera,
                             load_hand)
from solve_pressure import (_DEFAULT_MASS, _DEFAULT_MU, _DEFAULT_VMAX_KPA,
                            _GRAVITY, force_arrows, friction_tangent,
                            generic_tangent, object_com, solve_min_effort)
from render_contact_video import FrameRenderer, compose_frame
from scan_contact_scenes import _CRACKER_YCB_ID, _SERIAL

_NON_CONTACT = np.array([190, 190, 190, 255], dtype=np.uint8)


# 손 mesh를 예측 pressure로 칠한다(inferno, vmax 고정 → frame 간 비교 가능).
def pressure_hand_mesh(hand_mesh, kept, pressures_pa, vmax_pa):
    pv = np.zeros(len(hand_mesh.vertices))
    for c, p in zip(kept, pressures_pa):
        pv[c["verts"]] = p
    cmap = matplotlib.cm.get_cmap("inferno")
    colors = np.tile(_NON_CONTACT, (len(hand_mesh.vertices), 1))
    hot = (cmap(np.clip(pv / vmax_pa, 0.0, 1.0)) * 255).astype(np.uint8)
    colors[pv > 0] = hot[pv > 0]
    m = trimesh.Trimesh(vertices=hand_mesh.vertices.copy(),
                        faces=hand_mesh.faces.copy(), vertex_colors=colors,
                        process=False)
    return pyrender.Mesh.from_trimesh(m)


# 한 frame의 contact→SOCP. 풀리면 (kept, fn,ft1,ft2, normals,t1,t2, pressures, n_support),
# 안 풀리면 (None, 사유). 사유: 'no_contact' / 'few_usable' / 'infeasible'.
def solve_frame(hand_mesh, obj_mesh, finger, g_cam, thresh, min_verts, mass, mu,
                friction, torque=True):
    sd, contact, tri_id, closest = detect_contact(hand_mesh, obj_mesh, thresh)
    if contact.sum() == 0:
        return None, "no_contact"
    clusters = cluster_stats(hand_mesh, obj_mesh, finger, contact, sd, tri_id,
                             closest, g_cam, min_verts)
    kept, normals, t1s, t2s = [], [], [], []
    n_support = 0
    for c in clusters:
        if friction == "1d":
            t = friction_tangent(c["n_obj"], g_cam)
            if t is None:  # n_k ∥ gravity → 1D singularity, support 제외
                n_support += 1
                continue
        else:  # 2D: generic basis, support contact 포함
            t = generic_tangent(c["n_obj"])
        kept.append(c)
        normals.append(c["n_obj"])
        t1s.append(t[0])
        t2s.append(t[1])
    if len(kept) < 2:
        return None, "few_usable"
    # Torque balanced about the object COM: r_k = cluster centroid - COM.
    com, _ = object_com(obj_mesh)
    arms = np.array([c["centroid"] for c in kept]) - com
    fn, ft1, ft2, status = solve_min_effort(normals, t1s, t2s, g_cam, mass, mu,
                                            friction, arms=arms, torque=torque)
    if fn is None:
        return None, "infeasible"
    areas = np.array([c["area_m2"] for c in kept])
    pressures = fn / areas
    return (kept, fn, ft1, ft2, normals, t1s, t2s, pressures, n_support), status


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scene physics pressure prediction video")
    parser.add_argument("--name", default="s0_train")
    parser.add_argument("--ycb_id", type=int, default=_CRACKER_YCB_ID,
                        help="grasped YCB object id (2=cracker, 1=can, 13=bowl)")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--thresh", type=float, default=0.005)
    parser.add_argument("--min_verts", type=int, default=3)
    parser.add_argument("--mass", type=float, default=_DEFAULT_MASS)
    parser.add_argument("--mu", type=float, default=_DEFAULT_MU)
    parser.add_argument("--friction", choices=["1d", "2d"], default="2d")
    parser.add_argument("--no_torque", dest="torque", action="store_false",
                        help="disable torque equilibrium (force-only).")
    parser.set_defaults(torque=True)
    parser.add_argument("--vmax_kpa", type=float, default=_DEFAULT_VMAX_KPA,
                        help="fixed pressure colormap max [kPa], shared with "
                             "solve_pressure (comparable across scenes/frames)")
    parser.add_argument("--max_seqs", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--out", default=None,
                        help="defaults to vis/pressure/pressure_clusters[_<obj>].mp4")
    args = parser.parse_args()
    vmax_pa = args.vmax_kpa * 1000.0

    dataset = get_dataset(args.name)
    ycb_id = args.ycb_id
    obj_name = dataset.ycb_classes[ycb_id]
    if args.out is None:
        tag = "_%s" % obj_name if ycb_id != _CRACKER_YCB_ID else ""
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "vis", "pressure",
                                "pressure_clusters%s.mp4" % tag)
    cam_idx = dataset._serials.index(_SERIAL)
    mapping = dataset._mapping
    obj_canon = trimesh.load(dataset.obj_file[ycb_id], process=False,
                             force="mesh")
    seq_ids = [s for s in range(len(dataset._sequences))
               if dataset._ycb_ids[s][dataset._ycb_grasp_ind[s]]
               == ycb_id]
    if args.max_seqs:
        seq_ids = seq_ids[:args.max_seqs]
    print("%d %s sequences (stride %d, %s friction, torque=%s, "
          "vmax %.0f kPa)" % (len(seq_ids), obj_name, args.stride,
                              args.friction, args.torque, args.vmax_kpa))

    renderer, writer = None, None
    n_written = 0
    skip = {"no_contact": 0, "few_usable": 0, "infeasible": 0}
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
            obj_mesh.apply_transform(
                np.vstack((pose_y, [0, 0, 0, 1])).astype(np.float64))
            g_cam = gravity_in_camera(sample, dataset.data_dir)

            res, status = solve_frame(hand_mesh, obj_mesh, finger, g_cam,
                                      args.thresh, args.min_verts, args.mass,
                                      args.mu, args.friction, args.torque)
            if res is None:
                skip[status] = skip.get(status, 0) + 1
                continue
            kept, fn, ft1, ft2, normals, t1s, t2s, pressures, n_support = res

            pr_mesh = pressure_hand_mesh(hand_mesh, kept, pressures, vmax_pa)
            arrows = force_arrows(kept, normals, t1s, t2s, fn, ft1, ft2)
            if renderer is None:
                renderer = FrameRenderer(sample["intrinsics"], dataset.w,
                                         dataset.h)
            cam_im = renderer.camera_overlay(pr_mesh, obj_mesh,
                                             sample["color_file"],
                                             extra_meshes=arrows)
            turn_im = renderer.turntable(pr_mesh, obj_mesh, g_cam,
                                         2 * np.pi * n_written / 90.0,
                                         extra_meshes=arrows)

            entries = ["%s:%.1f" % (c["label"], p / 1000.0)
                       for c, p in zip(kept, pressures)]
            lines = ["%s  frame %d  (idx %d, %s)" %
                     (seq_name, f, idx, sample["mano_side"]),
                     "%s | fn %.1fN / w %.1fN | %d contacts (%d support-excl)" %
                     (status, fn.sum(), args.mass * _GRAVITY, len(kept),
                      n_support),
                     "pressure[kPa] (vmax %.0f):" % args.vmax_kpa]
            lines += ["  ".join(entries[i:i + 5])
                      for i in range(0, len(entries), 5)]
            frame = compose_frame(cam_im, turn_im, lines, args.scale)
            if writer is None:
                h, w = frame.shape[:2]
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("VideoWriter failed to open %s" % args.out)
            writer.write(frame)
            n_written += 1
            n_seq += 1
        print("  %s: %d frames  [%d total, skip %s, %.0fs]" %
              (seq_name, n_seq, n_written, skip, time.time() - t0))

    if renderer is not None:
        renderer.close()
    if writer is not None:
        writer.release()
    print("\nwrote %d frames to %s" % (n_written, args.out))
    print("skipped frames: %s" % skip)


if __name__ == "__main__":
    main()
