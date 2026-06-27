# Scan DexYCB scenes for frames suitable for the Simplest Case: an object held
# statically and lifted off the table with several finger contacts.
#
# For every s0_train sequence whose grasped object matches --ycb_id (default
# 003_cracker_box), frames are sampled with a stride on one camera. Cheap
# pose-only prefilters (hand present, object lifted off the table, and -- only
# for box-like objects when --vertical_min > 0 -- the object's longest axis
# aligned with gravity) run first; contact is then computed by transforming
# MANO vertices into the object canonical frame, so the object ProximityQuery
# is built only once.
#
# The "vertical" filter is box-specific (it assumes a meaningful long axis to
# stand upright). For smooth objects (can/bowl) pass --vertical_min 0 to skip
# it; vert_dot is still reported for reference.

import os

os.environ.setdefault("DEX_YCB_DIR", "/datasets/dexycb")

import argparse
import time

import numpy as np
import torch
import trimesh
import trimesh.proximity
import yaml

from manopth.manolayer import ManoLayer

from dex_ycb_toolkit.factory import get_dataset

from compute_contact import (_FINGER_NAMES, _PART_TO_FINGER, find_mano_root,
                             vertex_areas)

_SERIAL = "841412060263"  # camera used for reporting idx; geometry is global
_CRACKER_YCB_ID = 2

# MANO parts 3/6/9/12/15 are the distal phalanges (fingertips).
_TIP_PARTS = np.array([3, 6, 9, 12, 15])


def load_extrinsics(data_dir, extr_name):
  extr_file = os.path.join(data_dir, "calibration", "extrinsics_" + extr_name,
                           "extrinsics.yml")
  with open(extr_file) as f:
    extr = yaml.unsafe_load(f)["extrinsics"]
  T_cam = np.array(extr[_SERIAL], dtype=np.float64).reshape(3, 4)
  T_tag = np.array(extr["apriltag"], dtype=np.float64).reshape(3, 4)
  return T_cam, T_tag


def tag_frame_point(T_cam, T_tag, p_cam):
  """Camera-frame point -> apriltag (table) frame, z = height above table."""
  p_world = T_cam[:, :3] @ p_cam + T_cam[:, 3]
  return T_tag[:, :3].T @ (p_world - T_tag[:, 3])


class HandModel:
  """Caches one ManoLayer per side."""

  def __init__(self):
    self._layers = {}
    self._mano_root = find_mano_root()

  def layer(self, side):
    if side not in self._layers:
      self._layers[side] = ManoLayer(flat_hand_mean=False,
                                     ncomps=45,
                                     side=side,
                                     mano_root=self._mano_root,
                                     use_pca=True)
    return self._layers[side]

  def verts(self, side, betas, pose_m):
    layer = self.layer(side)
    betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)
    pose = torch.from_numpy(pose_m)
    with torch.no_grad():
      vert, _ = layer(pose[:, 0:48], betas, pose[:, 48:51])
    return (vert / 1000.0).view(778, 3).numpy().astype(np.float64)

  def part_labels(self, side):
    return self.layer(side).th_weights.numpy().argmax(axis=1)


def cluster_metrics(hand_verts_cam, faces, part, sd, tri_id, obj_normals_cam,
                    g_cam, thresh, min_verts):
  """Per-finger contact clusters; returns list of dicts (camera frame)."""
  contact = sd > -thresh
  if contact.sum() == 0:
    return [], contact
  hand_mesh = trimesh.Trimesh(vertices=hand_verts_cam, faces=faces,
                              process=False)
  va = vertex_areas(hand_mesh)
  vnorm = np.asarray(hand_mesh.vertex_normals)
  finger = _PART_TO_FINGER[part]
  clusters = []
  for fi, fname in enumerate(_FINGER_NAMES):
    sel = np.where(contact & (finger == fi))[0]
    if len(sel) < min_verts:
      continue
    n_obj = -obj_normals_cam[sel].mean(axis=0)
    n_obj /= np.linalg.norm(n_obj)
    n_hand = vnorm[sel].mean(axis=0)
    n_hand /= np.linalg.norm(n_hand)
    clusters.append({
        "finger": fname,
        "n_verts": len(sel),
        "area_m2": va[sel].sum(),
        "dot_g_obj": float(n_obj @ g_cam),
        "dot_g_hand": float(n_hand @ g_cam),
        "tip_frac": float(np.isin(part[sel], _TIP_PARTS).mean()),
    })
  return clusters, contact


def main():
  parser = argparse.ArgumentParser(
      description="Scan grasp scenes for static lifted holds")
  parser.add_argument("--name", default="s0_train")
  parser.add_argument("--ycb_id", type=int, default=_CRACKER_YCB_ID,
                      help="grasped YCB object id (2=cracker, 1=can, 13=bowl)")
  parser.add_argument("--stride", type=int, default=4)
  parser.add_argument("--thresh", type=float, default=0.005)
  parser.add_argument("--min_verts", type=int, default=3)
  parser.add_argument("--vertical_min", type=float, default=0.95,
                      help="min |object long axis . -g| to count as vertical; "
                           "0 disables (use for smooth objects)")
  parser.add_argument("--lift_min", type=float, default=0.03,
                      help="min box-center rise above its frame-0 height [m]")
  parser.add_argument("--out", default=None,
                      help="defaults to vis/contact/scan_results[_<obj>].npz")
  args = parser.parse_args()

  dataset = get_dataset(args.name)
  cam_idx = dataset._serials.index(_SERIAL)
  mapping = dataset._mapping

  ycb_id = args.ycb_id
  obj_name = dataset.ycb_classes[ycb_id]
  if args.out is None:
    tag = "_%s" % obj_name if ycb_id != _CRACKER_YCB_ID else ""
    args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis",
                            "contact", "scan_results%s.npz" % tag)
  obj_mesh = trimesh.load(dataset.obj_file[ycb_id], process=False,
                          force="mesh")
  pq = trimesh.proximity.ProximityQuery(obj_mesh)
  up_axis = int(np.argmax(obj_mesh.extents))
  print("%s extents %s -> long axis %d" %
        (obj_name, np.round(obj_mesh.extents, 3), up_axis))

  hands = HandModel()

  # Sequences grasping the target object.
  seq_ids = [
      s for s in range(len(dataset._sequences))
      if dataset._ycb_ids[s][dataset._ycb_grasp_ind[s]] == ycb_id
  ]
  print("%d %s sequences in %s" % (len(seq_ids), obj_name, args.name))

  rows = []
  t0 = time.time()
  for s in seq_ids:
    seq_name = dataset._sequences[s]
    seq_dir = os.path.join(dataset.data_dir, seq_name)
    with open(os.path.join(seq_dir, "meta.yml")) as f:
      meta = yaml.safe_load(f)
    T_cam, T_tag = load_extrinsics(dataset.data_dir, meta["extrinsics"])
    up_world = T_tag[:, 2]
    g_cam = T_cam[:, :3].T @ (-up_world)
    g_cam /= np.linalg.norm(g_cam)
    grasp_ind = dataset._ycb_grasp_ind[s]
    side = dataset._mano_side[s]
    betas = dataset._mano_betas[s]
    part = hands.part_labels(side)
    faces = hands.layer(side).th_faces.numpy()

    sel = np.where((mapping[:, 0] == s) & (mapping[:, 1] == cam_idx))[0]
    sel = sel[np.argsort(mapping[sel, 2])]
    z0 = None
    n_checked = 0
    for idx in sel:
      f = int(mapping[idx, 2])
      sample = dataset[idx]
      label = np.load(sample["label_file"])
      pose_y = label["pose_y"][grasp_ind]
      if np.all(pose_y == 0.0):
        continue
      R_y, t_y = pose_y[:, :3].astype(np.float64), pose_y[:, 3].astype(
          np.float64)
      center_cam = R_y @ obj_mesh.vertices.mean(axis=0) + t_y
      z_tag = tag_frame_point(T_cam, T_tag, center_cam)[2]
      if z0 is None:
        z0 = z_tag  # frame-0 (resting) height reference
      if f % args.stride != 0:
        continue
      vert_dot = abs(R_y[:, up_axis] @ (-g_cam))
      lift = z_tag - z0
      if np.all(label["pose_m"] == 0.0):
        continue
      if lift < args.lift_min:
        continue
      if args.vertical_min > 0 and vert_dot < args.vertical_min:
        continue

      # Contact in the object canonical frame.
      hand_cam = hands.verts(side, betas, label["pose_m"])
      hand_obj = (hand_cam - t_y) @ R_y
      closest, dist, tri_id = pq.on_surface(hand_obj)
      behind = np.einsum("ij,ij->i", hand_obj - closest,
                         obj_mesh.face_normals[tri_id]) < 0
      sd = np.where(behind, dist, -dist)
      obj_normals_cam = obj_mesh.face_normals[tri_id] @ R_y.T
      clusters, contact = cluster_metrics(hand_cam, faces, part, sd, tri_id,
                                          obj_normals_cam, g_cam, args.thresh,
                                          args.min_verts)
      n_checked += 1
      if len(clusters) < 4 or not any(c["finger"] == "thumb"
                                      for c in clusters):
        continue
      dots = np.array([c["dot_g_obj"] for c in clusters])
      rows.append({
          "idx": int(idx),
          "seq": seq_name,
          "frame": f,
          "side": side,
          "n_clusters": len(clusters),
          "fingers": ",".join(c["finger"] for c in clusters),
          "worst_dot": float(np.abs(dots).max()),
          "mean_dot": float(np.abs(dots).mean()),
          "vertical": float(vert_dot),
          "lift": float(lift),
          "tip_frac": float(np.mean([c["tip_frac"] for c in clusters])),
          "n_contact": int(contact.sum()),
      })
    print("  %s (%s): %d frames passed prefilter, %d rows total  [%.0fs]" %
          (seq_name, side, n_checked, len(rows), time.time() - t0))

  if not rows:
    print("no candidate frames found")
    return
  rows.sort(key=lambda r: (r["worst_dot"], -r["lift"]))
  print("\nTop candidates (sorted by worst |n_k.g| over clusters):")
  print("%-42s %5s %6s %5s %9s %9s %6s %6s %8s %s" %
        ("sequence", "frame", "idx", "side", "worst|ng|", "mean|ng|",
         "vert", "lift", "tipfrac", "fingers"))
  for r in rows[:25]:
    print("%-42s %5d %6d %5s %9.3f %9.3f %6.3f %6.3f %8.2f %s" %
          (r["seq"], r["frame"], r["idx"], r["side"], r["worst_dot"],
           r["mean_dot"], r["vertical"], r["lift"], r["tip_frac"],
           r["fingers"]))

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  np.savez(args.out, rows=np.array(rows, dtype=object))
  print("\nsaved %d candidate rows to %s" % (len(rows), args.out))


if __name__ == "__main__":
  main()
