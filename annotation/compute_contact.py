# Hand-object contact extraction via proximity threshold (Research Plan: Simplest Case, step 1).
#
# For one DexYCB sample, this script:
#   1. builds the MANO hand mesh and the grasped YCB object mesh in camera coordinates,
#   2. finds hand vertices within a proximity threshold of the object surface,
#   3. clusters contact vertices by finger (MANO skinning weights),
#   4. computes per-cluster representative normal n_k (hand-based and object-based)
#      and contact area_k, and checks n_k against the gravity direction,
#   5. renders the contact map on the hand mesh (camera-view overlay + orbit views),
#   6. saves per-cluster results to an .npz for the downstream SOCP step.

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("DEX_YCB_DIR", "/datasets/dexycb")

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import torch
import trimesh
import trimesh.proximity
import yaml

from manopth.manolayer import ManoLayer

from dex_ycb_toolkit.factory import get_dataset

_MANO_ROOT_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "dex-ycb-toolkit", "manopth", "mano", "models"),
    "/home/sjkim/Research/pressure/dex-ycb-toolkit/manopth/mano/models",
]
_DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "vis", "contact")

# MANO kinematic tree part order (th_weights columns):
# 0 = wrist/palm, 1-3 = index, 4-6 = middle, 7-9 = little, 10-12 = ring, 13-15 = thumb.
_PART_TO_FINGER = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
_FINGER_NAMES = ["palm", "index", "middle", "little", "ring", "thumb"]
_FINGER_COLORS = np.array([
    [177, 89, 40, 255],    # palm   - brown
    [51, 160, 44, 255],    # index  - green
    [31, 120, 180, 255],   # middle - blue
    [106, 61, 154, 255],   # little - purple
    [255, 127, 0, 255],    # ring   - orange
    [227, 26, 28, 255],    # thumb  - red
], dtype=np.uint8)
_NON_CONTACT_COLOR = np.array([190, 190, 190, 255], dtype=np.uint8)


def find_mano_root():
  for c in _MANO_ROOT_CANDIDATES:
    if os.path.isfile(os.path.join(c, "MANO_LEFT.pkl")):
      return c
  raise FileNotFoundError("MANO models not found in: %s" % _MANO_ROOT_CANDIDATES)


def load_hand(sample, label):
  """Returns hand mesh (camera frame, meters) and per-vertex finger labels."""
  pose_m = label["pose_m"]
  assert not np.all(pose_m == 0.0), "hand is not present in this frame"
  mano_layer = ManoLayer(flat_hand_mean=False,
                         ncomps=45,
                         side=sample["mano_side"],
                         mano_root=find_mano_root(),
                         use_pca=True)
  betas = torch.tensor(sample["mano_betas"], dtype=torch.float32).unsqueeze(0)
  pose = torch.from_numpy(pose_m)
  vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
  vert = (vert / 1000.0).view(778, 3).numpy().astype(np.float64)
  faces = mano_layer.th_faces.numpy().copy()
  mesh = trimesh.Trimesh(vertices=vert, faces=faces, process=False)
  part = mano_layer.th_weights.numpy().argmax(axis=1)
  finger = _PART_TO_FINGER[part]
  return mesh, finger


def load_object(sample, label, obj_file):
  """Returns the grasped object mesh posed in the camera frame."""
  grasp_ind = sample["ycb_grasp_ind"]
  ycb_id = sample["ycb_ids"][grasp_ind]
  pose = label["pose_y"][grasp_ind]
  assert not np.all(pose == 0.0), "grasped object pose missing"
  mesh = trimesh.load(obj_file[ycb_id], process=False, force="mesh")
  T = np.vstack((pose, [0, 0, 0, 1])).astype(np.float64)
  mesh.apply_transform(T)
  return mesh, ycb_id


def gravity_in_camera(sample, data_dir):
  """Unit gravity vector in this sample's camera frame.

  World 'up' is the apriltag z-axis (table normal); extrinsics map each
  camera frame to the common world frame.
  """
  seq_dir = os.path.dirname(os.path.dirname(sample["color_file"]))
  serial = os.path.basename(os.path.dirname(sample["color_file"]))
  with open(os.path.join(seq_dir, "meta.yml")) as f:
    meta = yaml.safe_load(f)
  extr_file = os.path.join(data_dir, "calibration",
                           "extrinsics_" + meta["extrinsics"],
                           "extrinsics.yml")
  with open(extr_file) as f:
    extr = yaml.unsafe_load(f)["extrinsics"]
  R_cam = np.array(extr[serial], dtype=np.float64).reshape(3, 4)[:, :3]
  R_tag = np.array(extr["apriltag"], dtype=np.float64).reshape(3, 4)[:, :3]
  up_world = R_tag[:, 2]
  g_cam = R_cam.T @ (-up_world)
  return g_cam / np.linalg.norm(g_cam)


def vertex_areas(mesh):
  """Per-vertex area: 1/3 of the area of each incident face."""
  va = np.zeros(len(mesh.vertices))
  np.add.at(va, mesh.faces.ravel(), np.repeat(mesh.area_faces / 3.0, 3))
  return va


def detect_contact(hand_mesh, obj_mesh, thresh):
  """Signed distance of hand vertices to the object surface, contact mask,
  and the object face index closest to each hand vertex."""
  pq = trimesh.proximity.ProximityQuery(obj_mesh)
  closest, dist, tri_id = pq.on_surface(hand_mesh.vertices)
  # Sign from the closest face's outward normal (robust to non-watertight
  # meshes): a vertex behind its closest face is inside the object.
  behind = np.einsum("ij,ij->i", hand_mesh.vertices - closest,
                     obj_mesh.face_normals[tri_id]) < 0
  sd = np.where(behind, dist, -dist)  # positive = inside the object
  contact = sd > -thresh
  return sd, contact, tri_id, closest


def unit(v):
  return v / np.linalg.norm(v)


def cluster_stats(hand_mesh, obj_mesh, finger, contact, sd, tri_id, g_cam,
                  min_verts):
  va = vertex_areas(hand_mesh)
  hand_normals = np.asarray(hand_mesh.vertex_normals)
  clusters = []
  for fi, fname in enumerate(_FINGER_NAMES):
    sel = np.where(contact & (finger == fi))[0]
    if len(sel) < min_verts:
      if len(sel) > 0:
        print("  [skip] %-6s: only %d contact vertices (< %d)" %
              (fname, len(sel), min_verts))
      continue
    area = va[sel].sum()
    # n_k points from the hand into the object.
    n_hand = unit(hand_normals[sel].mean(axis=0))
    n_obj = unit(-obj_mesh.face_normals[tri_id[sel]].mean(axis=0))
    clusters.append({
        "finger": fname,
        "finger_id": fi,
        "verts": sel,
        "n_verts": len(sel),
        "area_m2": area,
        "n_hand": n_hand,
        "n_obj": n_obj,
        "angle_hand_obj_deg": np.degrees(
            np.arccos(np.clip(n_hand @ n_obj, -1, 1))),
        "dot_g_hand": n_hand @ g_cam,
        "dot_g_obj": n_obj @ g_cam,
        "max_penetration_mm": sd[sel].max() * 1000.0,
        "centroid": hand_mesh.vertices[sel].mean(axis=0),
    })
  return clusters


def look_at(eye, target, up):
  z = unit(eye - target)  # pyrender camera looks along -z
  x = unit(np.cross(up, z))
  y = np.cross(z, x)
  T = np.eye(4)
  T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
  return T


def make_hand_render_mesh(hand_mesh, finger, contact):
  colors = np.tile(_NON_CONTACT_COLOR, (len(hand_mesh.vertices), 1))
  colors[contact] = _FINGER_COLORS[finger[contact]]
  m = trimesh.Trimesh(vertices=hand_mesh.vertices.copy(),
                      faces=hand_mesh.faces.copy(),
                      vertex_colors=colors,
                      process=False)
  return pyrender.Mesh.from_trimesh(m)


def render_overlay(sample, hand_rmesh, obj_mesh, w, h):
  """Camera-view render blended with the real image (pyrender flips y/z)."""
  flip = np.diag([1.0, -1.0, -1.0, 1.0])
  scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
  intr = sample["intrinsics"]
  cam = pyrender.IntrinsicsCamera(intr["fx"], intr["fy"], intr["ppx"],
                                  intr["ppy"])
  scene.add(cam, pose=np.eye(4))
  scene.add(hand_rmesh, pose=flip)
  obj_r = pyrender.Mesh.from_trimesh(obj_mesh.copy())
  scene.add(obj_r, pose=flip)
  r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
  im_render, _ = r.render(scene)
  r.delete()
  im_real = cv2.imread(sample["color_file"])[:, :, ::-1]
  im = (0.33 * im_real.astype(np.float32) +
        0.67 * im_render.astype(np.float32)).astype(np.uint8)
  return im


def render_orbit(sample, hand_rmesh, obj_mesh, g_cam, w, h, n_views=4,
                 radius=0.45, with_object=True):
  """Renders the contact-colored hand from cameras orbiting the object,
  with 'up' anti-parallel to gravity."""
  center = obj_mesh.vertices.mean(axis=0)
  up = -g_cam
  u = unit(np.cross(up, [0.0, 0.0, 1.0]))
  v = np.cross(up, u)
  intr = sample["intrinsics"]
  cam = pyrender.IntrinsicsCamera(intr["fx"], intr["fy"], intr["ppx"],
                                  intr["ppy"])
  obj_gray = pyrender.Mesh.from_trimesh(
      trimesh.Trimesh(vertices=obj_mesh.vertices, faces=obj_mesh.faces,
                      process=False),
      material=pyrender.MetallicRoughnessMaterial(
          baseColorFactor=[0.6, 0.75, 0.85, 0.45], alphaMode="BLEND"))
  ims = []
  r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
  for i in range(n_views):
    th = 2 * np.pi * i / n_views
    eye = center + radius * (np.cos(th) * u + np.sin(th) * v) + 0.10 * up
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=[0.4, 0.4, 0.4])
    scene.add(cam, pose=look_at(eye, center, up))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
    scene.add(light, pose=look_at(eye, center, up))
    scene.add(hand_rmesh)
    if with_object:
      scene.add(obj_gray)
    im, _ = r.render(scene)
    ims.append(im)
  r.delete()
  return ims


def main():
  parser = argparse.ArgumentParser(
      description="Proximity-threshold hand-object contact extraction")
  parser.add_argument("--name", default="s0_train")
  parser.add_argument("--idx", type=int, default=421470)
  parser.add_argument("--thresh", type=float, default=0.005,
                      help="proximity threshold in meters")
  parser.add_argument("--min_verts", type=int, default=3,
                      help="min contact vertices per finger cluster")
  parser.add_argument("--out_dir", default=_DEFAULT_OUT_DIR)
  args = parser.parse_args()

  dataset = get_dataset(args.name)
  sample = dataset[args.idx]
  label = np.load(sample["label_file"])
  print("sample %d: %s" % (args.idx, sample["color_file"]))

  hand_mesh, finger = load_hand(sample, label)
  obj_mesh, ycb_id = load_object(sample, label, dataset.obj_file)
  print("object: %s (watertight=%s), hand side: %s" %
        (dataset.ycb_classes[ycb_id], obj_mesh.is_watertight,
         sample["mano_side"]))

  g_cam = gravity_in_camera(sample, dataset.data_dir)
  print("gravity in camera frame: [%+.4f %+.4f %+.4f]" % tuple(g_cam))

  sd, contact, tri_id, _ = detect_contact(hand_mesh, obj_mesh, args.thresh)
  for t in (0.0025, 0.005, 0.010):
    print("  threshold %4.1f mm -> %3d contact vertices" %
          (t * 1000, (sd > -t).sum()))
  print("contact vertices @ %.1f mm: %d (max penetration %.2f mm)" %
        (args.thresh * 1000, contact.sum(), sd.max() * 1000))

  clusters = cluster_stats(hand_mesh, obj_mesh, finger, contact, sd, tri_id,
                           g_cam, args.min_verts)
  print("\n%-6s %6s %10s   %-26s %-26s %6s %8s %8s %8s" %
        ("finger", "nverts", "area[cm2]", "n_k(hand)", "n_k(object)",
         "ang[d]", "nh.g", "no.g", "pen[mm]"))
  for c in clusters:
    print("%-6s %6d %10.3f   [%+.3f %+.3f %+.3f]     [%+.3f %+.3f %+.3f]"
          "     %6.1f %+8.3f %+8.3f %8.2f" %
          ((c["finger"], c["n_verts"], c["area_m2"] * 1e4) +
           tuple(c["n_hand"]) + tuple(c["n_obj"]) +
           (c["angle_hand_obj_deg"], c["dot_g_hand"], c["dot_g_obj"],
            c["max_penetration_mm"])))

  os.makedirs(args.out_dir, exist_ok=True)
  np.savez(
      os.path.join(args.out_dir, "contact_%d.npz" % args.idx),
      idx=args.idx,
      thresh=args.thresh,
      gravity_cam=g_cam,
      signed_distance=sd,
      contact_mask=contact,
      finger_label=finger,
      cluster_fingers=np.array([c["finger"] for c in clusters]),
      cluster_n_verts=np.array([c["n_verts"] for c in clusters]),
      cluster_area_m2=np.array([c["area_m2"] for c in clusters]),
      cluster_n_hand=np.array([c["n_hand"] for c in clusters]),
      cluster_n_obj=np.array([c["n_obj"] for c in clusters]),
      cluster_centroid=np.array([c["centroid"] for c in clusters]),
      hand_vertices=hand_mesh.vertices,
      object_center=obj_mesh.vertices.mean(axis=0),
  )

  hand_rmesh = make_hand_render_mesh(hand_mesh, finger, contact)
  im_overlay = render_overlay(sample, hand_rmesh, obj_mesh, dataset.w,
                              dataset.h)
  ims_orbit = render_orbit(sample, hand_rmesh, obj_mesh, g_cam, dataset.w,
                           dataset.h)
  ims_hand = render_orbit(sample, hand_rmesh, obj_mesh, g_cam, dataset.w,
                          dataset.h, with_object=False)

  fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
  axes[0, 0].imshow(im_overlay)
  axes[0, 0].set_title("camera view overlay")
  axes[0, 1].axis("off")
  axes[0, 1].text(
      0.0, 0.5, "\n".join(
          ["idx %d, thresh %.1f mm" % (args.idx, args.thresh * 1000),
           "%-7s %7s %7s %7s" % ("finger", "nverts", "cm2", "n.g")] +
          ["%-7s %7d %7.2f %+7.2f" %
           (c["finger"], c["n_verts"], c["area_m2"] * 1e4, c["dot_g_obj"])
           for c in clusters]),
      fontsize=9, family="monospace", va="center")
  axes[0, 2].axis("off")
  handles = [
      plt.Rectangle((0, 0), 1, 1, color=_FINGER_COLORS[fi, :3] / 255.0)
      for fi in range(len(_FINGER_NAMES))
  ] + [plt.Rectangle((0, 0), 1, 1, color=_NON_CONTACT_COLOR[:3] / 255.0)]
  axes[0, 2].legend(handles, _FINGER_NAMES + ["no contact"], loc="center",
                    fontsize=9, frameon=False)
  axes[0, 3].axis("off")
  for i, im in enumerate(ims_orbit):
    axes[1, i].imshow(im)
    axes[1, i].set_title("orbit %d deg" % (i * 90))
  for i, im in enumerate(ims_hand):
    axes[2, i].imshow(im)
    axes[2, i].set_title("hand only, %d deg" % (i * 90))
  for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
  plt.tight_layout()
  out_png = os.path.join(args.out_dir, "contact_%d.png" % args.idx)
  plt.savefig(out_png, dpi=150, bbox_inches="tight")
  print("\nsaved %s" % out_png)


if __name__ == "__main__":
  main()
