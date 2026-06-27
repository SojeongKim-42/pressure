# Render a video of the 30-degree finger x normal-patch contact clustering
# across many cracker-box sequences.
#
# For every s0_train sequence grasping the cracker box, each frame (at a stride)
# where the hand and the box are both present is processed with the same
# pipeline as compute_contact.py (detect_contact + cluster_stats), then rendered
# as a two-panel frame:
#   left  = camera-view overlay (contact-colored hand + object on the real image)
#   right = hand-only turntable view (rotating about gravity so occluded patches
#           become visible), with the translucent object for context.
# Each contact cluster (= one finger patch) gets its own color; on-frame text
# lists the sequence, frame, cluster count and per-patch n.g. Frames are written
# to one mp4 walking through all selected sequences.
#
# NOTE: no vertical/lift prefilter -- we want to SEE the clustering on the full
# grasp, including messy wrap-around grips, not just the clean static-hold ones.

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("DEX_YCB_DIR", "/datasets/dexycb")

import argparse
import time

import cv2
import numpy as np
import pyrender
import trimesh

from dex_ycb_toolkit.factory import get_dataset

from compute_contact import (_NON_CONTACT_COLOR, _SPLIT_ANGLE_DEG,
                             cluster_arrows, cluster_color, cluster_stats,
                             detect_contact, gravity_in_camera, load_hand,
                             look_at, unit)
from scan_contact_scenes import _CRACKER_YCB_ID, _SERIAL

_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV(label) -> OpenGL(pyrender)


# 연속 프레임에서 같은 접촉 patch가 같은 색을 유지하도록 색 id를 추적한다.
class ColorTracker:
    """Matches each cluster to the previous frame's clusters by (finger_id,
    contact-normal angle) and reuses its color id; unmatched clusters get a
    new id. reset() is called at each sequence boundary."""

    def __init__(self, angle_thresh=_SPLIT_ANGLE_DEG):
        self.cos_thresh = np.cos(np.radians(angle_thresh))
        self.reset()

    def reset(self):
        self.prev = []  # (finger_id, n_obj, color_id)
        self.next_id = 0

    def assign(self, clusters):
        ids = []
        used = set()
        for c in clusters:
            best_cid, best_cos = None, self.cos_thresh
            for fid, n, cid in self.prev:
                if fid != c["finger_id"] or cid in used:
                    continue
                cval = float(np.dot(n, c["n_obj"]))
                if cval >= best_cos:
                    best_cos, best_cid = cval, cid
            if best_cid is None:
                best_cid, self.next_id = self.next_id, self.next_id + 1
            used.add(best_cid)
            ids.append(best_cid)
        self.prev = [(c["finger_id"], c["n_obj"], i)
                     for c, i in zip(clusters, ids)]
        return ids


# cluster를 (추적된) 색 id로 칠한 렌더용 손 mesh를 만든다.
def colored_hand_mesh(hand_mesh, clusters, color_ids):
    colors = np.tile(_NON_CONTACT_COLOR, (len(hand_mesh.vertices), 1))
    for c, cid in zip(clusters, color_ids):
        colors[c["verts"]] = cluster_color(cid)
    m = trimesh.Trimesh(vertices=hand_mesh.vertices.copy(),
                        faces=hand_mesh.faces.copy(), vertex_colors=colors,
                        process=False)
    return pyrender.Mesh.from_trimesh(m)


# 카메라/물체 mesh를 매 프레임 재로딩하지 않도록 persistent 렌더러로 묶는다.
class FrameRenderer:
    def __init__(self, intr, w, h):
        self.w, self.h = w, h
        self.cam = pyrender.IntrinsicsCamera(intr["fx"], intr["fy"], intr["ppx"],
                                             intr["ppy"])
        self.r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    # 카메라 시점: 접촉 색 손 + 물체를 실제 사진 위에 블렌딩.
    def camera_overlay(self, hand_rmesh, obj_mesh_cam, color_file, extra_meshes=None):
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        scene.add(self.cam, pose=np.eye(4))
        scene.add(hand_rmesh, pose=_FLIP)
        scene.add(pyrender.Mesh.from_trimesh(obj_mesh_cam.copy()), pose=_FLIP)
        for m in extra_meshes or []:  # 화살표도 손/물체와 같은 flip 적용
            scene.add(m, pose=_FLIP)
        im_render, _ = self.r.render(scene)
        im_real = cv2.imread(color_file)[:, :, ::-1]
        return (0.33 * im_real.astype(np.float32) +
                0.67 * im_render.astype(np.float32)).astype(np.uint8)

    # 중력 기준 수평 궤도의 한 각도에서 손(+반투명 물체)을 렌더.
    def turntable(self, hand_rmesh, obj_mesh_cam, g_cam, angle, radius=0.45,
                  extra_meshes=None):
        center = obj_mesh_cam.vertices.mean(axis=0)
        up = -g_cam
        u = unit(np.cross(up, [0.0, 0.0, 1.0]))
        v = np.cross(up, u)
        eye = center + radius * (np.cos(angle) * u + np.sin(angle) * v) + 0.10 * up
        pose = look_at(eye, center, up)
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                               ambient_light=[0.4, 0.4, 0.4])
        scene.add(self.cam, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5),
                  pose=pose)
        scene.add(hand_rmesh)
        obj_gray = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(vertices=obj_mesh_cam.vertices,
                            faces=obj_mesh_cam.faces, process=False),
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.6, 0.75, 0.85, 0.45], alphaMode="BLEND"))
        scene.add(obj_gray)
        for m in extra_meshes or []:  # 화살표는 카메라 좌표 그대로(flip 없음)
            scene.add(m)
        im, _ = self.r.render(scene)
        return im

    def close(self):
        self.r.delete()


# 두 패널을 가로로 붙이고 자막을 얹어 한 비디오 프레임(BGR)을 만든다.
def compose_frame(cam_im, turn_im, lines, scale):
    panel = np.hstack([cam_im, turn_im])
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        panel = cv2.resize(panel, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    # 작은 폰트로 왼쪽 정렬; cluster가 많아도 줄바꿈돼 잘리지 않음.
    y = 16
    for ln in lines:
        # 가독성을 위해 검은 외곽선 + 흰 글자.
        cv2.putText(panel, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 17
    return panel


def main():
    parser = argparse.ArgumentParser(
        description="Video of 30-deg contact clustering over cracker-box scenes")
    parser.add_argument("--name", default="s0_train")
    parser.add_argument("--stride", type=int, default=2, help="frame stride")
    parser.add_argument("--thresh", type=float, default=0.005)
    parser.add_argument("--min_verts", type=int, default=3)
    parser.add_argument("--split_angle", type=float, default=_SPLIT_ANGLE_DEG)
    parser.add_argument("--max_seqs", type=int, default=0,
                        help="limit sequences (0 = all cracker-box sequences)")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--scale", type=float, default=0.75,
                        help="output downscale factor")
    parser.add_argument("--out",
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "vis",
                            "contact", "contact_clusters.mp4"))
    args = parser.parse_args()

    dataset = get_dataset(args.name)
    cam_idx = dataset._serials.index(_SERIAL)
    mapping = dataset._mapping
    obj_canon = trimesh.load(dataset.obj_file[_CRACKER_YCB_ID], process=False,
                             force="mesh")

    seq_ids = [s for s in range(len(dataset._sequences))
               if dataset._ycb_ids[s][dataset._ycb_grasp_ind[s]]
               == _CRACKER_YCB_ID]
    if args.max_seqs:
        seq_ids = seq_ids[:args.max_seqs]
    print("%d cracker-box sequences (stride %d, split %.0f deg)" %
          (len(seq_ids), args.stride, args.split_angle))

    renderer = None
    writer = None
    tracker = ColorTracker()
    turn_angle = 0.0
    n_written = 0
    t0 = time.time()

    for s in seq_ids:
        seq_name = dataset._sequences[s]
        grasp_ind = dataset._ycb_grasp_ind[s]
        sel = np.where((mapping[:, 0] == s) & (mapping[:, 1] == cam_idx))[0]
        sel = sel[np.argsort(mapping[sel, 2])]
        tracker.reset()  # 시퀀스 경계에서 색 추적 초기화
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
            sd, contact, tri_id, closest = detect_contact(hand_mesh, obj_mesh,
                                                          args.thresh)
            clusters = cluster_stats(hand_mesh, obj_mesh, finger, contact, sd,
                                     tri_id, closest, g_cam, args.min_verts,
                                     args.split_angle)
            if not clusters:
                continue

            color_ids = tracker.assign(clusters)  # 연속 프레임 색 일관성
            hand_rmesh = colored_hand_mesh(hand_mesh, clusters, color_ids)
            # cluster별 contact-direction normal 화살표(색은 추적 색과 일치).
            arrows = cluster_arrows(clusters,
                                    [cluster_color(cid) for cid in color_ids])
            if renderer is None:
                renderer = FrameRenderer(sample["intrinsics"], dataset.w,
                                         dataset.h)
            cam_im = renderer.camera_overlay(hand_rmesh, obj_mesh,
                                             sample["color_file"],
                                             extra_meshes=arrows)
            turn_im = renderer.turntable(hand_rmesh, obj_mesh, g_cam, turn_angle,
                                         extra_meshes=arrows)
            turn_angle += 2 * np.pi / 90.0  # 90프레임마다 한 바퀴

            # cluster n.g 목록을 줄당 5개씩 wrap해 caption이 잘리지 않게.
            entries = ["%s:%+.2f" % (c["label"], c["dot_g_obj"])
                       for c in clusters]
            per_line = 5
            lines = ["%s  frame %d  (idx %d, %s)" %
                     (seq_name, f, idx, sample["mano_side"]),
                     "%d clusters  (label:n.g)" % len(clusters)]
            lines += ["  ".join(entries[i:i + per_line])
                      for i in range(0, len(entries), per_line)]
            frame = compose_frame(cam_im, turn_im, lines, args.scale)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("VideoWriter failed to open %s" % args.out)
            writer.write(frame)
            n_written += 1
            n_seq += 1
        print("  %s: %d frames  [%d total, %.0fs]" %
              (seq_name, n_seq, n_written, time.time() - t0))

    if renderer is not None:
        renderer.close()
    if writer is not None:
        writer.release()
    print("\nwrote %d frames to %s" % (n_written, args.out))


if __name__ == "__main__":
    main()
