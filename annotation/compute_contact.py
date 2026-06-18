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
#
# [개요] 하나의 DexYCB 샘플에 대해 손-물체 접촉을 추출하는 파이프라인의 1단계.
#   여기서 만든 cluster 정보(손가락별 normal n_k, area_k)가 solve_pressure.py의
#   SOCP 입력이 된다. scan_contact_scenes.py / solve_pressure.py가 이 파일의
#   함수들을 import해서 재사용하므로 파이프라인의 토대 역할.

import os

# 단독 실행/headless 환경 대비: 렌더링 백엔드(egl)와 데이터셋 경로를 미리 세팅.
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

# MANO 모델(.pkl) 위치 후보. 상대경로(레포 구조 기준) → 절대경로 순으로 fallback.
_MANO_ROOT_CANDIDATES = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "dex-ycb-toolkit",
        "manopth",
        "mano",
        "models",
    ),
    "/home/sjkim/Research/pressure/dex-ycb-toolkit/manopth/mano/models",
]
# 결과(png/npz) 기본 저장 폴더: annotation/vis/contact/
_DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vis", "contact"
)

# MANO kinematic tree part order (th_weights columns):
#   0 = wrist/palm, 1-3 = index, 4-6 = middle, 7-9 = little, 10-12 = ring, 13-15 = thumb.
# [주의] little과 ring 순서가 직관과 다름.
#   _PART_TO_FINGER는 16개 part(관절)를 6개 손가락 라벨(0=palm..5=thumb)로 매핑한다.
#   각 vertex는 skinning weight가 가장 큰 part에 속한 것으로 보고 이 표로 손가락을 정한다 (load_hand 참고).
_PART_TO_FINGER = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
_FINGER_NAMES = ["palm", "index", "middle", "little", "ring", "thumb"]
# 손가락별 시각화 색(RGBA). _FINGER_NAMES와 인덱스가 1:1 대응.
_FINGER_COLORS = np.array(
    [
        [177, 89, 40, 255],  # palm   - brown
        [51, 160, 44, 255],  # index  - green
        [31, 120, 180, 255],  # middle - blue
        [106, 61, 154, 255],  # little - purple
        [255, 127, 0, 255],  # ring   - orange
        [227, 26, 28, 255],  # thumb  - red
    ],
    dtype=np.uint8,
)
_NON_CONTACT_COLOR = np.array(
    [190, 190, 190, 255], dtype=np.uint8
)  # 비접촉 vertex 회색


# MANO 모델(.pkl) 폴더 경로를 후보들 중에서 찾아 반환.
def find_mano_root():
    # 후보 경로 중 MANO_LEFT.pkl이 실제로 있는 첫 폴더를 반환. 없으면 에러.
    for c in _MANO_ROOT_CANDIDATES:
        if os.path.isfile(os.path.join(c, "MANO_LEFT.pkl")):
            return c
    raise FileNotFoundError("MANO models not found in: %s" % _MANO_ROOT_CANDIDATES)


# MANO 파라미터로 손 mesh(카메라 좌표)와 vertex별 손가락 라벨을 만든다.
def load_hand(sample, label):
    """Returns hand mesh (camera frame, meters) and per-vertex finger labels."""
    # pose_m: MANO 파라미터 (0:48 = global rot 3 + PCA pose 45, 48:51 = translation).
    pose_m = label["pose_m"]
    # 손이 안 잡힌 프레임은 pose_m이 전부 0 → 그런 프레임은 사용 불가.
    assert not np.all(pose_m == 0.0), "hand is not present in this frame"
    # DexYCB 라벨은 PCA(45 comp) + flat_hand_mean=False 규약으로 생성됨. 동일하게 맞춤.
    mano_layer = ManoLayer(
        flat_hand_mean=False,
        ncomps=45,
        side=sample["mano_side"],  # 'right' or 'left'
        mano_root=find_mano_root(),
        use_pca=True,
    )
    betas = torch.tensor(sample["mano_betas"], dtype=torch.float32).unsqueeze(
        0
    )  # 손 모양(shape)
    pose = torch.from_numpy(pose_m)
    # MANO forward → 778개 vertex (mm 단위). pose 48 + trans 3을 넘긴다.
    vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
    # mm → m 변환, (778,3) float64로. 라벨이 카메라 좌표계라 결과도 카메라 좌표.
    vert = (vert / 1000.0).view(778, 3).numpy().astype(np.float64)
    faces = mano_layer.th_faces.numpy().copy()
    mesh = trimesh.Trimesh(vertices=vert, faces=faces, process=False)
    # vertex별로 영향이 가장 큰 part(skinning weight argmax) → 손가락 라벨로 변환.
    part = mano_layer.th_weights.numpy().argmax(axis=1)
    finger = _PART_TO_FINGER[part]
    return mesh, finger


# 잡고 있는 YCB 물체 mesh를 pose로 변환해 카메라 좌표계에 배치한다.
def load_object(sample, label, obj_file):
    """Returns the grasped object mesh posed in the camera frame."""
    # 샘플에 여러 물체가 있을 수 있으므로, 잡고 있는 물체의 인덱스를 고른다.
    grasp_ind = sample["ycb_grasp_ind"]
    ycb_id = sample["ycb_ids"][grasp_ind]
    pose = label["pose_y"][grasp_ind]  # 해당 물체의 6D pose (3x4, 카메라 좌표계)
    assert not np.all(pose == 0.0), "grasped object pose missing"
    # 물체 canonical mesh 로드 후 pose로 변환 → 카메라 좌표계에 배치.
    mesh = trimesh.load(obj_file[ycb_id], process=False, force="mesh")
    T = np.vstack((pose, [0, 0, 0, 1])).astype(np.float64)  # 3x4 → 4x4 homogeneous
    mesh.apply_transform(T)
    return mesh, ycb_id


# apriltag(테이블 기준)로부터 중력 방향을 카메라 좌표계 단위벡터로 구한다.
def gravity_in_camera(sample, data_dir):
    """Unit gravity vector in this sample's camera frame.

    World 'up' is the apriltag z-axis (table normal); extrinsics map each
    camera frame to the common world frame.
    """
    # 이 샘플이 속한 시퀀스 폴더와 카메라 serial 번호를 경로에서 추출.
    seq_dir = os.path.dirname(os.path.dirname(sample["color_file"]))
    serial = os.path.basename(os.path.dirname(sample["color_file"]))
    # meta.yml에 이 시퀀스가 쓴 extrinsics 캘리브레이션 id가 들어있다.
    with open(os.path.join(seq_dir, "meta.yml")) as f:
        meta = yaml.safe_load(f)
    extr_file = os.path.join(
        data_dir, "calibration", "extrinsics_" + meta["extrinsics"], "extrinsics.yml"
    )
    # unsafe_load: extrinsics.yml에 OpenCV opencv-matrix 태그가 들어있어 safe_load 불가.
    with open(extr_file) as f:
        extr = yaml.unsafe_load(f)["extrinsics"]
    # 각 카메라/apriltag의 pose(3x4)에서 회전부(3x3)만 사용.
    R_cam = np.array(extr[serial], dtype=np.float64).reshape(3, 4)[:, :3]
    R_tag = np.array(extr["apriltag"], dtype=np.float64).reshape(3, 4)[:, :3]
    # apriltag는 테이블 위에 평평히 부착 → 태그의 z축 = 테이블 표면 수직 = world 'up'.
    up_world = R_tag[:, 2]
    # 중력 = -up. world→camera는 R_cam의 전치(R_cam은 camera→world 회전).
    g_cam = R_cam.T @ (-up_world)
    return g_cam / np.linalg.norm(g_cam)  # 단위벡터로 정규화


# mesh의 vertex별 면적을 계산한다.
def vertex_areas(mesh):
    """Per-vertex area: 1/3 of the area of each incident face."""
    # 삼각형 하나의 면적을 세 꼭짓점에 똑같이 1/3씩 분배 (barycentric/voronoi 근사).
    va = np.zeros(len(mesh.vertices))
    # faces.ravel(): 모든 face의 vertex 인덱스를 일렬로. 각 vertex 위치에 누적 가산.
    np.add.at(va, mesh.faces.ravel(), np.repeat(mesh.area_faces / 3.0, 3))
    return va


# 손 vertex의 물체 표면까지 signed distance와 접촉 마스크/최근접 face를 구한다.
def detect_contact(hand_mesh, obj_mesh, thresh):
    """Signed distance of hand vertices to the object surface, contact mask,
    and the object face index closest to each hand vertex."""
    # 각 손 vertex에서 물체 표면까지의 최근접점/거리/face 인덱스.
    pq = trimesh.proximity.ProximityQuery(obj_mesh)
    closest, dist, tri_id = pq.on_surface(hand_mesh.vertices)
    # Sign from the closest face's outward normal (robust to non-watertight meshes)
    # : a vertex behind its closest face is inside the object.
    # [부호 판정] (vertex - 최근접점)을 최근접 face의 바깥 normal과 내적.
    #   음수면 vertex가 face 뒤쪽(물체 내부) → 관통. watertight가 아니어도 안전.
    behind = (
        np.einsum(
            "ij,ij->i", hand_mesh.vertices - closest, obj_mesh.face_normals[tri_id]
        )
        < 0
    )
    sd = np.where(behind, dist, -dist)  # positive = inside the object
    # 표면 밖이라도 thresh 이내로 가까우면 접촉으로 간주(관통은 sd>0이라 항상 포함).
    contact = sd > -thresh
    return sd, contact, tri_id, closest


# 벡터를 단위벡터로 정규화하는 헬퍼.
def unit(v):
    return v / np.linalg.norm(v)  # 벡터 정규화 헬퍼


# 접촉 vertex를 손가락별로 묶어 cluster 대표값(n_k, area_k 등)을 계산 → SOCP 입력.
def cluster_stats(hand_mesh, obj_mesh, finger, contact, sd, tri_id, g_cam, min_verts):
    # 접촉 vertex를 손가락 단위로 묶어 cluster별 대표값(normal/area 등)을 계산.
    va = vertex_areas(hand_mesh)
    hand_normals = np.asarray(hand_mesh.vertex_normals)
    clusters = []
    for fi, fname in enumerate(_FINGER_NAMES):
        # 이 손가락(fi)이면서 접촉 상태인 vertex 인덱스만 선택.
        sel = np.where(contact & (finger == fi))[0]
        # 접촉 vertex가 너무 적은 손가락은 노이즈로 보고 cluster에서 제외.
        if len(sel) < min_verts:
            if len(sel) > 0:
                print(
                    "  [skip] %-6s: only %d contact vertices (< %d)"
                    % (fname, len(sel), min_verts)
                )
            continue
        area = va[sel].sum()  # cluster 접촉 면적 = 소속 vertex area 합
        # n_k points from the hand into the object.
        # [두 가지 normal] 둘 다 '손→물체' 방향으로 통일.
        #   n_hand: 손 vertex normal 평균 (손가락 패드 곡면이라 noisy)
        #   n_obj : 최근접 물체 face의 바깥 normal에 -부호 (평평한 면에서 안정적)
        #   → 다운스트림 SOCP는 n_obj를 사용(Research context 문제 #3).
        n_hand = unit(hand_normals[sel].mean(axis=0))
        n_obj = unit(-obj_mesh.face_normals[tri_id[sel]].mean(axis=0))
        clusters.append(
            {
                "finger": fname,
                "finger_id": fi,
                "verts": sel,
                "n_verts": len(sel),
                "area_m2": area,
                "n_hand": n_hand,
                "n_obj": n_obj,
                # 두 normal이 얼마나 벌어졌는지(품질 진단용).
                "angle_hand_obj_deg": np.degrees(
                    np.arccos(np.clip(n_hand @ n_obj, -1, 1))
                ),
                # n_k·ĝ ≈ 0 이면 normal이 중력에 수직 → 현재 simplification 성립.
                "dot_g_hand": n_hand @ g_cam,
                "dot_g_obj": n_obj @ g_cam,
                "max_penetration_mm": sd[sel].max()
                * 1000.0,  # 최대 관통 깊이(annotation 품질)
                "centroid": hand_mesh.vertices[sel].mean(
                    axis=0
                ),  # cluster 중심(torque용 r_k 후보)
            }
        )
    return clusters


# eye에서 target을 바라보는 카메라 pose(4x4)를 만든다(렌더링용).
def look_at(eye, target, up):
    # eye에서 target을 바라보는 4x4 카메라 pose 생성(렌더링용).
    z = unit(eye - target)  # pyrender camera looks along -z
    x = unit(np.cross(up, z))
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


# 접촉 여부/손가락별로 색칠한 렌더용 손 mesh를 생성한다.
def make_hand_render_mesh(hand_mesh, finger, contact):
    # 손 mesh를 vertex color로 칠한 렌더용 mesh로 변환.
    colors = np.tile(_NON_CONTACT_COLOR, (len(hand_mesh.vertices), 1))  # 기본 회색
    colors[contact] = _FINGER_COLORS[finger[contact]]  # 접촉 vertex만 손가락 색
    m = trimesh.Trimesh(
        vertices=hand_mesh.vertices.copy(),
        faces=hand_mesh.faces.copy(),
        vertex_colors=colors,
        process=False,
    )
    return pyrender.Mesh.from_trimesh(m)


# 카메라 시점 렌더를 실제 사진과 블렌딩한 정합 확인용 오버레이를 만든다.
def render_overlay(sample, hand_rmesh, obj_mesh, w, h):
    """Camera-view render blended with the real image (pyrender flips y/z)."""
    # 라벨은 OpenCV 좌표(+y 아래, +z 앞), pyrender는 OpenGL 좌표 → y,z 부호 반전 필요.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    # 실제 카메라 내부 파라미터로 렌더(투영이 원본 이미지와 일치하도록).
    intr = sample["intrinsics"]
    cam = pyrender.IntrinsicsCamera(intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
    scene.add(cam, pose=np.eye(4))  # 카메라는 원점, mesh 쪽에 flip 적용
    scene.add(hand_rmesh, pose=flip)
    obj_r = pyrender.Mesh.from_trimesh(obj_mesh.copy())
    scene.add(obj_r, pose=flip)
    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    im_render, _ = r.render(scene)
    r.delete()
    # 실제 사진(BGR→RGB)과 렌더 결과를 0.33:0.67로 블렌딩 → 정합 확인용 오버레이.
    im_real = cv2.imread(sample["color_file"])[:, :, ::-1]
    im = (
        0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
    ).astype(np.uint8)
    return im


# 중력 기준 수평 궤도에서 손(+물체)을 여러 각도로 렌더한 이미지들을 반환한다.
def render_orbit(
    sample, hand_rmesh, obj_mesh, g_cam, w, h, n_views=4, radius=0.45, with_object=True
):
    """Renders the contact-colored hand from cameras orbiting the object,
    with 'up' anti-parallel to gravity."""
    # 물체 중심을 바라보며 중력 반대(up) 기준으로 수평 궤도를 도는 n_views개 시점 렌더.
    center = obj_mesh.vertices.mean(axis=0)
    up = -g_cam
    # up에 수직인 두 축 u,v로 궤도 평면을 구성(궤도가 항상 중력 기준 수평).
    u = unit(np.cross(up, [0.0, 0.0, 1.0]))
    v = np.cross(up, u)
    intr = sample["intrinsics"]
    cam = pyrender.IntrinsicsCamera(intr["fx"], intr["fy"], intr["ppx"], intr["ppy"])
    # 물체는 반투명 회색으로 그려 손의 접촉 색이 잘 보이게 함.
    obj_gray = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(
            vertices=obj_mesh.vertices, faces=obj_mesh.faces, process=False
        ),
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.6, 0.75, 0.85, 0.45], alphaMode="BLEND"
        ),
    )
    ims = []
    r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    for i in range(n_views):
        th = 2 * np.pi * i / n_views  # 0/90/180/270도
        # 궤도 위의 카메라 위치: 평면상 원 + 약간 위(0.10*up)에서 내려다봄.
        eye = center + radius * (np.cos(th) * u + np.sin(th) * v) + 0.10 * up
        # bg는 반드시 float [1.0,...] (pyrender 0.1.45는 int를 255로 나눠 검정이 됨).
        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.4]
        )
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


# 전체 실행: 샘플 로드→mesh 구성→중력→접촉 검출→cluster→npz 저장→시각화.
def main():
    # --- 인자 파싱 ---
    parser = argparse.ArgumentParser(
        description="Proximity-threshold hand-object contact extraction"
    )
    parser.add_argument("--name", default="s0_train")  # setup_split
    parser.add_argument(
        "--idx", type=int, default=421470
    )  # 확정 scene (Research context)
    parser.add_argument(
        "--thresh", type=float, default=0.005, help="proximity threshold in meters"
    )
    parser.add_argument(
        "--min_verts",
        type=int,
        default=3,
        help="min contact vertices per finger cluster",
    )
    parser.add_argument("--out_dir", default=_DEFAULT_OUT_DIR)
    args = parser.parse_args()

    # --- 샘플 로드 ---
    dataset = get_dataset(args.name)
    sample = dataset[args.idx]
    label = np.load(sample["label_file"])  # pose_m(손)/pose_y(물체) 등이 든 npz
    print("sample %d: %s" % (args.idx, sample["color_file"]))

    # --- mesh 구성 (카메라 좌표계) ---
    hand_mesh, finger = load_hand(sample, label)
    obj_mesh, ycb_id = load_object(sample, label, dataset.obj_file)
    print(
        "object: %s (watertight=%s), hand side: %s"
        % (dataset.ycb_classes[ycb_id], obj_mesh.is_watertight, sample["mano_side"])
    )

    # --- 중력 방향 (force equilibrium 및 simplification 검증 기준) ---
    g_cam = gravity_in_camera(sample, dataset.data_dir)
    print("gravity in camera frame: [%+.4f %+.4f %+.4f]" % tuple(g_cam))

    # --- 접촉 검출 + threshold별 민감도 출력 ---
    sd, contact, tri_id, _ = detect_contact(hand_mesh, obj_mesh, args.thresh)
    for t in (0.0025, 0.005, 0.010):
        print(
            "  threshold %4.1f mm -> %3d contact vertices" % (t * 1000, (sd > -t).sum())
        )
    print(
        "contact vertices @ %.1f mm: %d (max penetration %.2f mm)"
        % (args.thresh * 1000, contact.sum(), sd.max() * 1000)
    )

    # --- 손가락별 cluster 통계 계산 + 표 출력 ---
    clusters = cluster_stats(
        hand_mesh, obj_mesh, finger, contact, sd, tri_id, g_cam, args.min_verts
    )
    print(
        "\n%-6s %6s %10s   %-26s %-26s %6s %8s %8s %8s"
        % (
            "finger",
            "nverts",
            "area[cm2]",
            "n_k(hand)",
            "n_k(object)",
            "ang[d]",
            "nh.g",
            "no.g",
            "pen[mm]",
        )
    )
    for c in clusters:
        print(
            "%-6s %6d %10.3f   [%+.3f %+.3f %+.3f]     [%+.3f %+.3f %+.3f]"
            "     %6.1f %+8.3f %+8.3f %8.2f"
            % (
                (c["finger"], c["n_verts"], c["area_m2"] * 1e4)
                + tuple(c["n_hand"])
                + tuple(c["n_obj"])
                + (
                    c["angle_hand_obj_deg"],
                    c["dot_g_hand"],
                    c["dot_g_obj"],
                    c["max_penetration_mm"],
                )
            )
        )

    # --- npz 저장: SOCP(solve_pressure.py)에 필요한 모든 값 ---
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

    # --- 시각화: 카메라뷰 오버레이 + 궤도뷰(물체 포함/미포함) ---
    hand_rmesh = make_hand_render_mesh(hand_mesh, finger, contact)
    im_overlay = render_overlay(sample, hand_rmesh, obj_mesh, dataset.w, dataset.h)
    ims_orbit = render_orbit(sample, hand_rmesh, obj_mesh, g_cam, dataset.w, dataset.h)
    ims_hand = render_orbit(
        sample, hand_rmesh, obj_mesh, g_cam, dataset.w, dataset.h, with_object=False
    )

    # 3x4 grid: [0]행 = 오버레이/통계표/범례, [1]행 = 궤도뷰, [2]행 = 손만 궤도뷰.
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
    axes[0, 0].imshow(im_overlay)
    axes[0, 0].set_title("camera view overlay")
    axes[0, 1].axis("off")
    # 통계표를 monospace 텍스트로 패널에 출력.
    axes[0, 1].text(
        0.0,
        0.5,
        "\n".join(
            [
                "idx %d, thresh %.1f mm" % (args.idx, args.thresh * 1000),
                "%-7s %7s %7s %7s" % ("finger", "nverts", "cm2", "n.g"),
            ]
            + [
                "%-7s %7d %7.2f %+7.2f"
                % (c["finger"], c["n_verts"], c["area_m2"] * 1e4, c["dot_g_obj"])
                for c in clusters
            ]
        ),
        fontsize=9,
        family="monospace",
        va="center",
    )
    axes[0, 2].axis("off")
    # 손가락 색 범례.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_FINGER_COLORS[fi, :3] / 255.0)
        for fi in range(len(_FINGER_NAMES))
    ] + [plt.Rectangle((0, 0), 1, 1, color=_NON_CONTACT_COLOR[:3] / 255.0)]
    axes[0, 2].legend(
        handles, _FINGER_NAMES + ["no contact"], loc="center", fontsize=9, frameon=False
    )
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
