# Physics-based pressure estimation for one DexYCB sample (Research Plan:
# Simplest Case, step 2 + torque equilibrium).
#
# Per contact cluster k (from compute_contact): F_k = fn_k * n_k + ft1_k * t1_k
# + ft2_k * t2_k, n_k = contact-direction inward normal (hand -> object),
# (t1_k, t2_k) = arbitrary orthonormal tangent basis (2D friction). Solve
#   minimize    sum_k ||F_k||^2
#   subject to  sum_k F_k = -m * g_vec          (force equilibrium)
#               sum_k r_k x F_k = 0             (torque equilibrium, --torque)
#               fn_k >= 0, ||(ft1_k, ft2_k)|| <= mu * fn_k
# where r_k = (cluster centroid - object center of mass). Taking moments about
# the COM makes the gravity torque vanish, so no gravity term appears in the
# torque balance. then pressure_k = fn_k / area_k, visualized as a colormap on
# the hand mesh. Pass --no_torque for the original force-only formulation.

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("DEX_YCB_DIR", "/datasets/dexycb")

import argparse

import cvxpy as cp
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh

from dex_ycb_toolkit.factory import get_dataset

import pyrender as _pyrender  # for force-arrow meshes

from compute_contact import (
    _DEFAULT_OUT_DIR,
    _FINGER_NAMES,
    cluster_stats,
    detect_contact,
    gravity_in_camera,
    load_hand,
    load_object,
    make_arrow,
    render_orbit,
    render_overlay,
)

_GRAVITY = 9.81  # m/s^2

# Literature-based defaults for the cracker box (see Research Plan).
_DEFAULT_MASS = 0.411  # kg, 003_cracker_box
_DEFAULT_MU = 0.5  # dry skin on coated cardboard

# Fixed pressure normalization ceiling [kPa] for the colormap, shared by the
# single-scene (solve_pressure) and multi-scene (render_pressure_video) renders.
# Like OpenTouch's fixed tactile max_value (build_demo.py / load_data.py: 3072),
# every frame is clipped to [0, vmax] and normalized by this single value so the
# inferno colors are comparable ACROSS scenes/frames (no per-scene max). Tune
# with --vmax_kpa.
_DEFAULT_VMAX_KPA = 30

# Force-arrow 시각화: 손이 물체에 가하는 접촉력 F_k = fn·n + ft1·t1 + ft2·t2 방향.
# 길이는 pressure colormap의 vmax처럼 "고정 기준"으로 normalize한다(frame마다 바꾸지
# 않음 → frame/scene 간 비교 가능). |F_k| = _FORCE_ARROW_REF_N 이 _FORCE_ARROW_MAXLEN
# 에 대응하고 그 이상은 clip → torque balance에서 나오는 heavy tail(물체 무게의 최대
# ~40배 internal force)이 화면을 벗어나지 않는다. ref≈cracker 무게(4N)의 2.5배:
# 보통 엄지 ~8N→~5cm, 손가락 ~1N→~0.7cm. pressure inferno와 대비되도록 밝은 초록.
_FORCE_ARROW_COLOR = np.array([40, 220, 70, 255], dtype=np.uint8)
_FORCE_ARROW_REF_N = 10.0  # 이 접촉력[N]이 max 화살표 길이에 대응 (고정 기준)
_FORCE_ARROW_MAXLEN = 0.06  # m, |F_k| ≥ _FORCE_ARROW_REF_N 일 때의 화살표 길이


def force_arrows(kept, normals, tangents1, tangents2, fn, ft1, ft2):
    """Per-contact force vector F_k as arrows (centroid origin, length ∝ clip(|F_k|))."""
    arrows = []
    for c, n, t1, t2, a, b, d in zip(kept, normals, tangents1, tangents2, fn, ft1, ft2):
        fvec = a * np.asarray(n) + b * np.asarray(t1) + d * np.asarray(t2)
        mag = np.linalg.norm(fvec)
        if mag < 1e-6:
            continue
        arrows.append(
            _pyrender.Mesh.from_trimesh(
                make_arrow(
                    c["centroid"],
                    fvec,
                    _FORCE_ARROW_COLOR,
                    length=min(mag / _FORCE_ARROW_REF_N, 1.0) * _FORCE_ARROW_MAXLEN,
                    shaft_radius=0.0026,
                ),
                smooth=False,
            )
        )
    return arrows


# 1D friction 전용: n_k가 중력과 거의 평행하면 anti-gravity tangent t1이 undefined
# (support-case singularity) → 해당 cluster 제외. 2D는 generic_tangent를 써서
# 이 제외가 필요 없다(basis 방향이 결과에 무관, support도 풀림).
_SINGULAR_TANGENT_NORM = 0.2


def friction_tangent(n, g_hat):
    """1D-friction tangent basis: t1 = anti-gravity projection, t2 = n x t1.

    Returns None when n is ~parallel to gravity (t1 undefined). Used ONLY for the
    1D formulation, where the single friction axis must be the anti-gravity one.
    """
    t1 = -g_hat - (-g_hat @ n) * n
    norm = np.linalg.norm(t1)
    if norm < _SINGULAR_TANGENT_NORM:
        return None
    t1 = t1 / norm
    return t1, np.cross(n, t1)


def generic_tangent(n):
    """Arbitrary orthonormal tangent basis (t1, t2) perpendicular to n.

    For 2D friction the basis orientation does not affect the solution (the
    friction set is an isotropic disk), so any basis works -- and unlike
    friction_tangent it is always defined, so support contacts (n ~parallel to
    gravity) are handled instead of excluded.
    """
    n = np.asarray(n, dtype=np.float64)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def object_com(obj_mesh):
    """Object center of mass in the camera frame (uniform-density assumption).

    Torque is balanced about the COM so that gravity (applied at the COM)
    contributes no moment. Uses trimesh's volume centroid when the mesh is
    watertight; otherwise falls back to the vertex mean (same convention as the
    object_center saved by compute_contact). Returns (com, source_str).
    """
    if obj_mesh.is_watertight:
        return np.asarray(obj_mesh.center_mass, dtype=np.float64), "volume center_mass"
    return (
        obj_mesh.vertices.mean(axis=0).astype(np.float64),
        "vertex mean (non-watertight)",
    )


def solve_min_effort(
    normals,
    tangents1,
    tangents2,
    g_hat,
    mass,
    mu,
    friction="1d",
    arms=None,
    torque=False,
):
    """Min-L2-norm contact forces under force equilibrium + friction cone.

    friction='1d': friction only along t1 (anti-gravity projection),
    |ft1| <= mu*fn. friction='2d': full cone, ||(ft1, ft2)|| <= mu*fn.
    torque=True additionally enforces torque equilibrium sum_k r_k x F_k = 0,
    with r_k taken from `arms` (K x 3, contact centroid - object COM); requires
    `arms`. Returns (fn, ft1, ft2, status); force values are None unless solved.
    """
    K = len(normals)
    fn = cp.Variable(K, nonneg=True)
    ft1 = cp.Variable(K)
    ft2 = cp.Variable(K)
    N = np.asarray(normals)  # (K, 3)
    T1 = np.asarray(tangents1)
    T2 = np.asarray(tangents2)
    weight = mass * _GRAVITY * g_hat
    constraints = [
        N.T @ fn + T1.T @ ft1 + T2.T @ ft2 == -weight,
    ]
    if torque:
        if arms is None:
            raise ValueError("torque equilibrium requires arms (r_k = centroid - COM)")
        R = np.asarray(arms)  # (K, 3)
        # r_k x F_k is linear in the force scalars:
        #   sum_k fn_k (r_k x n_k) + ft1_k (r_k x t1_k) + ft2_k (r_k x t2_k) = 0.
        # Stack the per-contact cross products as 3 x K so each matrix @ vector
        # sums the contributions of all contacts (mirrors the force balance above).
        Cn = np.cross(R, N).T  # (3, K)
        Ct1 = np.cross(R, T1).T
        Ct2 = np.cross(R, T2).T
        constraints += [Cn @ fn + Ct1 @ ft1 + Ct2 @ ft2 == np.zeros(3)]
    if friction == "1d":
        constraints += [ft2 == 0, cp.abs(ft1) <= mu * fn]
    else:
        constraints += [cp.norm(cp.vstack([ft1, ft2]), axis=0) <= mu * fn]
    # n_k, t1_k, t2_k are orthonormal, so ||F_k||^2 = fn^2 + ft1^2 + ft2^2.
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(fn) + cp.sum_squares(ft1) + cp.sum_squares(ft2)),
        constraints,
    )
    prob.solve(solver=cp.ECOS)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None, None, None, prob.status
    return fn.value, ft1.value, ft2.value, prob.status


def pressure_colors(hand_mesh, clusters, pressures_pa, vmax_pa, cmap_name="inferno"):
    """Per-vertex RGBA colors: cluster pressure on contact verts, gray rest.

    Normalized by the FIXED ceiling vmax_pa (clipped to [0, vmax_pa]), not the
    per-scene max, so colors are comparable across scenes (OpenTouch convention).
    """
    pressure_v = np.zeros(len(hand_mesh.vertices))
    for c, p in zip(clusters, pressures_pa):
        pressure_v[c["verts"]] = p
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = np.tile(
        np.array([190, 190, 190, 255], dtype=np.uint8), (len(hand_mesh.vertices), 1)
    )
    hot = (cmap(np.clip(pressure_v / vmax_pa, 0.0, 1.0)) * 255).astype(np.uint8)
    colors[pressure_v > 0] = hot[pressure_v > 0]
    return colors


def main():
    parser = argparse.ArgumentParser(
        description="Min-effort contact force / pressure for one sample"
    )
    parser.add_argument("--name", default="s0_train")
    parser.add_argument("--idx", type=int, default=421470)
    parser.add_argument("--thresh", type=float, default=0.005)
    parser.add_argument("--min_verts", type=int, default=3)
    parser.add_argument("--mass", type=float, default=_DEFAULT_MASS)
    parser.add_argument("--mu", type=float, default=_DEFAULT_MU)
    parser.add_argument(
        "--friction",
        choices=["1d", "2d"],
        default="2d",
        help="2d: full friction cone (default, handles support); "
        "1d: plan formulation (excludes support singularity)",
    )
    parser.add_argument(
        "--no_torque",
        dest="torque",
        action="store_false",
        help="disable torque equilibrium (force-only, Step 2). "
        "Default: torque equilibrium enforced.",
    )
    parser.set_defaults(torque=True)
    parser.add_argument(
        "--vmax_kpa",
        type=float,
        default=_DEFAULT_VMAX_KPA,
        help="fixed pressure colormap ceiling [kPa]; pressures are "
        "clipped to [0, vmax] and normalized by it so colors "
        "are comparable across scenes (OpenTouch convention)",
    )
    parser.add_argument(
        "--out_dir", default=os.path.join(os.path.dirname(_DEFAULT_OUT_DIR), "pressure")
    )
    args = parser.parse_args()

    dataset = get_dataset(args.name)
    sample = dataset[args.idx]
    label = np.load(sample["label_file"])
    print("sample %d: %s" % (args.idx, sample["color_file"]))

    hand_mesh, finger = load_hand(sample, label)
    obj_mesh, ycb_id = load_object(sample, label, dataset.obj_file)
    g_cam = gravity_in_camera(sample, dataset.data_dir)
    sd, contact, tri_id, closest = detect_contact(hand_mesh, obj_mesh, args.thresh)
    clusters = cluster_stats(
        hand_mesh, obj_mesh, finger, contact, sd, tri_id, closest, g_cam, args.min_verts
    )
    print(
        "%s, m=%.3f kg, mu=%.2f, %d contact clusters"
        % (dataset.ycb_classes[ycb_id], args.mass, args.mu, len(clusters))
    )

    # 2D는 generic basis라 support contact 포함; 1D는 anti-gravity 축이라 support 제외.
    kept, normals, tangents1, tangents2 = [], [], [], []
    for c in clusters:
        if args.friction == "1d":
            t = friction_tangent(c["n_obj"], g_cam)
            if t is None:
                print(
                    "  [excluded] %s: n_k parallel to gravity (|n.g|=%.2f), 1D only"
                    % (c["label"], abs(c["dot_g_obj"]))
                )
                continue
        else:
            t = generic_tangent(c["n_obj"])
        kept.append(c)
        normals.append(c["n_obj"])
        tangents1.append(t[0])
        tangents2.append(t[1])
    if len(kept) < 2:
        print("infeasible: fewer than 2 usable contact clusters")
        return

    # Torque is balanced about the object COM: r_k = cluster centroid - COM.
    com, com_src = object_com(obj_mesh)
    arms = np.array([c["centroid"] for c in kept]) - com
    if args.torque:
        print(
            "torque equilibrium ON, COM (%s): [%+.4f %+.4f %+.4f]"
            % ((com_src,) + tuple(com))
        )

    fn, ft1, ft2, status = solve_min_effort(
        normals,
        tangents1,
        tangents2,
        g_cam,
        args.mass,
        args.mu,
        args.friction,
        arms=arms,
        torque=args.torque,
    )
    print(
        "solver status (%s friction, torque=%s): %s"
        % (args.friction, args.torque, status)
    )
    if fn is None:
        print("infeasible problem logged and excluded (see Research Plan)")
        return

    areas = np.array([c["area_m2"] for c in kept])
    pressures = fn / areas
    ft_mag = np.hypot(ft1, ft2)
    resid = (
        np.asarray(normals).T @ fn
        + np.asarray(tangents1).T @ ft1
        + np.asarray(tangents2).T @ ft2
        + args.mass * _GRAVITY * g_cam
    )
    print("force equilibrium residual: %.2e N" % np.linalg.norm(resid))
    # Net torque about the COM: sum_k r_k x F_k (should be ~0 when --torque on).
    fvecs = (
        fn[:, None] * np.asarray(normals)
        + ft1[:, None] * np.asarray(tangents1)
        + ft2[:, None] * np.asarray(tangents2)
    )
    torque_resid = np.cross(arms, fvecs).sum(axis=0)
    print(
        "torque equilibrium residual: %.2e N*m %s"
        % (
            np.linalg.norm(torque_resid),
            "" if args.torque else "(force-only; not constrained)",
        )
    )
    print(
        "\n%-7s %8s %8s %8s %8s %10s %12s %10s"
        % (
            "patch",
            "fn[N]",
            "ft1[N]",
            "ft2[N]",
            "|F|[N]",
            "area[cm2]",
            "press[kPa]",
            "|ft|/mu*fn",
        )
    )
    for c, fnk, f1k, f2k, pk in zip(kept, fn, ft1, ft2, pressures):
        fmag = np.sqrt(fnk**2 + f1k**2 + f2k**2)
        sat = np.hypot(f1k, f2k) / (args.mu * fnk) if fnk > 1e-9 else 0.0
        print(
            "%-7s %8.3f %8.3f %8.3f %8.3f %10.3f %12.2f %10.2f"
            % (c["label"], fnk, f1k, f2k, fmag, c["area_m2"] * 1e4, pk / 1000.0, sat)
        )
    print(
        "total normal force %.3f N, total friction %.3f N (weight %.3f N)"
        % (fn.sum(), ft_mag.sum(), args.mass * _GRAVITY)
    )

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "%s%s" % (args.friction, "_torque" if args.torque else "")
    np.savez(
        os.path.join(args.out_dir, "pressure_%d_%s.npz" % (args.idx, tag)),
        idx=args.idx,
        mass=args.mass,
        mu=args.mu,
        thresh=args.thresh,
        friction=args.friction,
        torque=args.torque,
        object_com=com,
        arms=arms,
        torque_residual=torque_resid,
        gravity_cam=g_cam,
        fingers=np.array([c["finger"] for c in kept]),
        labels=np.array([c["label"] for c in kept]),
        normals=np.array(normals),
        tangents1=np.array(tangents1),
        tangents2=np.array(tangents2),
        f_normal=fn,
        f_friction1=ft1,
        f_friction2=ft2,
        area_m2=areas,
        pressure_pa=pressures,
    )

    # Pressure colormap on the hand mesh (fixed ceiling, comparable across scenes).
    vmax_pa = args.vmax_kpa * 1000.0
    colors = pressure_colors(hand_mesh, kept, pressures, vmax_pa)
    over = pressures.max() / 1000.0
    if over > args.vmax_kpa:
        print(
            "note: max pressure %.1f kPa exceeds vmax %.1f kPa (color saturates)"
            % (over, args.vmax_kpa)
        )
    pr_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(
            vertices=hand_mesh.vertices.copy(),
            faces=hand_mesh.faces.copy(),
            vertex_colors=colors,
            process=False,
        )
    )
    # 접촉력 F_k 화살표 (길이 ∝ 힘 크기).
    arrows = force_arrows(kept, normals, tangents1, tangents2, fn, ft1, ft2)
    im_overlay = render_overlay(
        sample, pr_mesh, obj_mesh, dataset.w, dataset.h, extra_meshes=arrows
    )
    ims_orbit = render_orbit(
        sample, pr_mesh, obj_mesh, g_cam, dataset.w, dataset.h, extra_meshes=arrows
    )
    ims_hand = render_orbit(
        sample,
        pr_mesh,
        obj_mesh,
        g_cam,
        dataset.w,
        dataset.h,
        with_object=False,
        extra_meshes=arrows,
    )

    fig, axes = plt.subplots(3, 4, figsize=(16, 9.5))
    axes[0, 0].imshow(im_overlay)
    axes[0, 0].set_title("camera view overlay")
    axes[0, 1].axis("off")
    axes[0, 1].text(
        0.0,
        0.5,
        "\n".join(
            [
                "idx %d, m=%.3f kg, mu=%.2f, %s friction%s"
                % (
                    args.idx,
                    args.mass,
                    args.mu,
                    args.friction,
                    ", +torque" if args.torque else "",
                ),
                "%-7s %7s %7s %9s" % ("patch", "fn[N]", "|ft|[N]", "kPa"),
            ]
            + [
                "%-7s %7.2f %7.2f %9.2f" % (c["label"], fnk, ftk, pk / 1000.0)
                for c, fnk, ftk, pk in zip(kept, fn, ft_mag, pressures)
            ]
        ),
        fontsize=9,
        family="monospace",
        va="center",
    )
    sm = matplotlib.cm.ScalarMappable(
        cmap="inferno", norm=plt.Normalize(0.0, args.vmax_kpa)
    )
    fig.colorbar(sm, ax=axes[0, 2], label="pressure [kPa]", fraction=0.4)
    axes[0, 2].axis("off")
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
    out_png = os.path.join(args.out_dir, "pressure_%d_%s.png" % (args.idx, tag))
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print("\nsaved %s" % out_png)


if __name__ == "__main__":
    main()
