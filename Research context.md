# Research Context — Hand-Object Pressure Pseudo-Labeling

**업데이트:** 2026-06-12

> Research Plan.txt의 "검증 계획: Simplest Case" 진행 기록. 이 문서는 나중에 다시 봐도 바로 이어서 작업할 수 있도록 진행 상황 / 발견한 문제 / 코드 구성 / 실행법을 기록한다.

---

## 1. 한 줄 요약

DexYCB cracker box scene(idx 421470)에서 proximity 기반 contact 추출과 min-effort SOCP(torque 제외)까지 구현·검증 완료. force 추정치는 계획서 기대치와 일치 (엄지 ~4.4N / 손가락 ~0.8-1.5N). 단, 계획서의 1D friction formulation은 best-case scene에서도 infeasible임을 확인 → 2D friction(full cone)으로 해결. 모든 코드는 `annotation/` 폴더에 있음 (dex-ycb-toolkit 내부에서 작업하지 않기로 함).

---

## 2. 진행 타임라인 (2026-06-12)

### [Step 1] Contact 추출 + 시각화 (완료)

- MANO hand mesh와 posed YCB object mesh를 카메라 좌표계에서 구성.
- Proximity threshold(기본 5mm, signed distance라 penetration 포함)로 contact vertex 추출 → MANO skinning weight 기반으로 손가락별 clustering → cluster별 대표 normal n_k(hand/object 기준 둘 다)와 area_k 계산.
- 중력 방향: apriltag world frame의 z축(테이블 normal)을 camera frame으로 변환. n_k·ĝ로 "n_k ⊥ gravity" simplification 성립 여부 검증.

### [Step 1.5] Scene 선정 스캔 (완료)

- 처음 계획했던 idx 3752 scene이 부적합한 것으로 판명(아래 문제 #1).
- s0_train의 cracker box 시퀀스 41개 전체를 스캔(stride 4 frame, 카메라 841412060263)해서 "수직으로 세운 박스를 옆면으로 잡고 정적으로 들고 있는" frame 47개를 찾음 → 최적 scene 확정.

**★ 확정 scene: idx 421470 (s0_train)**

- 경로: `datasets/dexycb/20201022-subject-10/20201022_111015/841412060263/color_000048.jpg`
- 오른손. 네 손가락이 박스 한쪽 큰 면, 엄지가 반대 면 (antipodal).
- 5개 손가락 모두 contact, 모든 |n_obj·ĝ| ≤ 0.04.
- 박스 수직도 0.998, 테이블에서 0.21m 들림, frame 40~72 동안 거의 정지 (= static holding 구간. frame 48 사용 중).
- 백업: idx 2588 (20200709-subject-01/20200709_142211, frame 36, 오른손, |n·g| ≤ 0.085, 더 fingertip 위주).
- 주의: idx 188216 (subject-05)은 normal은 깨끗하지만 hand mesh가 박스를 최대 12mm 관통(annotation 오차) → 사용하지 말 것.

### [Step 2] Min-effort force/pressure SOCP, torque 제외 (완료)

- 변수: f_n_k(normal), f_t_k(friction). F_k = f_n·n_k + f_t·t_k.
- minimize Σ||F_k||²  s.t.  ΣF_k = -mg_vec, f_n ≥ 0, friction cone.
- n_k는 object mesh 기준 사용 (아래 문제 #3).
- 결과는 아래 문제 #4, #5 참고.

### [다음 단계] (미착수)

- torque equilibrium 추가.
- 1D friction을 공식 formulation에서 유지할지 결정 필요 (문제 #4).
- cracker box 전체 scene으로 확장, infeasibility rate 측정.
- contact area 정의(threshold)에 따른 pressure 스케일 보정 검토 (문제 #5).

---

## 3. 발견한 문제 / 핵심 결정 사항

### 문제 #1. 원래 scene(idx 3752)은 "윗모서리 덮어잡기" grasp이었음

- Research Plan에 적힌 idx 3752 (20200709_142321, 왼손, frame 32): 박스는 거의 완벽히 수직(장축·g=0.999)이지만, 손이 박스 윗모서리를 덮어 잡아서 index/little 손끝이 박스 "윗면"에 닿음.
- little finger의 n_k·ĝ ≈ +1.0 (frame 32/60/71 모두) → n_k ∥ g라서 t_k(중력 반대방향의 tangential projection)가 undefined가 되는 singularity. 계획서가 배제하려던 support-case singularity가 그대로 발생.
- → scene 교체 결정 (스캔으로 idx 421470 선정, 사용자 승인).

### 문제 #2. Proximity threshold에 따라 contact area가 크게 변함

- idx 3752 기준: 2.5mm→90개, 5mm→139개, 10mm→242개 vertex.
- 5mm 기준 손가락당 area 2~18.5 cm² — 계획서가 pressure 기대치(40/13 kPa) 계산에 깔았던 ~1 cm²(fingertip만)보다 훨씬 큼.
- 손가락이 박스 면에 감기면 fingertip뿐 아니라 중간 마디도 contact에 포함됨(idx 421470의 index: 18.5 cm², tip 비율 0.84).

### 문제 #3. n_k를 hand 기준 vs object 기준 — object 기준으로 결정

- 계획서에서 "구현 단계에서 확인 필요"라고 했던 부분.
- 두 normal이 3~55°까지 벌어짐. hand mesh 기준은 손가락 패드 곡면의 평균이라 기울어지고 noisy함. 박스처럼 평평한 면에서는 object mesh 기준이 훨씬 안정적 → object 기준 n_k 사용 (n_k = -object outward normal, 즉 손→물체 방향).

### 문제 #4. ★가장 중요★ 1D friction은 best-case scene에서도 INFEASIBLE

- idx 421470에서 1D friction SOCP가 infeasible.
- 원인(기하학적): 중력에 수직인 평면에 normal들을 projection하면, 네 손가락 방향이 -132.6° ~ -155.7° 범위에 모여 있는데 엄지의 반대 방향은 -169.3°. 즉 엄지의 수평 힘을 상쇄해야 할 방향이 손가락 cone 바깥에 13.6° 벗어나 있음. 1D friction(수직 방향뿐)으로는 이 lateral gap을 메울 수 없어 force equilibrium 자체가 불가능.
- 계획서의 known limitation("1D friction은 lateral friction을 표현 못해 일부 grasp에서 infeasibility 유발")이 torque 단계가 아니라 force equilibrium 단계에서, 가장 이상적인 scene에서도 발생. 실제 grasp은 완벽한 antipodal이 아니기 때문 (normal들이 ~20° 기울어짐).
- → solve_pressure.py에 `--friction {1d,2d}` 옵션 추가. 1d는 계획서 formulation 그대로(infeasible 시 logging 후 종료), 2d는 full friction cone ||(f_t1,f_t2)|| ≤ μ·f_n (SOCP). 2D friction을 기본 formulation으로 승격할지 결정 필요 — 1D는 거의 모든 scene에서 infeasible할 가능성 높음.

### 문제 #5. 2D friction 결과: force는 기대치와 일치, pressure 스케일은 area 때문에 낮음

- idx 421470, m=0.411kg, μ=0.5, torque 제외, equilibrium residual ~1e-10:

  | finger | f_n[N] | \|f_t\|[N] | area[cm²] | pressure[kPa] |
  |--------|--------|-----------|-----------|---------------|
  | thumb  | 4.37   | 2.18      | 7.94      | 5.50          |
  | ring   | 1.46   | 0.73      | 7.57      | 1.92          |
  | middle | 1.42   | 0.71      | 10.68     | 1.32          |
  | index  | 0.90   | 0.45      | 18.54     | 0.49          |
  | little | 0.78   | 0.39      | 2.45      | 3.19          |

  합계: f_n 8.92 N, friction 4.46 N (무게 4.03 N)

- force는 계획서 기대치(엄지 ~4N / 손가락 ~1.3N)와 거의 일치 ✓
- pressure는 0.5~5.5 kPa로 기대치(40/13 kPa)보다 7~25배 낮음 — 전적으로 area 정의 차이(5mm threshold area >> 1cm² 가정). force가 맞으므로 pressure 스케일은 contact area convention 문제로 분리해서 다루면 됨.
- 모든 contact가 friction cone 경계에 있음(|f_t| = μ·f_n). min-effort 해의 특성: normal force를 아끼기 위해 friction을 한계까지 사용. (계획서의 "systematic underestimation" limitation과 일관됨.)

### 문제 #6 (사소). DexYCB annotation 품질

- hand-object 관통이 흔함 (idx 421470은 최대 5mm, idx 188216은 12mm).
- 관통도 contact로 처리(signed distance > -threshold)하고 있음.

---

## 4. 코드 구성 (annotation/ 폴더)

```
annotation/
├── compute_contact.py      # [Step 1] contact 추출 + cluster + 시각화
├── scan_contact_scenes.py  # [Step 1.5] 전체 scene 스캔/랭킹
├── solve_pressure.py       # [Step 2] min-effort SOCP → pressure + 시각화
└── vis/
    ├── contact/            # contact_<idx>.png (시각화), contact_<idx>.npz,
    │                       # scan_results.npz (스캔 랭킹 47개)
    └── pressure/           # pressure_<idx>_<1d|2d>.png / .npz
```

### compute_contact.py — 핵심 함수 (다른 스크립트에서 import해서 재사용)

- **load_hand(sample, label):** MANO mesh(778 verts, 카메라 좌표, m 단위)와 vertex별 손가락 라벨(skinning weight argmax; MANO part 순서는 wrist/index/middle/little/ring/thumb 주의 — 파일 상단 주석 참고).
- **load_object(...):** 잡은 YCB object mesh를 pose_y로 변환해 카메라 좌표로.
- **gravity_in_camera(...):** apriltag extrinsics에서 중력 단위벡터를 카메라 frame으로. (calibration/extrinsics_<id>/extrinsics.yml의 'apriltag' 항목 z축이 테이블 위쪽.)
- **detect_contact(...):** trimesh ProximityQuery로 hand vertex의 signed distance (closest face normal 부호 판정이라 non-watertight mesh OK). contact = sd > -threshold (관통 포함).
- **cluster_stats(...):** 손가락별 cluster의 n_k(hand/object), area_k (vertex당 인접 face 면적의 1/3 합), n·g, 관통 깊이.
- **render_overlay / render_orbit:** pyrender EGL 렌더. orbit은 중력 기준 수직 up으로 4방향. 출력 figure는 3x4 grid (overlay+통계 / orbit / hand-only).
- npz에는 SOCP에 필요한 모든 값 저장 (cluster normals/areas/centroids, gravity, hand verts, object center).

### scan_contact_scenes.py

- s0_train에서 cracker box를 잡는 시퀀스를 찾아 stride frame마다 평가.
- 빠른 prefilter(라벨만 사용): 박스 수직(|장축·(-g)|>0.95), 들림(frame 0 대비 +3cm), 손 존재 → 통과한 frame만 contact 계산.
- 속도 최적화: hand vertex를 object canonical frame으로 역변환해서 ProximityQuery를 한 번만 빌드. 41개 시퀀스 전체 ~26초.
- 통과 기준: thumb 포함 cluster ≥ 4개. worst |n_obj·g| 오름차순 랭킹 출력, vis/contact/scan_results.npz에 저장.

### solve_pressure.py

- compute_contact의 파이프라인을 import해 contact cluster를 다시 계산한 뒤 cvxpy(ECOS)로 min-effort 문제 풀이. torque equilibrium은 아직 미포함.
- **friction_tangent():** t1 = -ĝ의 tangential projection(단위벡터), t2 = n × t1. ||projection|| < 0.2면 singularity로 해당 cluster 제외+로깅.
- `--friction 1d`: f_t2=0, |f_t1| ≤ μf_n (계획서 formulation, QP). `--friction 2d`: ||(f_t1,f_t2)|| ≤ μf_n (full cone, SOCP).
- infeasible이면 status 로깅 후 종료(계획서 방침). optimal이면 pressure_k = f_n_k / area_k 계산, 손 mesh에 inferno colormap 시각화.

---

## 5. 실행 방법

### 환경

- conda env: dexycb-toolkit (`/ssd/sjkim/anaconda3/envs/dexycb-toolkit/bin/python`, Python 3.7). base/dexycb env에는 torch가 없으니 주의.
- 이 env에 2026-06-12 추가 설치한 패키지: rtree (trimesh proximity용), cvxpy+ECOS (SOCP용).
- 환경변수는 스크립트가 직접 세팅함: DEX_YCB_DIR=/datasets/dexycb, PYOPENGL_PLATFORM=egl (headless 렌더링).
- MANO 모델: dex-ycb-toolkit/manopth/mano/models (gitignore됨). annotation/ 스크립트는 상대경로 → 절대경로 순으로 fallback (compute_contact.py의 find_mano_root()).

### 명령어 (annotation/ 디렉토리에서 실행)

```Shell
# 1) contact 추출 + 시각화 (기본 idx 421470)
/ssd/sjkim/anaconda3/envs/dexycb-toolkit/bin/python compute_contact.py
/ssd/sjkim/anaconda3/envs/dexycb-toolkit/bin/python compute_contact.py --idx 2588 --thresh 0.0025

# 2) scene 스캔 (41개 시퀀스, ~30초)
/ssd/sjkim/anaconda3/envs/dexycb-toolkit/bin/python scan_contact_scenes.py

# 3) force/pressure 계산 + 시각화
/ssd/sjkim/anaconda3/envs/dexycb-toolkit/bin/python solve_pressure.py --friction 2d
/ssd/sjkim/anaconda3/envs/dexycb-toolkit/bin/python solve_pressure.py --friction 1d   # infeasible 로깅 확인용
```

### 주요 옵션 (공통)

- `--idx` — dataset index (기본 421470. idx↔frame: 같은 시퀀스/카메라에서 연속이므로 idx = (frame 0의 idx) + frame)
- `--thresh` — proximity threshold [m] (기본 0.005)
- `--min_verts` — cluster 최소 vertex 수 (기본 3)
- `--mass/--mu` — solve_pressure만. 기본 0.411kg / 0.5 (cracker box)

### 기타 알아두면 좋은 것

- pyrender 0.1.45에서 Scene(bg_color=[1,1,1,1])처럼 int를 주면 255로 나눠져 검정이 됨 → 반드시 float([1.0,...])로.
- 라벨(pose_y/pose_m)은 카메라 좌표계 기준. visualize용 y/z 부호 반전은 pyrender 카메라 convention 때문이며 물리 계산에서는 하지 않음.
- MANO part→손가락 매핑 순서가 직관과 다름: 1-3 index, 4-6 middle, 7-9 little, 10-12 ring, 13-15 thumb.
