# Research Context — Hand-Object Pressure Pseudo-Labeling

**업데이트:** 2026-06-18

> Research Plan.md의 "검증 계획: Simplest Case" 진행 기록. 이 문서는 나중에 다시 봐도 바로 이어서 작업할 수 있도록 진행 상황 / 발견한 문제 / 코드 구성 / 실행법을 기록한다.

---

## 1. 한 줄 요약

DexYCB cracker box scene(idx 421470)에서 proximity 기반 contact 추출과 min-effort SOCP(torque 제외)까지 구현·검증 완료. force 추정치는 계획서 기대치와 일치 (엄지 ~4.4N / 손가락 ~0.8-1.5N). 단, 계획서의 1D friction formulation은 best-case scene에서도 infeasible임을 확인 → 2D friction(full cone)으로 해결. 모든 코드는 `annotation/` 폴더에 있음 (dex-ycb-toolkit 내부에서 작업하지 않기로 함).

**(2026-06-18 갱신)** contact clustering을 "손가락 단위"에서 **"손가락 × normal-patch"** 로 정교화: 한 손가락이 두 면에 걸치면(예: 모서리 감아쥠) normal을 평균낼 때 엉뚱한 방향이 나오므로 patch별로 분리. 대표 normal n_k도 nearest-face normal 대신 **contact direction(손 vertex→가장 가까운 물체 표면점 방향)의 평균**으로 변경 — 물리적 force 방향이고 임의 물체에 일반화되며 noise가 적음(문제 #3, #7). 분할 임계각 30°는 41개 cracker scene의 within-finger normal spread 분포(bimodal, 골짜기 ~30°)로 결정. 화살표 시각화·검증 비디오 추가. pressure research 전용 conda env `pressure` 신설.

**(2026-06-18 추가)** pressure 예측을 단일 scene(idx 421470)에서 **여러 cracker scene**으로 확장: 수직 holding뿐 아니라 **기운 grip도 포함**, support contact(n∥g)는 제외, 2D friction, infeasible/contact<2 frame은 skip. 손을 예측 pressure로 칠하고 **접촉력 F_k를 화살표**(길이 ∝ |F_k|)로 그린 멀티 scene 비디오(`render_pressure_video.py`) 생성. 1D vs 2D는 grasp 의존적임을 재확인(문제 #4 갱신).

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

### [Step 1.6] within-finger normal-patch 분할 + contact-direction normal (2026-06-18, 완료)

지금까지 한 손가락 = cluster 1개였는데, 한 손가락이 서로 다른 면에 동시에 닿는 경우(모서리 감아쥠, 윗면+옆면 등)를 분리하도록 cluster 단위를 세분화했다.

- **분할 임계각 결정 (analyze_normal_spread.py):** 41개 cracker scene을 스캔해 손가락별 접촉 vertex의 normal이 손가락 평균에서 벌어진 각도 분포를 측정. per-vertex / per-finger MAX / per-finger P90 세 분포 모두 **bimodal**(single-face ~수° vs multi-face ~70°), 골짜기가 ~30°. → **분할 임계각 30°** 확정. P90 기준 약 42%의 손가락 접촉이 multi-patch(>30°)라 분할이 예외가 아니라 흔함. (MAX는 stray vertex 1~2개에도 민감 → P90이 robust한 판정자.)
- **분할 방식 (compute_contact.split_by_normal):** 손가락 내에서 contact-direction 벡터를 average-linkage(cosine)로 군집화, 30° 초과 시 patch 분리. vertex 3개 미만 sub-cluster는 stray로 **drop**(보수적 outlier 제거; merge 안 함).
- **대표 normal을 contact direction으로 변경 (문제 #3 갱신, #7):** n_k = 각 vertex의 `closest_point − hand_vertex`(관통 시 부호 반전; 손→물체 방향) 벡터를 군집 평균낸 것. 평평한 면에선 inward normal과 동일, edge에선 두 면 사이로 blend(nearest-face normal의 90° 스냅 artifact 완화). 순수 기하라 MANO normal보다 noise 적고, face-normal 후보 enumerate 없이 임의 물체에 일반화됨. **군집화·대표값 모두 이 contact direction으로 통일.**
- **검증:** idx 3752(over-edge)는 옆면 patch(n·g≈0.09)와 윗면 patch(n·g≈0.99)로 정확히 분리(예전 little-finger singularity 해소). idx 421470(flat side grip)은 수직 모서리를 감은 부분만 분리. SOCP도 분할 cluster로 optimal(residual ~1e-11) 유지. 분포는 contact-direction으로 재측정해도 거의 동일(30° robust).
- **화살표 시각화 (make_arrow/cluster_arrows):** cluster centroid에서 n_k 방향 화살표를 cluster 색과 동일하게 렌더 → "손→물체로 향하는지" 직접 검증. `compute_contact.py --idx`의 정적 figure와 비디오 모두 포함.
- **검증 비디오 (render_contact_video.py):** 41개 cracker 시퀀스를 좌(카메라 오버레이)/우(중력 기준 turntable, 화살표 포함) 2-패널로 렌더한 mp4. 연속 프레임에서 같은 patch가 같은 색을 유지하도록 ColorTracker(finger_id+normal 각도 매칭). caption은 작은 폰트·좌측 정렬·줄당 5개 wrap.

### [Step 2.5] pressure 멀티 scene 예측 + force 화살표 (2026-06-18, 완료)

단일 scene(idx 421470)에서만 풀던 SOCP를 **여러 cracker scene**으로 확장하고, 예측 결과(접촉력)를 화살표로 시각화했다.

- **force 화살표 (solve_pressure.force_arrows):** 푼 접촉력 F_k = fn·n + ft1·t1 + ft2·t2를 cluster centroid에서 시작하는 초록 화살표로 렌더(길이 ∝ |F_k|, scale 0.022 m/N). pressure inferno colormap과 대비. `solve_pressure.py --idx`의 정적 figure와 비디오 모두 포함.
- **멀티 scene 비디오 (render_pressure_video.py):** 41개 cracker 시퀀스 전체에서 contact→cluster→SOCP(2D friction)→pressure 파이프라인 실행. **수직 prefilter 없음(기운 grip 포함)**, support contact(n∥g, |n·g|>0.98) 제외, 사용 가능 contact<2 또는 SOCP infeasible frame은 skip(사유별 카운트). 손은 예측 pressure(inferno, **고정 vmax 8kPa**로 frame 간 비교 가능)로 칠하고 force 화살표 추가. 좌(카메라 오버레이)/우(turntable) 2-패널 mp4(`vis/pressure/pressure_clusters.mp4`, 729 frames). `solve_pressure`/`render_contact_video`의 함수 재사용.
- **1D vs 2D 재확인 (문제 #4 갱신):** 1D feasibility는 grasp 의존적 — 깔끔한 antipodal(idx 421470)은 1D infeasible/2D optimal이지만, 기운 grip(idx 235246)은 1D도 optimal. 2D의 feasible 집합이 1D의 상위집합(1D = 2D + ft2=0)이라, 멀티 scene에서 가장 많은 frame을 푸는 2D를 기본값으로 채택. ("1D 항상 infeasible"은 과장이었음.)

### [다음 단계] (미착수)

- torque equilibrium 추가.
- cracker box 전체에 대해 SOCP infeasibility/support-제외 rate를 정량 측정(현재 비디오는 skip만 카운트).
- contact area 정의(threshold)에 따른 pressure 스케일 보정 검토 (문제 #5).
- edge-graze patch(문제 #7)를 force 단계에서 어떻게 다룰지 (현재는 분리만; grazing contact의 spurious force 가능성). `ang(hand,obj)` 진단 지표로 모니터링 중.
- 다른 object로 확장 시 분할 임계각 30° 재측정 (현재는 cracker box 기준).

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

### 문제 #3. n_k 정의 — (1차) object 기준 → (2차, 2026-06-18) contact direction

- 계획서에서 "구현 단계에서 확인 필요"라고 했던 부분. **해결됨.**
- (1차 결정) hand mesh vertex normal은 손가락 패드 곡면 평균이라 noisy하고 object normal과 3~55° 벌어짐 → object mesh 기준 채택.
- (2차 결정, 문제 #7 때문) nearest-face normal은 edge에서 가장 가까운 면으로 90° 스냅되는 artifact가 있음. → **n_k = contact direction**(각 vertex의 `closest_point − hand_vertex` 방향, 관continued penetration 부호 반전, 군집 평균)으로 변경. 평면에선 inward object normal과 동일하지만 edge에선 두 면 사이로 blend되고, 순수 기하라 noise가 적으며 임의 물체에 일반화됨. 군집화 기준도 이 벡터로 통일. (compute_contact.contact_directions / split_by_normal / cluster_stats)

### 문제 #4. ★중요★ 1D friction infeasibility는 grasp 의존적 → 2D를 기본값으로

- **1D가 깨지는 대표 케이스 = 깔끔한 antipodal 그립 (idx 421470).** 중력 수직 평면에 normal projection: 네 손가락이 -132.6°~-155.7°에 모여 있는데 엄지 반대 방향은 -169.3°로 손가락 cone 밖 13.6°. 1D friction(중력반대 한 축뿐)은 이 lateral gap을 못 메워 force equilibrium 자체가 infeasible.
- **단, 1D infeasibility는 항상은 아님 — grasp 기하 의존적.** 재확인(2026-06-18): idx 421470 → 1D infeasible / 2D optimal, 기운 grip idx 235246 → **1D·2D 모두 optimal**. 비대칭/기운 그립은 contact 방향이 다양해 1D로도 풀림. (이전 "1D 거의 항상 infeasible"은 과장이었음.)
- **2D를 기본값으로 쓰는 이유:** 2D의 feasible 집합은 1D의 **상위집합**(1D = 2D + ft2=0 제약). 따라서 2D는 1D가 풀리는 모든 경우 + antipodal 케이스까지 풀어, 멀티 scene에서 푸는 frame 수를 최대화. 계획서도 1D를 limitation으로 명시하고 2D 확장을 해결책으로 제시했음.
- → **2026-06-18: 모든 스크립트 기본 2d로 확정** (`--friction` 기본값 2d, 사용자: "1D 고집할 필요 없다"). 1d=계획서 formulation(|f_t1|≤μf_n), 2d=full cone ||(f_t1,f_t2)||≤μf_n.
- **support case 처리:** 2D는 tangent basis 방향이 해에 무관(등방적 disk)하므로 `generic_tangent(n)`(n에 수직인 임의 직교 basis)을 써서 n∥g인 **support/수직 접점도 제외 없이 포함**한다. min-effort가 위로 향한 normal force로 무게를 받쳐 grip을 줄이는 걸 자동으로 잡음. 1d만 `friction_tangent`(anti-gravity 축)을 써서 support를 제외(genuine singularity). 검증: idx 3752 2d는 윗면 누르는 수직 접점들을 이제 포함하되 min-effort가 fn=0을 줌(아래로 누르는 접점이라 무게에 안 보탬 → 물리적으로 정확).

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

### 문제 #7. edge-graze artifact — nearest-face normal의 90° 스냅 (2026-06-18)

- 손을 곧게 펴 앞/뒷면을 잡았는데 손가락 옆구리가 옸면 모서리에 스치면, 그 vertex들이 "가장 가까운 면 = 옆면"으로 잡혀 옆면 normal(90° 꺾인)을 받음. 실제론 옆면을 누르는 게 아니라 앞면을 누르는 중이라 normal이 잘못됨.
- 판별자: `ang(hand_normal, n_k)`. 진짜 conform이면 hand normal도 같이 꺾여 불일치 작음(~수십°), 우연한 graze면 hand normal은 앞면 그대로라 불일치 ~90°. 검증 scene에서 graze로 의심되는 patch들이 실제로 불일치 57~77°로 튐.
- (부분) 해결: n_k를 contact direction으로 바꿔 edge에서 90° 스냅 대신 blend(문제 #3). 단 진짜로 옆면에 가장 가까운 vertex는 여전히 옆면을 가리키므로, grazing patch가 force 단계에서 spurious force를 받을 가능성은 남음 → 다음 단계 과제, `ang` 지표로 모니터링.
- 일반화 고려로 "면 normal 후보 enumerate 후 hand normal로 면 선택" 방식은 채택 안 함(둥근/복잡한 물체엔 면 목록이 없음). object별 hard cutoff도 피함.

---

## 4. 코드 구성 (annotation/ 폴더)

```
annotation/
├── compute_contact.py        # [Step 1] contact 추출 + cluster(손가락×patch) + 화살표 시각화
├── scan_contact_scenes.py    # [Step 1.5] 전체 scene 스캔/랭킹
├── analyze_normal_spread.py  # [Step 1.6] within-finger normal spread 분포(분할 임계각 결정)
├── render_contact_video.py   # [Step 1.6] 30° 클러스터링+화살표 검증 비디오(turntable)
├── solve_pressure.py         # [Step 2] min-effort SOCP → pressure + force 화살표 시각화
├── render_pressure_video.py  # [Step 2.5] 멀티 scene pressure 예측 비디오(force 화살표)
└── vis/
    ├── contact/            # contact_<idx>.png (화살표 포함 figure), contact_<idx>.npz,
    │                       # scan_results.npz, normal_spread.{png,npz},
    │                       # contact_clusters.mp4 (검증 비디오)
    └── pressure/           # pressure_<idx>_<1d|2d>.png / .npz,
                            # pressure_clusters.mp4 (멀티 scene 비디오)
```

### compute_contact.py — 핵심 함수 (다른 스크립트에서 import해서 재사용)

- **load_hand(sample, label):** MANO mesh(778 verts, 카메라 좌표, m 단위)와 vertex별 손가락 라벨(skinning weight argmax; MANO part 순서는 wrist/index/middle/little/ring/thumb 주의 — 파일 상단 주석 참고).
- **load_object(...):** 잡은 YCB object mesh를 pose_y로 변환해 카메라 좌표로.
- **gravity_in_camera(...):** apriltag extrinsics에서 중력 단위벡터를 카메라 frame으로. (calibration/extrinsics_<id>/extrinsics.yml의 'apriltag' 항목 z축이 테이블 위쪽.)
- **detect_contact(...):** trimesh ProximityQuery로 hand vertex의 signed distance + **closest point** (closest face normal 부호 판정이라 non-watertight mesh OK). contact = sd > -threshold (관통 포함).
- **contact_directions(...):** 각 접촉 vertex의 손→물체 방향(`closest − hand_vertex`, 관통 부호 반전, 퇴화 시 face normal fallback). n_k와 군집화의 기준 벡터.
- **split_by_normal(vectors, angle):** average-linkage(cosine) 군집화로 angle(기본 30°) 초과 시 분리, patch 라벨 반환.
- **cluster_stats(...):** 손가락별로 contact direction을 split_by_normal로 patch 분할(stray <min_verts drop) → patch별 n_k(=contact direction 평균), area_k, n·g, hand normal과의 불일치각(graze 진단), centroid. 한 손가락에서 여러 cluster 가능(label 예: "ring0","ring1").
- **make_arrow / cluster_arrows:** cluster centroid에서 n_k 방향 화살표 mesh(색=cluster 색).
- **render_overlay / render_orbit:** pyrender EGL 렌더, `extra_meshes`로 화살표 추가. orbit은 중력 기준 수직 up으로 4방향. 출력 figure는 3x4 grid (overlay+통계 / orbit / hand-only).
- npz에는 SOCP에 필요한 모든 값 저장 (cluster normals/areas/centroids, patch 라벨, gravity, hand verts, object center).

### scan_contact_scenes.py

- s0_train에서 cracker box를 잡는 시퀀스를 찾아 stride frame마다 평가.
- 빠른 prefilter(라벨만 사용): 박스 수직(|장축·(-g)|>0.95), 들림(frame 0 대비 +3cm), 손 존재 → 통과한 frame만 contact 계산.
- 속도 최적화: hand vertex를 object canonical frame으로 역변환해서 ProximityQuery를 한 번만 빌드. 41개 시퀀스 전체 ~26초.
- 통과 기준: thumb 포함 cluster ≥ 4개. worst |n_obj·g| 오름차순 랭킹 출력, vis/contact/scan_results.npz에 저장.

### analyze_normal_spread.py

- 41개 cracker 시퀀스(stride 4, prefilter 없음 — messy grip 포함)에서 손가락별 접촉 vertex의 contact direction이 손가락 평균에서 벌어진 각도를 수집.
- per-vertex / per-finger MAX / per-finger P90 분포를 히스토그램(vis/contact/normal_spread.png)·percentile·임계별 비율로 출력. multi-patch 의심 cluster top-15도. → 분할 임계각 30° 근거.

### render_contact_video.py

- cracker 시퀀스를 프레임별로 좌(카메라 오버레이)/우(turntable, 화살표 포함) 2-패널로 렌더해 하나의 mp4로 연결(vis/contact/contact_clusters.mp4).
- ColorTracker: 직전 프레임 cluster와 (finger_id, normal 각도<30°) 매칭해 색 id 유지 → 연속 프레임 색 일관. 시퀀스 경계에서 reset.
- FrameRenderer: OffscreenRenderer 1개 재사용. turntable은 90프레임당 1바퀴 회전. caption은 작은 폰트·좌측 정렬·줄당 5개 wrap.

### solve_pressure.py

- compute_contact의 파이프라인을 import해 contact cluster를 다시 계산한 뒤 cvxpy(ECOS)로 min-effort 문제 풀이. torque equilibrium은 아직 미포함.
- **tangent basis:** 2d는 `generic_tangent(n)`(n에 수직인 임의 직교 basis, 항상 정의됨 → support 포함). 1d는 `friction_tangent(n,g)`(t1=anti-gravity projection; n∥g면 None → support 제외).
- `--friction 1d`: f_t2=0, |f_t1| ≤ μf_n (계획서 formulation, QP). `--friction 2d`: ||(f_t1,f_t2)|| ≤ μf_n (full cone, SOCP). **기본 2d** (문제 #4).
- infeasible이면 status 로깅 후 종료(계획서 방침). optimal이면 pressure_k = f_n_k / area_k 계산, 손 mesh에 inferno colormap + **접촉력 F_k 초록 화살표(force_arrows, 길이 ∝ |F_k|)** 시각화.

### render_pressure_video.py

- solve_pressure(SOCP, force_arrows) + render_contact_video(FrameRenderer, compose_frame)를 재사용해 41개 cracker 시퀀스의 pressure 예측을 비디오로(vis/pressure/pressure_clusters.mp4).
- **수직 prefilter 없음(기운 grip 포함)**, 2d는 support 포함(generic_tangent), 사용 가능 contact<2 또는 SOCP infeasible frame은 skip하고 사유별(no_contact/few_usable/infeasible) 카운트.
- 손을 예측 pressure(inferno, **고정 vmax**=`--vmax_kpa` 기본 8kPa, frame 간 비교 가능)로 칠하고 force 화살표 추가. 좌(카메라 오버레이)/우(turntable) 2-패널. `--friction` 기본 2d.

---

## 5. 실행 방법

### 환경

- **conda env: `pressure`** (`/ssd/sjkim/anaconda3/envs/pressure/bin/python`, Python 3.7). 2026-06-18에 `dexycb-toolkit`을 clone해서 만든 pressure research 전용 env (`conda create --clone dexycb-toolkit -n pressure`). 기존 `dexycb-toolkit`도 동일하게 동작하지만 앞으로는 `pressure` 사용. base/dexycb env에는 torch가 없으니 주의.
- 포함 패키지(clone으로 상속): torch 1.13.1+cu117, trimesh 4.4.1, pyrender, rtree(proximity), cvxpy+ECOS(SOCP), scipy(군집화), cv2(비디오), manopth/dex_ycb_toolkit(editable).
- 환경변수는 스크립트가 직접 세팅함: DEX_YCB_DIR=/datasets/dexycb, PYOPENGL_PLATFORM=egl (headless 렌더링).
- MANO 모델: dex-ycb-toolkit/manopth/mano/models (gitignore됨). annotation/ 스크립트는 상대경로 → 절대경로 순으로 fallback (compute_contact.py의 find_mano_root()).

### 명령어 (annotation/ 디렉토리에서 실행; `conda activate pressure`)

```Shell
# 1) contact 추출 + 화살표 시각화 (기본 idx 421470)
python compute_contact.py
python compute_contact.py --idx 3752 --split_angle 30   # over-edge 분할 확인

# 2) scene 스캔 (41개 시퀀스, ~30초)
python scan_contact_scenes.py

# 3) within-finger normal spread 분포(분할 임계각 근거)
python analyze_normal_spread.py

# 4) 30° 클러스터링+화살표 검증 비디오 (전체 cracker 시퀀스, ~30분)
python render_contact_video.py            # --max_seqs N 으로 일부만, --stride 로 간격 조정

# 5) force/pressure 계산 + force 화살표 시각화 (단일 scene)
python solve_pressure.py --idx 421470 --friction 2d
python solve_pressure.py --idx 421470 --friction 1d   # 1D infeasible 확인용(이 scene)

# 6) 멀티 scene pressure 예측 비디오 (전체 cracker, 기운 grip 포함, ~25-35분)
python render_pressure_video.py           # --max_seqs N, --stride, --vmax_kpa 8
```

### 주요 옵션 (공통)

- `--idx` — dataset index (기본 421470. idx↔frame: 같은 시퀀스/카메라에서 연속이므로 idx = (frame 0의 idx) + frame)
- `--thresh` — proximity threshold [m] (기본 0.005)
- `--min_verts` — cluster(및 sub-patch) 최소 vertex 수 (기본 3; 미만 stray는 drop)
- `--split_angle` — within-finger normal-patch 분할 임계각 [deg] (기본 30)
- `--mass/--mu` — solve_pressure / render_pressure_video. 기본 0.411kg / 0.5 (cracker box)
- `--friction {1d,2d}` — solve_pressure / render_pressure_video (기본 2d)
- `--vmax_kpa` — render_pressure_video pressure colormap 상한 [kPa] (기본 8, 초과 clip)
- `--max_seqs / --stride` — 비디오 스크립트: 처리 시퀀스 수 제한 / frame 간격

### 기타 알아두면 좋은 것

- pyrender 0.1.45에서 Scene(bg_color=[1,1,1,1])처럼 int를 주면 255로 나눠져 검정이 됨 → 반드시 float([1.0,...])로.
- 라벨(pose_y/pose_m)은 카메라 좌표계 기준. visualize용 y/z 부호 반전은 pyrender 카메라 convention 때문이며 물리 계산에서는 하지 않음.
- MANO part→손가락 매핑 순서가 직관과 다름: 1-3 index, 4-6 middle, 7-9 little, 10-12 ring, 13-15 thumb.
