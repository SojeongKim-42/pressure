# Research Context — Hand-Object Pressure Pseudo-Labeling

**업데이트:** 2026-06-18

> Research Plan.md의 "검증 계획: Simplest Case" 진행 기록. 이 문서는 나중에 다시 봐도 바로 이어서 작업할 수 있도록 진행 상황 / 발견한 문제 / 코드 구성 / 실행법을 기록한다.

---

## 1. 한 줄 요약

DexYCB cracker box scene(idx 421470)에서 proximity 기반 contact 추출과 min-effort SOCP(force + torque equilibrium)까지 구현·검증 완료. force-only 추정치는 계획서 기대치와 일치(엄지 ~4.4N / 손가락 ~0.8-1.5N)했고, torque equilibrium 추가 시 회전 평형 때문에 grip force가 ~2.3배 증가(Step 3). 단, 계획서의 1D friction formulation은 best-case scene에서도 infeasible임을 확인 → 2D friction(full cone)으로 해결. 모든 코드는 `annotation/` 폴더에 있음 (dex-ycb-toolkit 내부에서 작업하지 않기로 함).

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

### [Step 3] torque equilibrium 추가 (2026-06-23, 완료)

force-only SOCP에 torque equilibrium `Σ_k r_k × F_k = 0`을 추가했다. 계획서의 full formulation 완성.

- **수식:** r_k = (cluster centroid − object COM). **물체 COM 기준으로 모멘트를 잡으면 중력(COM에 작용)의 모멘트 팔이 0**이라 torque 식에 중력 항이 안 들어간다(계획서 formulation과 동일). cross product `r_k × F_k`는 force 스칼라(fn, ft1, ft2)에 대해 선형이므로 SOCP가 그대로 유지됨 — `Σ_k fn_k(r_k×n_k) + ft1_k(r_k×t1_k) + ft2_k(r_k×t2_k) = 0`을 3×K 행렬 3개로 구성(force balance 구조와 동형).
- **COM (solve_pressure.object_com):** watertight면 trimesh `center_mass`(균일밀도 부피 중심), 아니면 vertex mean fallback. YCB cracker mesh는 non-watertight라 vertex mean 사용(박스라 기하 중심 ≈ COM, 합리적 근사). torque는 COM 정확도에 민감하다는 점이 limitation.
- **구현:** `solve_min_effort(..., arms, torque)`, `solve_pressure.py`/`render_pressure_video.py` 모두 `--no_torque`로 force-only 비교 가능(기본 torque ON). npz에 object_com/arms/torque_residual 저장, 출력 파일에 `_torque` 태그.
- **★핵심 결과 (idx 421470, m=0.411kg, μ=0.5, 2D):** torque를 켜면 **grip force가 크게 증가**(total f_n 8.97N → 20.3N, ≈2.3배; 엄지 3.99N → 8.84N). force-only min-effort는 회전 평형을 무시해 systematic under-grip을 하는데, torque를 강제하면 net moment를 상쇄하려 더 큰 대향력이 필요하고 하중이 재분배됨(index1·thumb·little1이 받고 index0·ring0는 ~0). **계획서 Known Limitations의 "systematic underestimation"에 대해 torque가 grip force 하한을 끌어올린다**는 의미. 두 모드 모두 optimal, force residual ~1e-10, torque residual ~1e-11(ON)/~0.3 N·m(OFF, 미구속).
- **멀티 scene infeasibility (6 seq, stride 6, 동일 candidate 41 frame[≥2 contact] 기준):** torque ON에서 infeasible이 늘어남 — force-only 12/41(29%) → torque 15/41(37%), +3 frame(+8pp). torque 제약이 feasible set을 좁히는 건 예상된 결과(force-only의 부분집합). 정량 측정은 전체 cracker로 확장 필요(다음 단계).

### [Step 4] 다른 object 확장 + 분할 임계각 일반화 (2026-06-28, 진행 중)

cracker box(평면) 외 곡면 물체로 확장 시작. 첫 대상은 **can(002_master_chef_can, ycb_id=1)**, μ=1.11(aluminium-skin, Derler/문헌 1.11±0.48에서 채택), mass는 추후 YCB 표값. box→bowl 중간 난이도라 can을 먼저(강체라 COM·torque 깨끗, 곡면 clustering 문제를 단순한 형태로 선행; bowl은 속이 비어 COM이 기하중심이 아니라 torque 보정 필요).

- **scan/analyze 일반화:** `scan_contact_scenes.py`·`analyze_normal_spread.py`에 `--ycb_id` 추가(2/cracker, 1/can, 13/bowl). scan의 box-전용 "vertical(장축∥g)" prefilter는 `--vertical_min 0`으로 끌 수 있게(곡면용); lift/정지 filter는 generic. 출력은 object명으로 태깅(cracker 결과 안 덮음).
- **★can spread 측정 결과 — box의 bimodal 구조가 재현 안 됨:** can 526 frame/2077 finger-cluster에서 within-finger contact-normal spread는 per-vertex median **13.3°**(0° floor 없음), per-finger P90 median 23°. 즉 평면처럼 "단일면=spread 0"인 noise floor가 없고 곡률 때문에 연속적으로 퍼짐 → cracker의 bimodal 골짜기(~30°)가 **없음**. 30°는 can에서 주 봉우리의 어깨에 걸쳐 임의 과분할을 유발.
- **★split-threshold sweep (sweep_split_angle.py, 9개 물체: box 3/can 3/bowl·mug·bottle):** T를 10~90° sweep하며 ① 손가락당 patch 수 ② patch 공간지름[cm] ③ within-patch P90 spread ④ resultant R 측정. **T=20°에서 patch 공간지름이 모든 물체에서 ~1.5-2.2cm(손가락 패드 한 개)로 수렴**하고 within-patch R≈0.99·P90 spread<10°. T<15°는 패드를 쪼개는 과분할(area artifact 시작), T>40°는 패드를 넘겨 wrap을 한 덩어리로 묶음. box는 평면이라 T에 둔감(cracker 20°/30° 사실상 동일). R 단독은 너무 관대(~55-60°까지 0.95↑)라 binding이 아니고, **진짜 기준은 patch 공간크기≈패드(~2cm)**. → **`_SPLIT_ANGLE_DEG`를 30°→20°로 전 물체 통일**(compute_contact.py; contact/pressure 전 파이프라인에 전파). 단 20°는 YCB 곡률대(반경 3-7cm)에서 2cm에 정렬된 값이라, 곡률이 크게 다른 물체엔 "지름 2cm 초과 시 추가 분할" 같은 공간크기 가드가 필요(미구현).
- 산출물: `vis/contact/normal_spread_<obj>.{png,npz}`, `vis/contact/split_sweep.{png,npz}`.

### [다음 단계] (미착수)

- can으로 contact/pressure 계산 + normal 저장 실행(다음 작업). 이후 bowl(ycb_id=13, COM 보정 필요).
- cracker box 전체에 대해 SOCP infeasibility/support-제외 rate를 정량 측정(현재 비디오는 skip만 카운트). torque ON/OFF infeasibility 차이도 포함.
- contact area 정의(threshold)에 따른 pressure 스케일 보정 검토 (문제 #5).
- edge-graze patch(문제 #7)를 force 단계에서 어떻게 다룰지 (현재는 분리만; grazing contact의 spurious force 가능성). `ang(hand,obj)` 진단 지표로 모니터링 중.
- 곡률 불변 일반화: 각도 대신 patch 공간크기(~2cm 패드) 기준 split 가드, 공간 연속성 우선 grouping (Step 4).

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
├── scan_contact_scenes.py    # [Step 1.5/4] 전체 scene 스캔/랭킹 (--ycb_id로 임의 물체)
├── analyze_normal_spread.py  # [Step 1.6/4] within-finger normal spread 분포 (--ycb_id)
├── sweep_split_angle.py      # [Step 4] 여러 물체 split 임계각 sweep (patch 크기/spread/R)
├── render_contact_video.py   # [Step 1.6] 20° 클러스터링+화살표 검증 비디오(turntable)
├── solve_pressure.py         # [Step 2/3] min-effort SOCP(+torque) → pressure + force 화살표 시각화
├── render_pressure_video.py  # [Step 2.5/3] 멀티 scene pressure 예측 비디오(+torque, force 화살표)
└── vis/
    ├── contact/            # contact_<idx>.png (화살표 포함 figure), contact_<idx>.npz,
    │                       # scan_results.npz, normal_spread.{png,npz},
    │                       # contact_clusters.mp4 (검증 비디오)
    └── pressure/           # pressure_<idx>_<1d|2d>[_torque].png / .npz,
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

- compute_contact의 파이프라인을 import해 contact cluster를 다시 계산한 뒤 cvxpy(ECOS)로 min-effort 문제 풀이. force + torque equilibrium 모두 포함(`--no_torque`로 force-only; Step 3). torque는 COM 기준 `Σ r_k×F_k=0`, COM은 `object_com()`.
- **tangent basis:** 2d는 `generic_tangent(n)`(n에 수직인 임의 직교 basis, 항상 정의됨 → support 포함). 1d는 `friction_tangent(n,g)`(t1=anti-gravity projection; n∥g면 None → support 제외).
- `--friction 1d`: f_t2=0, |f_t1| ≤ μf_n (계획서 formulation, QP). `--friction 2d`: ||(f_t1,f_t2)|| ≤ μf_n (full cone, SOCP). **기본 2d** (문제 #4).
- infeasible이면 status 로깅 후 종료(계획서 방침). optimal이면 pressure_k = f_n_k / area_k 계산, 손 mesh에 inferno colormap + **접촉력 F_k 초록 화살표(force_arrows)** 시각화.
- **화살표 길이 정규화 (2026-06-27):** 예전엔 `길이 = |F_k| × 0.022 m/N`(무한대) — torque로 grip force가 ~2.3배 커지고 force tail이 무게의 최대 ~40배(p100 162N)라 화살표가 0.2~3.5m로 화면을 벗어났음. 이제 vmax와 같은 방식의 **고정 기준 + clip**: `길이 = min(|F_k|/_FORCE_ARROW_REF_N, 1) × _FORCE_ARROW_MAXLEN`(ref=10N≈cracker 무게 4N의 2.5배 → 6cm). frame마다 바꾸지 않으므로 frame/scene 간 화살표 크기 비교 가능. 분포 측정(torque ON, 152 frame/1166 contact): per-contact |F| p50 1.2 / p90 9.3 / **p100 162N**, per-frame-max p50 8.4 / p90 21.7N → ref 10N이면 보통 엄지 ~5cm, 손가락 ~0.7cm, 162N artifact는 6cm로 clip. force_arrows는 solve_pressure/render_pressure_video가 공유.
- **pressure colormap 정규화 (2026-06-23):** 예전엔 per-scene `pressure_v.max()`로 정규화해 scene마다 색 스케일이 달랐음. 이제 **고정 ceiling `_DEFAULT_VMAX_KPA`(=30kPa, `--vmax_kpa`)** 로 통일 — `[0,vmax]`로 clip 후 나눔(OpenTouch `build_demo.py`/`load_data.py`가 고정 `max_value=3072`로 tactile을 정규화하는 방식과 동일). solve_pressure와 render_pressure_video가 같은 상수를 import해 단일 scene/멀티 scene/frame 간 색이 모두 비교 가능.
- **vmax=30 근거 (분포 측정, 41 cracker scene · stride 10 · torque ON · 152 frame/1166 patch):** per-patch pressure는 heavy-tail(p50 1.8 / p75 8.8 / p90 23.4 / p95 39.7 / p99 169 / **p100 1998 kPa**), per-frame max는 p50 21.5 / p95 179. 극단 tail(~2MPa)은 신호가 아니라 **area artifact**(작은 patch → `fn/area` 폭발, 문제 #5)다. vmax=30이면 patch의 7.5%만 clip되고 신호 대부분(median~p90, 1.8~23kPa)이 색 gradient 안에 들어옴. 40 이상은 clip을 2.5pp밖에 못 줄이면서 typical patch(median 1.8kPa)를 거의 검정으로 밀어 low-end 대비를 잃음 → 30이 sweet spot. per-frame-max 백분위(100~500kPa)를 따라 vmax를 키우면 안 됨(소수 outlier 때문에 실제 contact가 다 검정이 됨; OpenTouch가 3072 위를 clip하는 것과 같은 취지). **vmax는 시각화 선택일 뿐, heavy tail 자체는 area convention/stray-patch 필터링으로 따로 잡아야 함.**

### render_pressure_video.py

- solve_pressure(SOCP, force_arrows) + render_contact_video(FrameRenderer, compose_frame)를 재사용해 41개 cracker 시퀀스의 pressure 예측을 비디오로(vis/pressure/pressure_clusters.mp4).
- **수직 prefilter 없음(기운 grip 포함)**, 2d는 support 포함(generic_tangent), 사용 가능 contact<2 또는 SOCP infeasible frame은 skip하고 사유별(no_contact/few_usable/infeasible) 카운트.
- 손을 예측 pressure(inferno, **고정 vmax**=`--vmax_kpa`, 기본 `_DEFAULT_VMAX_KPA`=30kPa를 solve_pressure와 공유 → scene/frame 간 비교 가능, OpenTouch 방식; 30 근거는 solve_pressure.py 절 참고)로 칠하고 force 화살표 추가. 좌(카메라 오버레이)/우(turntable) 2-패널. `--friction` 기본 2d, torque equilibrium 기본 ON(`--no_torque`로 force-only; Step 3).

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
python solve_pressure.py --idx 421470 --friction 2d              # force + torque (기본)
python solve_pressure.py --idx 421470 --friction 2d --no_torque  # force-only 비교 (Step 2)
python solve_pressure.py --idx 421470 --friction 1d   # 1D infeasible 확인용(이 scene)

# 6) 멀티 scene pressure 예측 비디오 (전체 cracker, 기운 grip 포함, ~25-35분)
python render_pressure_video.py           # --max_seqs N, --stride, --vmax_kpa 8, --no_torque
```

### 주요 옵션 (공통)

- `--idx` — dataset index (기본 421470. idx↔frame: 같은 시퀀스/카메라에서 연속이므로 idx = (frame 0의 idx) + frame)
- `--thresh` — proximity threshold [m] (기본 0.005)
- `--min_verts` — cluster(및 sub-patch) 최소 vertex 수 (기본 3; 미만 stray는 drop)
- `--split_angle` — within-finger normal-patch 분할 임계각 [deg] (기본 20; Step 4 sweep 근거)
- `--mass/--mu` — solve_pressure / render_pressure_video. 기본 0.411kg / 0.5 (cracker box)
- `--friction {1d,2d}` — solve_pressure / render_pressure_video (기본 2d)
- `--vmax_kpa` — render_pressure_video pressure colormap 상한 [kPa] (기본 8, 초과 clip)
- `--max_seqs / --stride` — 비디오 스크립트: 처리 시퀀스 수 제한 / frame 간격

### 기타 알아두면 좋은 것

- pyrender 0.1.45에서 Scene(bg_color=[1,1,1,1])처럼 int를 주면 255로 나눠져 검정이 됨 → 반드시 float([1.0,...])로.
- 라벨(pose_y/pose_m)은 카메라 좌표계 기준. visualize용 y/z 부호 반전은 pyrender 카메라 convention 때문이며 물리 계산에서는 하지 않음.
- MANO part→손가락 매핑 순서가 직관과 다름: 1-3 index, 4-6 middle, 7-9 little, 10-12 ring, 13-15 thumb.
