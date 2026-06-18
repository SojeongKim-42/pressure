# Hand-Object Pressure Pseudo-Labeling: Research Plan

## 연구 목표

RGB 기반 HOI 영상에 physics-based pseudo pressure label (N/m²)을 자동으로 생성하는 pipeline을 구축하는 것이 목표다. 별도의 pressure sensor 없이 물체의 mass, friction coefficient, 손과 물체의 mesh 정보만을 이용해 각 contact point에서의 pressure를 추정한다. 이렇게 생성된 pseudo-label은 대규모 egocentric HOI dataset에 적용 가능한 weakly supervised training signal로 활용한다.

## 현재 Scope

Dataset은 hand/object mesh annotation의 정확도를 기준으로 DexYCB를 사용한다. 현재는 static holding만 가정하여 가속도 a ≈ 0으로 둔다.

> **구현 결정 (2026-06-18):** Friction은 **2D(full cone)를 기본**으로 채택한다. 당초 계획한 1D friction(t_k = 중력 반대방향 projection)은 (a) 깔끔한 antipodal 그립에서 force equilibrium조차 infeasible해지고, (b) n_k ∥ gravity일 때 t_k가 undefined가 되어 **support case를 배제해야 했다**. 2D에서는 마찰력 집합이 등방적 disk라 tangent basis 방향이 해에 무관하므로, n_k에 수직인 임의 직교 basis를 쓰면 **support case(손이 물체를 일부 떠받쳐 grip force가 줄어드는 경우)도 그대로 풀린다** — min-effort 해가 위로 향한 normal force로 무게를 받쳐 squeeze를 줄이는 것을 자동으로 잡아낸다. 따라서 support case는 더 이상 scope에서 제외하지 않는다 (1D 모드를 명시적으로 쓸 때만 제외).

## 물리 Formulation

각 contact point k에서 손이 물체에 가하는 force는 다음과 같이 분해된다.

> F_k = f_normal_k · n_k + f_friction_k · t_k

n_k는 손에서 물체 방향의 inward surface normal이고, t_k는 중력 반대 방향 벡터를 n_k가 이루는 tangential plane에 projection한 단위 벡터다. Static holding 조건에서 force equilibrium은 Σ_k F_k = -mg_vec이고, friction cone constraint는 |f_friction_k| ≤ μ · f_normal_k, f_normal_k ≥ 0이다.

> **구현 채택형 (2026-06-18):** 위는 1D 버전이고, 실제로는 **2D friction**을 쓴다 — 마찰을 tangent plane의 두 직교 성분 (t1, t2)로 두고 F_k = f_n·n_k + f_t1·t1 + f_t2·t2, cone constraint ‖(f_t1, f_t2)‖ ≤ μ·f_normal_k. (t1, t2)는 n_k에 수직인 **임의의** 직교 basis면 되고(해에 무관, 등방적 disk), n_k는 contact direction(손 vertex→최근접 표면점 평균; Research context.md 문제 #3)으로 정의한다.

이 문제는 contact point 수 K가 충분히 크면 (일반적으로 K ≥ 4) 방정식 수 (force equilibrium 3개, torque equilibrium 3개)보다 미지수 수 (2K개)가 많아 underdetermined system이 된다. Unique solution을 얻기 위해 minimum L2 norm 기준의 SOCP를 사용한다.

```
minimize:   Σ_k ||F_k||²
subject to: Σ_k F_k = -mg_vec
            Σ_k (r_k × F_k) = 0
            f_normal_k ≥ 0
            |f_friction_k| ≤ μ · f_normal_k
```

이는 "물리적으로 feasible한 해 중 최소 effort 분배"에 해당하며, robotics 문헌에서 minimum effort grasp로 알려진 접근이다. Feasible solution이 존재할 때 strictly convex objective가 unique solution을 보장하지만, contact geometry가 나쁘거나 μ가 너무 작은 경우 infeasibility가 발생할 수 있으며 이 케이스는 logging하고 제외한다.

각 contact point의 pressure는 다음과 같이 계산된다.

> pressure_k = f_normal_k / area_k    (N/m²)

## Friction Coefficient

마찰계수는 Derler & Gerhardt (2012)와 Seo et al. 기반으로 설정한다. 일반 dry skin의 경우 μ ≈ 0.5를 기본값으로 사용하고, 검증 대상인 cracker box (coated/printed cardboard)의 경우 raw cardboard 값 0.47 ± 0.17보다 코팅 처리로 인해 약간 높을 것으로 판단하여 μ = 0.5를 사용한다. 이 값은 문헌 기반 추정이며 직접 측정값이 아님을 명시한다.

## Known Limitations

Minimum L2 norm은 F_rnn = 0으로 두는 것과 수학적으로 동치다. 즉 "실제 인간의 grasp behavior prior"가 전혀 없는 formulation이다. 실제 인간은 필요한 force보다 더 크게 쥐는 경향이 있으므로 이 방법은 systematic underestimation에 가까울 수 있다. 이를 임의의 scaling factor로 수정하는 것은 근거가 없으므로, limitation으로 명시하고 이후 OPENTOUCH validation에서 실제 distribution과 비교하여 data-driven correction을 검토한다. 또한 1D friction 가정은 lateral friction을 표현하지 못해 일부 grasp에서 infeasibility를 유발했고(특히 깔끔한 antipodal 그립), **2026-06-18부터 2D friction을 기본으로 채택**해 이를 해결했다 (위 Formulation 노트 참고). 남은 limitation: torque equilibrium 미포함, contact area 정의에 따른 pressure 스케일 민감도, edge-graze artifact (Research context.md 문제 #5/#7).

## 검증 계획: Simplest Case

먼저 아래 조건의 가장 단순한 케이스로 pipeline 전체를 구현하고 시각적으로 검증한다.

DexYCB에서 cracker box (003_cracker_box, m = 0.411 kg) scene을 선택하여 simple static holding, fingertips only 상황을 사용한다.

→ 검증할 scene은 손이 cracker box를 거의 수직으로 집어 올리는 것으로 선택한다.

Proximity threshold로 hand mesh와 object mesh 간 contact vertices를 추출하고, **손가락 단위로 1차 grouping한 뒤 contact normal 방향으로 다시 patch 단위로 분할**하여 각 cluster(= 손가락 × patch)의 대표 normal n_k와 contact area_k를 계산한다. 한 손가락이 서로 다른 면에 동시에 닿으면(모서리 감아쥠 등) 별도 cluster로 나뉘므로 손가락당 cluster가 여러 개일 수 있다.

> **구현 결정 (2026-06-18, Research context.md 문제 #3/#7):** n_k는 hand mesh normal(noisy)도 nearest-face normal(edge에서 90° 스냅 artifact)도 아닌, **각 접촉 vertex의 contact direction**(`closest_point − hand_vertex`, 손→물체 방향)을 cluster 평균낸 값으로 정의한다. 물리적 force 방향이고, 평면에선 inward normal과 같으며, 순수 기하라 noise가 적고 임의 물체에 일반화된다. 분할 임계각은 cracker box scene들의 within-finger normal spread 분포(bimodal, 골짜기 ~30°)로 30°로 정했고, 다른 object에선 재측정한다.

n_k의 z 성분이 거의 0인지 (n_k ⊥ gravity) 확인하여 현재 simplification이 성립하는 scene인지 검증한다. 이후 torque equilibrium을 먼저 제외한 상태에서 SOCP를 풀어 f_normal_k를 구하고, pressure_k = f_normal_k / area_k를 계산한 뒤 hand mesh 위에 color map으로 시각화한다.

기대 결과는 cracker box (411g, μ = 0.5) 기준 엄지 ~4N / 나머지 손가락 ~1.3N, pressure ~40 kPa / ~13 kPa 수준이다.

## 이후 단계

단순 케이스 검증 후 torque equilibrium을 다시 포함하고, DexYCB 전체 cracker box scene으로 확장하여 SOCP infeasibility rate를 측정한다. 이후 다른 object (다양한 material, mass)로 확장하고, 최종적으로 ContactPose에서 physics pseudo-label 품질을 확인한 뒤 vision-only 모델 학습 → OPENTOUCH validation 순서로 진행한다.
