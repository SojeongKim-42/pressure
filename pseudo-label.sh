#!/bin/bash
cd /home/sjkim/Research/pressure/annotation

# 물체별 mass(kg) / mu(마찰계수)는 annotation/ycb_object_params.json 에 정의돼 있음.
# precompute_pressure.py 가 자동으로 읽어 온다 (스크립트에서 따로 넘길 필요 없음).
# ycb_id: 1~21 (1: master_chef_can, 2: cracker_box, 13: bowl, ...)
OBJECTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)

# RENDER=1 로 두면 저장된 값으로 검수용 비디오까지 만든다(느림). 0=값만 저장.
# ex) RENDER=1 bash pseudo-label.sh
RENDER=${RENDER:-0}

# # --------------- 일부 scene만 손으로 확인하고 싶을 때 ---------------
# # 표면이 평평한 물체는 box-vertical filter(--vertical_min 1)를 켜야 함
# # 1) can scene 스캔 (곡면이라 filter 끔) → 후보 랭킹 + scan_results_002_master_chef_can.npz
# python scan_contact_scenes.py --ycb_id 1 --vertical_min 0
# #    출력 상위에서 정적으로 들고 있는 frame의 idx 하나 고르기
#
# # 2) contact 추출 + normal/area/centroid 저장 (→ vis/contact/contact_<idx>.npz, .png)
# python compute_contact.py --idx <CAN_IDX>
#
# # 3) 한 프레임 pressure 시각화 (can: μ=1.11, mass=0.414kg)
# #    → vis/pressure/pressure_<idx>_2d_torque.{png,npz}  (force-only 비교: --no_torque)
# python solve_pressure.py --idx <CAN_IDX> --mass 0.414 --mu 1.11 --friction 2d


# ============ 1) pseudo-label 생성: contact + pressure 값만 저장 (렌더 없음) ============
# 각 물체의 모든 grasp frame에 대해 contact → cluster → min-effort SOCP를 한 번 풀어
# pressure_labels/<model>.npz 로 값만 저장한다 (mass/mu는 json에서 자동).
# 한 번에 전체로 돌리려면:  python precompute_pressure.py --ycb_id 0
# 아래는 물체별 loop(한 물체가 죽어도 나머지는 계속)로 동일하게 처리.
for ycb_id in "${OBJECTS[@]}"; do
    python precompute_pressure.py --ycb_id "$ycb_id"
done

# ============ 2) (선택) 저장값으로 빠른 렌더 ============
# precompute가 저장한 npz를 읽어 ProximityQuery/clustering/SOCP 재계산 없이 비디오 생성.
# 물체/물리 파라미터(ycb_id, mass, mu, friction)는 npz에서 자동으로 가져온다.
if [[ "$RENDER" == "1" ]]; then
    for npz in pressure_labels/*.npz; do
        python render_pressure_video.py --from_data "$npz"
    done
fi
