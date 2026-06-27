conda activate pressure
cd /home/sjkim/Research/pressure/annotation
OBJECTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)
OBJ_NAMES=(master_chef_can
            cracker_box
            sugar_box
            tomato_soup_can
            mustard_bottle 
            tuna_fish_can
            pudding_box
            gelatin_box
            potted_meat_can
            banana 
            pitcher_base
            bleach_cleanser
            bowl
            mug
            power_drill
            wood_block
            scissors
            large_marker
            large_clamp
            extra_large_clamp foam_brick)
# # --------------- 일부 scene만 확인하고 싶을 때 ---------------
# # 표면이 평평한 물체는 box-vertical filter를 켜야 함 
# # 1) can scene 스캔 (곡면이라 box-vertical filter 끔) → 후보 랭킹 + scan_results_002_master_chef_can.npz
# python scan_contact_scenes.py --ycb_id 1 --vertical_min 0
# #    출력 상위에서 정적으로 들고 있는 frame의 idx 하나 고르기

# # 2) contact 추출 + normal/area/centroid 저장 (→ vis/contact/contact_<idx>.npz, .png)
# python compute_contact.py --idx <CAN_IDX>

# # 3) pressure (can: μ=1.11, mass=0.414kg) → vis/pressure/pressure_<idx>_2d_torque.{png,npz}
# python solve_pressure.py --idx <CAN_IDX> --mass 0.414 --mu 1.11 --friction 2d
# #    force-only 비교: 위에 --no_torque 추가



# ------------- pressure rendering for all --------------
# ycb_id: 1~21 (1: master can, 2: cracker box), mass: kg, mu: friction coefficient

for ycb_id in OBJECTS; do
    # if box: vertical_min=1, if not: vertical_min=0
    obj_name=${OBJ_NAMES[$((ycb_id-1))]}
    if [[ 'box' in $obj_name || 'block' in $obj_name ]]; then
        vertical_min=1
    else
        vertical_min=0
    fi
    python render_pressure_video.py --ycb_id $ycb_id --mass 0.414 --mu 1.11
done
# python render_pressure_video.py --ycb_id 1 --mass 0.414 --mu 1.11
