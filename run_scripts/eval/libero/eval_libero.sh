#!/bin/bash
# Local (non-slurm) LIBERO evaluation runner.
# Usage: eval_libero.sh <CKPT_NAME> <MODEL_PATH> [GPU_ID] [MAX_PARALLEL]
set -u

export NO_ALBUMENTATIONS_UPDATE=1

CKPT_NAME="${1:?CKPT_NAME required}"
MODEL_PATH="${2:?MODEL_PATH required}"
GPU_ID="${3:-0}"
MAX_PARALLEL="${4:-4}"

BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
LIBERO_PY="$BASE_DIR/rldx/eval/sim/LIBERO/libero_uv/.venv/bin/python"

find_free_port() {
  local port=$1
  while ss -lnt | awk '{print $4}' | grep -q ":$port$"; do
    port=$((port + 1))
    [ "$port" -gt 65000 ] && port=20000
  done
  echo "$port"
}
PORT=$(find_free_port $((20000 + RANDOM % 40000)))

echo "[i] CKPT_NAME=$CKPT_NAME"
echo "[i] MODEL_PATH=$MODEL_PATH"
echo "[i] GPU_ID=$GPU_ID  MAX_PARALLEL=$MAX_PARALLEL  PORT=$PORT"

cd "$BASE_DIR"
CUDA_VISIBLE_DEVICES=$GPU_ID uv run python rldx/eval/run_rldx_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag GENERAL_EMBODIMENT \
    --use-sim-policy-wrapper \
    --no-strict \
    --host 127.0.0.1 \
    --port "$PORT" &
SERVE_PID=$!
trap 'echo "[i] killing server PID=$SERVE_PID"; kill $SERVE_PID 2>/dev/null' EXIT

echo "[i] Waiting for server (PID=$SERVE_PID) readiness..."
for i in $(seq 1 120); do
  if ss -lnt | awk '{print $4}' | grep -q ":$PORT$"; then
    echo "[i] Server listening on :$PORT after ${i}s"
    break
  fi
  if ! kill -0 $SERVE_PID 2>/dev/null; then
    echo "[!] Server died before binding to :$PORT"; exit 1
  fi
  sleep 1
done
sleep 5  # give server a moment past port-open

LIBERO_10_TASKS=(
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
    "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
    "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
    "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
    "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
    "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
    "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)
LIBERO_GOAL_TASKS=(
    "libero_sim/open_the_middle_drawer_of_the_cabinet"
    "libero_sim/put_the_bowl_on_the_stove"
    "libero_sim/put_the_wine_bottle_on_top_of_the_cabinet"
    "libero_sim/open_the_top_drawer_and_put_the_bowl_inside"
    "libero_sim/put_the_bowl_on_top_of_the_cabinet"
    "libero_sim/push_the_plate_to_the_front_of_the_stove"
    "libero_sim/put_the_cream_cheese_in_the_bowl"
    "libero_sim/turn_on_the_stove"
    "libero_sim/put_the_bowl_on_the_plate"
    "libero_sim/put_the_wine_bottle_on_the_rack"
)
LIBERO_OBJECT_TASKS=(
    "libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_butter_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_milk_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
    "libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket"
)
LIBERO_SPATIAL_TASKS=(
    "libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
    "libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
)

ALL_TASKS=() ALL_SUITES=() ALL_TASK_INDICES=()
for i in "${!LIBERO_10_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_10_TASKS[$i]}"); ALL_SUITES+=("libero_10"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_GOAL_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_GOAL_TASKS[$i]}"); ALL_SUITES+=("libero_goal"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_OBJECT_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_OBJECT_TASKS[$i]}"); ALL_SUITES+=("libero_object"); ALL_TASK_INDICES+=($i)
done
for i in "${!LIBERO_SPATIAL_TASKS[@]}"; do
    ALL_TASKS+=("${LIBERO_SPATIAL_TASKS[$i]}"); ALL_SUITES+=("libero_spatial"); ALL_TASK_INDICES+=($i)
done

TOTAL=${#ALL_TASKS[@]}
echo "[i] Total tasks: $TOTAL, batching $MAX_PARALLEL at a time"

RUN_PIDS=()
for idx in "${!ALL_TASKS[@]}"; do
    TASK="${ALL_TASKS[$idx]}"
    SUITE="${ALL_SUITES[$idx]}"
    TIDX="${ALL_TASK_INDICES[$idx]}"
    CLEAN="${TASK#libero_sim/}"
    OUT="$BASE_DIR/output_final/libero/$CKPT_NAME/$SUITE/$CLEAN"
    mkdir -p "$OUT"
    if [ "$SUITE" == "libero_10" ]; then N_EP=50; else N_EP=20; fi
    echo "[i] [$((idx+1))/$TOTAL] $SUITE :: $CLEAN (n_ep=$N_EP)"
    "$LIBERO_PY" "$BASE_DIR/rldx/eval/rollout_policy.py" \
        --n_episodes $N_EP \
        --policy_client_host 127.0.0.1 \
        --policy_client_port "$PORT" \
        --max_episode_steps 720 \
        --env_name "$TASK" \
        --n_action_steps 8 \
        --n_envs 5 \
        --video_dir "$OUT" \
        >& "$OUT/eval-local-$TIDX.log" &
    RUN_PIDS+=($!)
    # throttle to MAX_PARALLEL in flight
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 5
    done
done

echo "[i] All tasks launched, waiting for completion..."
for pid in "${RUN_PIDS[@]}"; do wait "$pid"; done

echo "[i] ===== Summary ====="
for idx in "${!ALL_TASKS[@]}"; do
    SUITE="${ALL_SUITES[$idx]}"
    CLEAN="${ALL_TASKS[$idx]#libero_sim/}"
    TIDX="${ALL_TASK_INDICES[$idx]}"
    LOG="$BASE_DIR/output_final/libero/$CKPT_NAME/$SUITE/$CLEAN/eval-local-$TIDX.log"
    SR=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOG" 2>/dev/null | tail -1 || echo "N/A")
    echo "[i] $SUITE :: $CLEAN -> $SR"
done
