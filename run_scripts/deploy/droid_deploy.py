import os
import re
import csv
import time
import tqdm
import tyro
import enum
import signal
import contextlib
import dataclasses
import datetime
import faulthandler
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from moviepy.editor import ImageSequenceClip
from PIL import Image as PILImage

import torch
from droid.robot_env import RobotEnv

from rldx.policy.rldx_policy import RLDXPolicy
from rldx.data.embodiment_tags import EmbodimentTag

faulthandler.enable()
DROID_CONTROL_FREQUENCY = 10
DUMMY_IMAGE = np.zeros((720, 1280, 3), dtype=np.uint8)


class EnvMode(enum.Enum):
    GROOT = "groot"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str
    seed: int = 7


@dataclasses.dataclass
class Args:
    left_camera_id: str = "34022131"
    wrist_camera_id: str = "10623639"
    max_timesteps: int = 700
    open_loop_horizon: int = 16

    env: EnvMode = EnvMode.GROOT

    default_prompt: str | None = None

    policy: Checkpoint = dataclasses.field(
        default_factory=lambda: Checkpoint(
            config="rldx",
            dir="/YOUR_CHECKPOINT_HERE",
        )
    )

    logging_path: str = "/tmp"

    embodiment_tag: EmbodimentTag = EmbodimentTag.GENERAL_EMBODIMENT
    denoising_steps: int = 4

    instruction: str | None = None
    binarize_gripper: bool = False

    num_steps_wait: int = 10
    num_frames: int = 4   # T: temporal horizon for video (overridden at runtime from policy config)
    video_stride: int = 2  # stride between sampled frames (e.g., stride=2 -> [-6,-4,-2,0])

    video_key_exterior: str = "34022131_left"
    video_key_wrist: str = "10623639_left"

    language_key: str = "task"

    resize_video: str | None = None  # e.g., "168,336" -> resize each input frame to (H, W) before policy inference
    deactivate_memory: bool = False  # force-disable memory even if checkpoint was trained with use_memory=True

    state_space: str = "eef"  # eef or joint
    action_space: str = "eef"  # eef or joint

    # WowSkin tactile sensor USB ports
    wowskin_port_right: str = "/dev/ttyACM0"
    wowskin_port_left: str = "/dev/ttyACM1"


def create_policy(args: Args) -> RLDXPolicy:
    policy = RLDXPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.policy.dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        deactivate_memory=args.deactivate_memory,
        require_physics=True,
    )
    return policy


def get_success_rate() -> float:
    success: str | float | None = None
    while not isinstance(success, float):
        success = input("Did the rollout succeed? (y=1.0, n=0.0, h=0.5): ")
        if success == "y":
            success = 1.0
        elif success == "n":
            success = 0.0
        elif success == "h":
            success = 0.5
        else:
            print(f"Invalid input: {success}")
            continue
        if not (0 <= success <= 1):
            print(f"Success must be in [0, 1] but got: {success}")
            success = None
    return success


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def build_observation(
    args: Args,
    image_queue: deque,
    wrist_image_queue: deque | None,
    curr_obs: dict,
    instruction: str,
    *,
    resize_hw: tuple[int, int] | None = None,
    video_model_keys: list[str] | None = None,
    state_keys_map: dict[str, str] | None = None,
    physics_queues: dict[str, deque] | None = None,
    physics_num_frames: int = 0,
) -> dict:
    """
    Build nested observation dict for RLDXPolicy.

    RLDXPolicy expects:
        {
            "video":    {model_key: np.ndarray(B, T, H, W, C), dtype=uint8},
            "state":    {model_key: np.ndarray(B, T, D),       dtype=float32},
            "language": {key: [[instruction]]},
            "physics":  {sub_key: np.ndarray(B, T, D), dtype=float32},  # optional
        }
    """
    # --- video ---
    def _sample_frames(queue: deque, num_frames: int, stride: int) -> np.ndarray:
        q = list(queue)
        selected = [q[-(1 + i * stride)] for i in range(num_frames - 1, -1, -1)]
        return np.stack(selected)  # (T, H, W, C)

    def _resize_frames(frames: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        h, w = hw
        return np.stack([
            np.array(PILImage.fromarray(f).resize((w, h), PILImage.BILINEAR))
            for f in frames
        ])  # (T, H, W, C)

    video = {}
    if video_model_keys is None:
        video_model_keys = ["primary", "wrist"]

    # Map model video keys to camera sources
    exterior_frames = _sample_frames(image_queue, args.num_frames, args.video_stride)
    if resize_hw is not None:
        exterior_frames = _resize_frames(exterior_frames, resize_hw)
    video[video_model_keys[0]] = exterior_frames[None].astype(np.uint8)  # (1, T, H, W, C)

    if len(video_model_keys) > 1 and wrist_image_queue is not None:
        wrist_frames = _sample_frames(wrist_image_queue, args.num_frames, args.video_stride)
        if resize_hw is not None:
            wrist_frames = _resize_frames(wrist_frames, resize_hw)
        video[video_model_keys[1]] = wrist_frames[None].astype(np.uint8)  # (1, T, H, W, C)

    # --- state ---
    state = {}
    if state_keys_map is not None:
        for model_key, obs_key in state_keys_map.items():
            val = curr_obs[obs_key].astype(np.float32)
            state[model_key] = val[None, None, :]  # (1, 1, D)
    else:
        # Fallback: EEF mode
        state["end_effector_position"] = curr_obs["eef_position"].astype(np.float32)[None, None, :]
        state["end_effector_rotation"] = curr_obs["eef_rotation"].astype(np.float32)[None, None, :]
        state["gripper_position"] = curr_obs["gripper_position"].astype(np.float32)[None, None, :]

    # --- language ---
    language = {args.language_key: [[instruction]]}

    obs = {"video": video, "state": state, "language": language}

    # --- physics ---
    if physics_queues and physics_num_frames > 0:
        physics = {}
        for sub_key, q in physics_queues.items():
            q_list = list(q)
            # Take last physics_num_frames entries (stride=1, contiguous history)
            selected = q_list[-physics_num_frames:]
            physics[sub_key] = np.stack(selected)[None].astype(np.float32)  # (1, T, D)
        obs["physics"] = physics

    return obs


def main(args: Args):
    assert args.state_space in ("eef", "joint"), (
        f"--state_space must be 'eef' or 'joint', got '{args.state_space}'"
    )
    assert args.action_space in ("eef", "joint"), (
        f"--action_space must be 'eef' or 'joint', got '{args.action_space}'"
    )

    # --- Create env with matching action space ---
    if args.action_space == "eef":
        env_action_space = "cartesian_velocity"
        env_gripper_action_space = "position"
    else:
        env_action_space = "joint_position"
        env_gripper_action_space = "position"

    print("Creating the RLDX DROID env ...")
    env = RobotEnv(action_space=env_action_space, gripper_action_space=env_gripper_action_space)
    print(f"Created the RLDX DROID env! (action_space={env_action_space}, gripper={env_gripper_action_space})")

    policy = create_policy(args)
    modality_cfg = policy.get_modality_config()

    # --- Auto-detect video T and language key ---
    video_T = len(modality_cfg["video"].delta_indices)
    args.num_frames = video_T
    args.language_key = policy.language_key

    # --- Video key mapping: model keys -> camera sources ---
    video_model_keys = modality_cfg["video"].modality_keys
    print(f"  Video model keys: {video_model_keys}")
    print(f"  Camera sources  : exterior={args.video_key_exterior}, wrist={args.video_key_wrist}")

    # --- State key mapping ---
    model_state_keys = modality_cfg["state"].modality_keys
    if args.state_space == "eef":
        # Map model state keys to _extract_observation output keys
        _EEF_KEY_MAP = {
            "end_effector_position": "eef_position",
            "end_effector_rotation": "eef_rotation",
            "gripper_position": "gripper_position",
        }
        state_keys_map = {}
        for mk in model_state_keys:
            if mk not in _EEF_KEY_MAP:
                raise ValueError(
                    f"Model expects state key '{mk}' but state_space='eef' only supports "
                    f"{list(_EEF_KEY_MAP.keys())}. Use --state_space=joint or retrain."
                )
            state_keys_map[mk] = _EEF_KEY_MAP[mk]
    else:  # joint
        _JOINT_KEY_MAP = {
            "joint_pos_abs": "joint_position",
            "gripper_close": "gripper_position",
        }
        state_keys_map = {}
        for mk in model_state_keys:
            if mk not in _JOINT_KEY_MAP:
                raise ValueError(
                    f"Model expects state key '{mk}' but state_space='joint' only supports "
                    f"{list(_JOINT_KEY_MAP.keys())}. Use --state_space=eef or retrain."
                )
            state_keys_map[mk] = _JOINT_KEY_MAP[mk]

    # --- Physics setup ---
    physics_keys = policy.physics_keys  # e.g., ["tactile", "torque"]
    physics_queues: dict[str, deque] | None = None
    physics_num_frames = 0

    if physics_keys:
        # Determine how many history frames the model needs per physics key
        # All physics keys share the same delta_indices (validated at training time)
        for pk in physics_keys:
            if pk in modality_cfg:
                physics_num_frames = len(modality_cfg[pk].delta_indices)
                break

        if physics_num_frames > 0:
            # Build mapping: physics sub-keys -> sensor data extraction
            # Based on modality config (e.g., tactile -> tactile.left, tactile.right; torque -> torque.torque)
            physics_sub_keys = []
            for pk in physics_keys:
                if pk in modality_cfg:
                    for sk in modality_cfg[pk].modality_keys:
                        physics_sub_keys.append(f"{pk}.{sk}")

            physics_queues = {}
            for sub_key in physics_sub_keys:
                physics_queues[sub_key] = deque(maxlen=physics_num_frames)

            print(f"  Physics keys    : {physics_keys}")
            print(f"  Physics sub-keys: {physics_sub_keys}")
            print(f"  Physics T (hist): {physics_num_frames} frame(s)")

    # Detect which physics sensor types are needed
    use_tactile = physics_queues is not None and any(
        k.startswith("tactile.") for k in physics_queues
    )
    use_torque = physics_queues is not None and "torque.torque" in physics_queues
    wowskin_right, wowskin_left = None, None

    resize_hw: tuple[int, int] | None = None
    if args.resize_video is not None:
        parts = args.resize_video.split(",")
        assert len(parts) == 2, f"--resize_video must be 'H,W' (e.g., '168,336'), got: {args.resize_video}"
        resize_hw = (int(parts[0]), int(parts[1]))

    print("Loaded RLDX policy model!")
    print(f"  Checkpoint : {args.policy.dir}")
    print(f"  Embodiment : {args.embodiment_tag}")
    print(f"  Memory     : {policy.use_memory}")
    print(f"  Video T    : {video_T} frame(s) per inference")
    print(f"  Language   : {args.language_key}")
    print(f"  State space: {args.state_space} (keys: {list(state_keys_map.keys())})")
    print(f"  Action space: {args.action_space} (env: {env_action_space})")
    print(f"  Resize     : {resize_hw if resize_hw is not None else 'disabled (using processor transform)'}")
    print(f"  Modality   : {modality_cfg}")

    df = pd.DataFrame(columns=[
        "task", "grasp", "success", "grasp_rate", "success_rate", "num_episodes", "avg_inference_time"
    ])
    task_inference_times = defaultdict(list)

    num_trials = 0
    last_instruction = None
    success_list = []
    grasp = 0
    success_cnt = 0

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    policy_folder = f"{timestamp}"

    while True:
        inference_times = []

        if args.instruction is not None:
            instruction = args.instruction
            print(f"Instruction: {instruction}")
        else:
            instruction = input("Enter instruction: ")
            if last_instruction is not None and instruction == "":
                print(f"Using last instruction: {last_instruction}")
                instruction = last_instruction

        task_name = re.sub(r'[^\w\-]', '_', instruction.strip())[:60]
        videos_dir = os.path.join(args.logging_path, args.policy.config, task_name, policy_folder, "videos")
        results_dir = os.path.join(args.logging_path, args.policy.config, task_name, policy_folder, "results")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        tsv_filename = os.path.join(results_dir, "eval.tsv")
        if not os.path.exists(tsv_filename):
            df.to_csv(tsv_filename, sep='\t', index=False)

        process_tracking_tsv = os.path.join(results_dir, "process_tracking.tsv")
        if not os.path.exists(process_tracking_tsv):
            with open(process_tracking_tsv, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["episode_id", "action_chunk_idx", "action_chunk", "success"])

        actions_from_chunk_completed = 0
        concatenated_action = None
        action_chunk_idx = 0
        episode_id = num_trials

        # Memory state management: unique session per episode, reset on first inference
        session_id = f"real_robot_{os.getpid()}_{num_trials}"
        is_first_inference = True

        video_log = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        start_rollout_time = time.time()
        interrupted = False

        # deque holds enough history to cover the furthest frame index.
        # For stride S and T frames: indices [-( T-1)*S, ..., -S, 0]
        # -> need (T-1)*S + 1 slots in the deque.
        queue_maxlen = (args.num_frames - 1) * args.video_stride + 1
        initial_frames = queue_maxlen - 1  # pre-fill with dummies so first real frame completes the window
        image_queue = deque([DUMMY_IMAGE] * initial_frames, maxlen=queue_maxlen)
        wrist_image_queue = deque([DUMMY_IMAGE] * initial_frames, maxlen=queue_maxlen) \
            if args.video_key_wrist is not None else None

        # Reset physics queues for new episode
        if physics_queues is not None:
            for q in physics_queues.values():
                q.clear()

        # Initialize/reinitialize WowSkin tactile sensors each episode for fresh baseline
        if use_tactile:
            if wowskin_right is not None:
                wowskin_right.stop()
            if wowskin_left is not None:
                wowskin_left.stop()
            time.sleep(0.5)
            from droid.trajectory_utils.wowskin import WowSkin
            print("Initializing WowSkin tactile sensors...")
            wowskin_right = WowSkin(
                port=args.wowskin_port_right, compute_baseline=True, baseline_samples=5,
            )
            time.sleep(0.8)
            wowskin_left = WowSkin(
                port=args.wowskin_port_left, compute_baseline=True, baseline_samples=5,
            )
            print("WowSkin tactile sensors ready.")

        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(args, env.get_observation())

                exterior_image = curr_obs["left_image"]
                wrist_image = curr_obs.get("wrist_image")

                if wrist_image is not None:
                    concat_image = np.concatenate([exterior_image, wrist_image], axis=1)
                else:
                    concat_image = exterior_image
                video_log.append(concat_image)

                if t_step == args.num_steps_wait:
                    for _ in range(queue_maxlen):
                        image_queue.append(exterior_image)
                        if wrist_image_queue is not None and wrist_image is not None:
                            wrist_image_queue.append(wrist_image)
                else:
                    image_queue.append(exterior_image)
                    if wrist_image_queue is not None and wrist_image is not None:
                        wrist_image_queue.append(wrist_image)

                # Accumulate physics sensor data
                if physics_queues is not None:
                    # At num_steps_wait, flush queues with current reading (like video flush)
                    flush = (t_step == args.num_steps_wait)
                    n_push = physics_num_frames if flush else 1

                    if use_torque:
                        val = curr_obs.get("joint_torques")
                        if val is None:
                            raise RuntimeError(
                                "Model requires torque but 'joint_torques_computed' not found "
                                "in robot_state. Check _extract_observation."
                            )
                        val = val.astype(np.float32)
                        for _ in range(n_push):
                            physics_queues["torque.torque"].append(val)

                    if use_tactile:
                        r = wowskin_right.read(as_dict=False)
                        l = wowskin_left.read(as_dict=False)
                        r_flat = r.reshape(-1).astype(np.float32) if r is not None else np.zeros(15, dtype=np.float32)
                        l_flat = l.reshape(-1).astype(np.float32) if l is not None else np.zeros(15, dtype=np.float32)
                        for _ in range(n_push):
                            physics_queues["tactile.right"].append(r_flat)
                            physics_queues["tactile.left"].append(l_flat)

                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # Skip inference until we have enough physics history
                    if physics_queues is not None and any(
                        len(q) < physics_num_frames for q in physics_queues.values()
                    ):
                        continue

                    observation = build_observation(
                        args, image_queue, wrist_image_queue, curr_obs, instruction,
                        resize_hw=resize_hw,
                        video_model_keys=video_model_keys,
                        state_keys_map=state_keys_map,
                        physics_queues=physics_queues,
                        physics_num_frames=physics_num_frames,
                    )

                    with prevent_keyboard_interrupt():
                        infer_start = time.time()
                        # action: {key: np.ndarray(B=1, T, D)}
                        # Always send session/reset signal — feature independence.
                        # Server routes by enabled features (memory, RTC, ...);
                        # gating on any single feature here drops reset propagation
                        # for the others.
                        options = {
                            "reset_memory": [is_first_inference],
                            "session_ids": [session_id],
                        }
                        action_dict, _ = policy.get_action(observation, options=options)
                        is_first_inference = False
                        infer_end = time.time()

                    # Concatenate all action keys along last dim: (T, total_action_dim)
                    action_keys = list(action_dict.keys())
                    concatenated_action = np.concatenate(
                        [action_dict[k][0] for k in action_keys],  # each: (T, D)
                        axis=-1
                    )

                    if args.binarize_gripper:
                        concatenated_action[:, -1] = (concatenated_action[:, -1] > 0.5).astype(np.float32)

                    inference_time = infer_end - infer_start
                    inference_times.append(inference_time)

                    with open(process_tracking_tsv, "a", newline="") as f:
                        writer = csv.writer(f, delimiter="\t")
                        writer.writerow([episode_id, action_chunk_idx, "", ""])
                    action_chunk_idx += 1

                if concatenated_action is None:
                    # Still warming up physics buffer, skip execution
                    continue

                # Execute current action from chunk
                action = concatenated_action[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                env.step(action)

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)

            except KeyboardInterrupt:
                interrupted = True
                break

        if interrupted:
            cancel = input("Do you want to cancel this execution? (y/n): ").lower()
            if cancel in ["y"]:
                print("This execution will not be included in the statistics.")
                if input("Do one more eval? (enter y or n) ").lower() not in ["y", ""]:
                    break
                last_instruction = instruction
                env.reset()
                while input("Does robot reset? (enter y or n) ").lower() in ["n"]:
                    env.reset()
                continue

        video_arr = np.stack(video_log)
        video_filename = f"{num_trials+1:03d}.mp4"
        save_filename = os.path.join(videos_dir, video_filename)
        ImageSequenceClip(list(video_arr), fps=15).write_videofile(save_filename, codec="libx264")

        success = get_success_rate()
        success_list.append(success)
        num_trials += 1
        if success == 0.5:
            grasp += 1
        elif success == 1.0:
            success_cnt += 1
            grasp += 1
        grasp_rate = grasp / len(success_list)
        success_rate = success_cnt / len(success_list)
        avg_inference_time = np.mean(inference_times) if inference_times else 0

        task_key = instruction.lower().strip()
        task_inference_times[task_key].append(avg_inference_time)

        print(f"Current trajectory success: {success}")
        print(f"Overall success rate: {success_rate:.3f}")
        print(f"Overall grasp rate: {grasp_rate:.3f}")
        print(f"Average inference time (this trial): {avg_inference_time:.4f} sec")

        df = df[df["task"].str.lower().str.strip() != task_key]
        df.loc[len(df)] = [
            instruction, grasp, success_cnt,
            f"{grasp_rate * 100:.3f}", f"{success_rate * 100:.3f}",
            num_trials, f"{avg_inference_time:.4f}",
        ]
        df.to_csv(tsv_filename, sep='\t', index=False)

        if input("Do one more eval? (enter y or n) ").lower() not in ["y", ""]:
            break
        last_instruction = instruction
        env.reset()
        while input("Does robot reset? (enter y or n) ").lower() in ["n"]:
            env.reset()

    # Cleanup WowSkin sensors
    if wowskin_right is not None:
        wowskin_right.stop()
    if wowskin_left is not None:
        wowskin_left.stop()

    for idx, row in df.iterrows():
        task_key = row["task"].lower().strip()
        avg_time = np.mean(task_inference_times[task_key]) if task_inference_times[task_key] else 0
        df.at[idx, "avg_inference_time"] = f"{avg_time:.4f}"
    df.to_csv(tsv_filename, sep='\t', index=False)
    print(f"Results saved to {tsv_filename}")


def _extract_observation(args: Args, obs_dict: dict) -> dict:
    image_observations = obs_dict["image"]
    left_image, wrist_image = None, None

    for key in image_observations:
        if "left" not in key:
            continue
        if args.left_camera_id in key:
            left_image = image_observations[key]
        elif args.wrist_camera_id in key:
            wrist_image = image_observations[key]

    left_image = left_image[..., :3][..., ::-1].astype(np.uint8)
    if wrist_image is not None:
        wrist_image = wrist_image[..., :3][..., ::-1].astype(np.uint8)

    robot_state = obs_dict["robot_state"]
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])
    cartesian_position = np.array(robot_state["cartesian_position"][:3])
    cartesian_rotation = np.array(robot_state["cartesian_position"][3:])

    result = {
        "left_image": left_image,
        "wrist_image": wrist_image,
        "joint_position": joint_position,
        "eef_position": cartesian_position,
        "eef_rotation": cartesian_rotation,
        "gripper_position": gripper_position,
    }

    # Torque: 7-dim joint torques from robot state
    result["joint_torques"] = np.array(robot_state["joint_torques_computed"], dtype=np.float32)

    return result


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
