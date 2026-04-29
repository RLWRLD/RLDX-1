import collections
from collections import defaultdict
import dataclasses
import json
import os
from pathlib import Path
import pickle
import sys
import time

import imageio
import numpy as np
from PIL import Image
import tyro


# CALVIN is an external repo checked out locally. Default path is the
# `external_dependencies/calvin` checkout at the RLDX project root; override
# with `CALVIN_REPO` for non-standard layouts (e.g. shared clusters).
_DEFAULT_CALVIN = Path(__file__).resolve().parents[4] / "external_dependencies" / "calvin"
repo_path = os.environ.get("CALVIN_REPO", str(_DEFAULT_CALVIN))
sys.path.insert(0, repo_path)

from calvin_agent.evaluation.utils import (  # noqa: E402
    count_success,
    get_env_state_for_initial_condition,
    print_and_save,
)
from calvin_env.envs.play_table_env import get_env  # noqa: E402
import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402
from termcolor import colored  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402


EP_LEN = 360
NUM_SEQUENCES = 1000
CONF_DIR = repo_path


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 5

    #################################################################################################################
    # Calvin environment-specific parameters
    #################################################################################################################
    dataset_path: str = ""  # Path to the dataset root directory
    debug: bool = False  # Print debug info and show detailed execution
    save_video_dir: str = None  # Path to save video file

    #################################################################################################################
    # Evaluation parameters
    #################################################################################################################
    num_sequences: int = NUM_SEQUENCES  # Number of evaluation sequences
    ep_len: int = EP_LEN  # Episode length
    sequence_path: str = str(Path(repo_path) / "calvin_eval_sequences")
    task_idx: int = -1  # Task index to run

    num_frames: int = 1  # Number of frames to concatenate
    video_stride: int = 1


def make_env(dataset_path):
    """Create CALVIN environment from dataset path."""
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)
    return env


def create_video_name(task_idx, seq_idx, success_count, eval_sequence):
    instr_str = "@".join(eval_sequence)
    return f"task_{task_idx}_seq_{seq_idx}_success_{success_count}_{instr_str}.mp4"


def save_frames_as_video(frames, output_path, fps=30):
    """Save a list of RGB frames as a video file."""
    if not frames:
        print("No frames to save.")
        return

    imageio.mimsave(output_path, frames, fps=fps)


def evaluate_policy(env, args: Args):
    """
    Run model evaluation on CALVIN challenge following evaluate_policy.py structure.
    """
    # Load task configuration and annotations (same as evaluate_policy.py)
    conf_dir = Path(CONF_DIR) / "calvin_models" / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # Create ClientPolicy
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Get evaluation sequences

    sequence_path = Path(args.sequence_path) / f"eval_seq_v3_{args.task_idx}.pkl"
    with open(sequence_path, "rb") as f:
        eval_sequences = pickle.load(f)

    print(f"Loaded {len(eval_sequences)} evaluation sequences for task {args.task_idx}")

    results = []
    plans = defaultdict(list)
    collect_frames = args.save_video_dir is not None

    if not args.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for seq_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        print(f"#[{seq_idx}] {eval_sequence}")

        # check if rollout already exists
        rollout_exists = False
        for success_count in range(6):
            video_name = create_video_name(args.task_idx, seq_idx, success_count, eval_sequence)
            save_video_path = Path(args.save_video_dir) / video_name
            if save_video_path.exists():
                print(
                    f"Video already exists for sequence {seq_idx}, success count: {success_count}"
                )
                results.append(success_count)
                rollout_exists = True
                break

        if rollout_exists:
            continue

        if collect_frames:
            success_count, frames = evaluate_sequence(
                env,
                client,
                task_oracle,
                initial_state,
                eval_sequence,
                val_annotations,
                plans,
                args,
                collect_frames,
            )
            results.append(success_count)

            if frames:
                video_name = create_video_name(args.task_idx, seq_idx, success_count, eval_sequence)
                save_video_path = Path(args.save_video_dir) / video_name
                save_frames_as_video(frames, save_video_path)

        else:
            result = evaluate_sequence(
                env, client, task_oracle, initial_state, eval_sequence, val_annotations, plans, args
            )
            results.append(result)

        avg_len = sum(results) / len(results)
        eval_sequences.set_description(
            " ".join(
                [f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]
            )
            + "|"
            + f"Avg len: {avg_len:.3f}"
            + "|"
        )

    # Print results in same format as evaluate_policy.py
    print_and_save(
        results, eval_sequences, log_dir=Path(args.save_video_dir), epoch="websocket_policy"
    )

    # save result to file
    result_save_path = Path(args.save_video_dir) / f"task_{args.task_idx}_results.json"
    with open(result_save_path, "w") as f:
        json.dump(results, f)

    return results


def evaluate_sequence(
    env,
    client,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    plans,
    args,
    collect_frames=False,
):
    """
    Evaluates a sequence of language instructions with WebsocketClientPolicy.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    all_frames = []

    if args.debug:
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")

    for subtask in eval_sequence:
        if collect_frames:
            result = rollout(
                env, client, task_checker, subtask, val_annotations, plans, args, collect_frames
            )
            if isinstance(result, tuple):
                success, frames = result
                all_frames.extend(frames)
            else:
                success = result
        else:
            success = rollout(env, client, task_checker, subtask, val_annotations, plans, args)

        if success:
            success_counter += 1
        else:
            if collect_frames:
                return success_counter, all_frames
            return success_counter

    if collect_frames:
        return success_counter, all_frames
    return success_counter


def rollout(env, client, task_oracle, subtask, val_annotations, plans, args, collect_frames=False):
    """
    Run the actual rollout on one subtask with WebsocketClientPolicy and action chunk replanning.
    """
    if args.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()
    # Get language annotation for subtask
    lang_annotation = val_annotations[subtask][0]

    if args.debug:
        print(f"\nLanguage instruction: '{lang_annotation}'")

    start_info = env.get_info()

    # Initialize action plan for chunk-based execution
    action_plan = collections.deque()

    # Collect frames for video if requested
    frames = []

    image_queue = collections.deque(maxlen=args.num_frames * args.video_stride)
    wrist_image_queue = collections.deque(maxlen=args.num_frames * args.video_stride)

    for step in range(args.ep_len):
        # Check if we need to get new action chunk (replan)
        if not action_plan:
            # Prepare observation element for WebsocketClientPolicy
            # Calvin environment provides observations in different format than LIBERO
            # Need to adapt to expected format
            img = obs["rgb_obs"]["rgb_static"]  # Main camera view
            wrist_img = obs["rgb_obs"]["rgb_gripper"]  # Gripper camera view

            # convert to (256, 256, 3)
            img_pil = Image.fromarray(img)
            img_resized = img_pil.resize((256, 256), resample=Image.Resampling.LANCZOS)
            img = np.array(img_resized)

            wrist_img_pil = Image.fromarray(wrist_img)
            wrist_img_resized = wrist_img_pil.resize((256, 256), resample=Image.Resampling.LANCZOS)
            wrist_img = np.array(wrist_img_resized)

            if step == 0:
                for i in range(args.num_frames * args.video_stride):
                    image_queue.append(img)
                    wrist_image_queue.append(wrist_img)
            else:
                image_queue.append(img)
                wrist_image_queue.append(wrist_img)

            robot_pos = obs["robot_obs"]
            image_obs = np.array(image_queue)[args.video_stride - 1 :: args.video_stride]
            wrist_image_obs = np.array(wrist_image_queue)[
                args.video_stride - 1 :: args.video_stride
            ]

            element = {
                "video": {
                    "image": image_obs[np.newaxis],
                    "wrist_image": wrist_image_obs[np.newaxis],
                },
                "state": {"state": np.array([robot_pos], dtype=np.float32)[np.newaxis]},
                "language": {"annotation.human.action.task_description": [[str(lang_annotation)]]},
            }

            action_chunk = client.infer(element)[0]

            # CALVIN
            gripper_close = action_chunk["gripper_close"]
            gripper_close = np.where(gripper_close > 0, 1, -1)

            action_chunk = np.concatenate(
                [action_chunk["eef_pos_delta"], action_chunk["eef_rot_delta"], gripper_close],
                axis=-1,
            ).reshape(-1, 7)

            # Ensure we have enough actions for replanning
            assert len(action_chunk) >= args.replan_steps, (
                f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
            )

            # Add actions to plan for next replan_steps
            action_plan.extend(action_chunk[: args.replan_steps])

        # Get next action from plan
        action = action_plan.popleft()

        # Execute action in environment
        obs, _, _, current_info = env.step(action)

        # Collect frame for video if requested
        if collect_frames:
            raw_img = obs["rgb_obs"]["rgb_static"]
            raw_img = Image.fromarray(raw_img).resize((256, 256), resample=Image.Resampling.LANCZOS)
            frames.append(np.array(raw_img))

        if args.debug and step < 5:  # Show first few actions
            print(f"  Step {step}: Action {action}")

        # Check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if args.debug:
                print(colored("success", "green"), end=" ")
            if collect_frames:
                return True, frames
            return True

    if args.debug:
        print(colored("fail", "red"), end=" ")
    if collect_frames:
        return False, frames
    return False


def main(args: Args) -> None:
    # Validate required arguments
    if not args.dataset_path:
        raise ValueError("dataset_path is required")

    # Create environment
    print(f"Loading environment from: {args.dataset_path}")
    env = make_env(args.dataset_path)

    print(f"Starting WebsocketClientPolicy evaluation on {args.num_sequences} sequences...")
    print("This follows the same evaluation protocol as evaluate_policy.py")
    print("=" * 60)

    # create save video directory if it doesn't exist
    if args.save_video_dir is not None:
        Path(args.save_video_dir).mkdir(parents=True, exist_ok=True)

    # Run the full sequence evaluation
    evaluate_policy(env, args)

    print("=" * 60)
    print("Evaluation completed!")


if __name__ == "__main__":
    tyro.cli(main)
