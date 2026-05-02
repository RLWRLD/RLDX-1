import cv2
import h5py
import tqdm
import tyro
import json
import tempfile
import dataclasses
from pathlib import Path
from typing import Literal, Optional
import pyarrow.parquet as pq

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


""" USAGE (single dataset, single task, inline physics flags):
python convert_h5_to_v2.py \
  --raw-dir /path/to/raw_datasets/swap_cubes \
  --output-path /path/to/output/swap_cubes \
  --repo-id swap_cubes \
  --task "Swap the positions of two cubes, starting with the blue one." \
  --robot-type franka_panda \
  --action-type joint \
  --mode video \
  --fps 10 \
  --no-tactile

USAGE (multiple sub-datasets, JSON-based task & physics mapping):
python convert_h5_to_v2.py \
  --raw-dir /path/to/raw_datasets/tactile_moss \
  --output-path /path/to/output/tactile_moss \
  --repo-id tactile_moss \
  --task-map /path/to/task_map.json \
  --physics-map /path/to/physics_map.json \
  --robot-type franka_panda \
  --action-type eef \
  --mode video \
  --fps 10

  task_map.json example:
  {
    "gripper_unstack_cup_final": "Pick up the top red cup ...",
    "dust_clean_up": "Pick up the brush ..."
  }

  physics_map.json example (value is CLI flags string):
  {
    "tactile_moss": "",
    "dust_clean_up": "--no-tactile",
    "some_dataset": "--no-tactile --no-torque"
  }
"""


# ----------------------------
# Config
# ----------------------------
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True  # if True, LeRobot will store episode images as mp4
    tolerance_s: float = 0.0001
    image_writer_processes: int = 8
    image_writer_threads: int = 4
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


# ----------------------------
# Utilities
# ----------------------------
MOTORS_JOINT = [f"joint{i}" for i in range(1, 8)] + ["gripper"]  # 8-dim
MOTORS_EEF = [f"eef{i}" for i in range(1, 7)] + ["gripper"]      # 7-dim


def _get_motors(action_type: str) -> list[str]:
    if action_type == "joint":
        return MOTORS_JOINT
    return MOTORS_EEF


def _decode_video_bytes_to_frames(video_bytes: bytes) -> list[np.ndarray]:
    """Decode mp4 bytes to a list of RGB frames (H, W, 3) uint8."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        frames: list[np.ndarray] = []
        ok, frame_bgr = cap.read()
        while ok:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            ok, frame_bgr = cap.read()
        cap.release()
    return frames


def _estimate_fps_from_timestamps(ep: h5py.File) -> Optional[int]:
    """Estimate fps from control timestamps (nanoseconds)."""
    key = "/observation/timestamp/control/step_start"
    if key not in ep:
        return None
    t = ep[key][:]
    if len(t) < 3:
        return None
    dt = np.diff(t).astype(np.float64)
    med = np.median(dt)
    if med <= 0:
        return None
    fps = int(round(1e9 / med))
    if 1 <= fps <= 240:
        return fps
    return None


def _read_camera_names(ep: h5py.File) -> list[str]:
    if "/observation/videos" not in ep:
        return []
    return list(ep["/observation/videos"].keys())


def _probe_image_shape(frames: list[np.ndarray]) -> tuple[int, int]:
    """Return (H, W) from first frame."""
    if len(frames) == 0:
        raise RuntimeError("No frames decoded from video.")
    h, w = frames[0].shape[:2]
    return h, w


# ----------------------------
# LeRobot dataset creation
# ----------------------------
def create_empty_dataset(
    *,
    repo_id: str,
    output_path: Path,
    robot_type: str,
    cameras: list[str],
    image_hw: tuple[int, int],
    mode: Literal["video", "image"],
    action_type: Literal["joint", "eef"],
    has_velocity: bool,
    has_tactile: bool = True,
    has_torque: bool = True,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    fps: int,
) -> LeRobotDataset:

    motors = _get_motors(action_type)
    H, W = image_hw
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": "state",
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": "action",
        },
    }

    # cartesian pose stored as a separate feature only in joint mode
    if action_type == "joint":
        features["observation.cartesian_position"] = {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "rx", "ry", "rz"],
        }

    if has_tactile:
        features["observation.tactile.left"] = {
            "dtype": "float32",
            "shape": (15,),
            "names": "tactile_left",
        }
        features["observation.tactile.right"] = {
            "dtype": "float32",
            "shape": (15,),
            "names": "tactile_right",
        }

    if has_torque:
        features["observation.torque"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": "torque",
        }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": "velocity",
        }

    # cameras
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, H, W),
            "names": ["channels", "height", "width"],
        }

    print(f"Path: {output_path}")
    output_path = Path(output_path)
    if output_path.exists():
        print(f"[WARN] {output_path} already exists. Aborting to avoid overwriting.")
        return None

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        root=output_path,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )
    return ds


# ----------------------------
# Episode loading
# ----------------------------
@dataclasses.dataclass
class EpisodeData:
    state: torch.Tensor                        # (T, 8) for joint, (T, 7) for eef
    action: torch.Tensor                       # (T, 8) for joint, (T, 7) for eef
    velocity: Optional[torch.Tensor]           # (T, 8) or None
    cartesian: Optional[torch.Tensor]          # (T, 6) or None
    tactile_left: Optional[torch.Tensor]       # (T, 15) or None
    tactile_right: Optional[torch.Tensor]      # (T, 15) or None
    torque: Optional[torch.Tensor]             # (T, 7) or None
    images_per_cam: dict[str, list[np.ndarray]]


def load_episode(
    ep_path: Path,
    camera_names: list[str],
    action_type: Literal["joint", "eef"],
    has_tactile: bool = True,
    has_torque: bool = True,
) -> EpisodeData:
    with h5py.File(ep_path, "r") as ep:
        # --- state & action (differs by action_type) ---
        if action_type == "joint":
            qpos = ep["/observation/robot_state/joint_positions"][:]       # (T, 7)
            gr_pos = ep["/observation/robot_state/gripper_position"][:]    # (T,)
            state = np.concatenate([qpos, gr_pos[:, None]], axis=1).astype(np.float32)  # (T, 8)

            a_qpos = ep["/action/joint_position"][:]                      # (T, 7)
            a_gr = ep["/action/gripper_position"][:]                      # (T,)
            action = np.concatenate([a_qpos, a_gr[:, None]], axis=1).astype(np.float32)  # (T, 8)
        else:  # eef
            obs_cart = ep["/observation/robot_state/cartesian_position"][:]  # (T, 6)
            obs_grip = ep["/observation/robot_state/gripper_position"][:]    # (T,)
            state = np.concatenate([obs_cart[:, :3], obs_cart[:, 3:6], obs_grip[:, None]], axis=1).astype(np.float32)  # (T, 7)

            action_cart_vel = ep["/action/cartesian_velocity"][:]           # (T, 6)
            action_gripper = ep["/action/gripper_position"][:]             # (T,)
            action = np.concatenate([action_cart_vel, action_gripper[:, None]], axis=1).astype(np.float32)  # (T, 7)

        # --- velocity (optional) ---
        vel: Optional[np.ndarray] = None
        if "/observation/robot_state/joint_velocities" in ep and "/action/gripper_velocity" in ep:
            jv = ep["/observation/robot_state/joint_velocities"][:]        # (T, 7)
            gv = ep["/action/gripper_velocity"][:]                        # (T,)
            vel = np.concatenate([jv, gv[:, None]], axis=1).astype(np.float32)  # (T, 8)

        # --- cartesian pose (only stored separately in joint mode) ---
        cart: Optional[np.ndarray] = None
        if action_type == "joint":
            if "/observation/robot_state/cartesian_position" in ep:
                cart = ep["/observation/robot_state/cartesian_position"][:].astype(np.float32)  # (T, 6)
            else:
                cart = np.zeros((state.shape[0], 6), dtype=np.float32)

        # --- tactile (optional) ---
        tactile_left: Optional[np.ndarray] = None
        tactile_right: Optional[np.ndarray] = None
        if has_tactile:
            if "/tactile_left" in ep and "/tactile_right" in ep:
                raw_left = ep["/tactile_left"][:].astype(np.float32)      # (T, 5, 3)
                raw_right = ep["/tactile_right"][:].astype(np.float32)    # (T, 5, 3)
                tactile_left = raw_left.reshape(raw_left.shape[0], -1)    # (T, 15)
                tactile_right = raw_right.reshape(raw_right.shape[0], -1) # (T, 15)
            else:
                tactile_left = np.zeros((state.shape[0], 15), dtype=np.float32)
                tactile_right = np.zeros((state.shape[0], 15), dtype=np.float32)

        # --- torque (optional) ---
        torque: Optional[np.ndarray] = None
        if has_torque:
            if "/observation/robot_state/prev_joint_torques_computed_safened" in ep:
                torque = ep["/observation/robot_state/prev_joint_torques_computed_safened"][:].astype(np.float32)  # (T, 7)
            else:
                torque = np.zeros((state.shape[0], 7), dtype=np.float32)

        # --- videos ---
        images_per_cam: dict[str, list[np.ndarray]] = {}
        if "/observation/videos" in ep:
            for cam in camera_names:
                data = ep[f"/observation/videos/{cam}"][()]
                frames = _decode_video_bytes_to_frames(data)
                images_per_cam[cam] = frames
        else:
            images_per_cam = {cam: [] for cam in camera_names}

        # --- align lengths ---
        lengths = [state.shape[0], action.shape[0]]
        if cart is not None:
            lengths.append(cart.shape[0])
        if tactile_left is not None:
            lengths.append(tactile_left.shape[0])
        if tactile_right is not None:
            lengths.append(tactile_right.shape[0])
        if torque is not None:
            lengths.append(torque.shape[0])
        if vel is not None:
            lengths.append(vel.shape[0])
        for cam, frames in images_per_cam.items():
            lengths.append(len(frames))

        T = min(lengths)

        state = state[:T]
        action = action[:T]
        if cart is not None:
            cart = cart[:T]
        if tactile_left is not None:
            tactile_left = tactile_left[:T]
        if tactile_right is not None:
            tactile_right = tactile_right[:T]
        if torque is not None:
            torque = torque[:T]
        if vel is not None:
            vel = vel[:T]
        for cam in list(images_per_cam.keys()):
            images_per_cam[cam] = images_per_cam[cam][:T]

        return EpisodeData(
            state=torch.from_numpy(state),
            action=torch.from_numpy(action),
            velocity=torch.from_numpy(vel) if vel is not None else None,
            cartesian=torch.from_numpy(cart) if cart is not None else None,
            tactile_left=torch.from_numpy(tactile_left) if tactile_left is not None else None,
            tactile_right=torch.from_numpy(tactile_right) if tactile_right is not None else None,
            torque=torch.from_numpy(torque) if torque is not None else None,
            images_per_cam=images_per_cam,
        )


# ----------------------------
# Population routine
# ----------------------------
def _resolve_task(ep_path: Path, raw_dir: Path, task: Optional[str], task_map: Optional[dict[str, str]]) -> str:
    """Determine the task string for a given episode file.

    Lookup order when task_map is provided (first match wins):
      1. Each directory component of the relative path, from deepest to shallowest
         e.g. raw_dir/A/B/NNNN/file.h5 → try B, then A
      2. raw_dir's own directory name (flat layout fallback)
      3. --task fallback
    """
    if task_map is not None:
        rel = ep_path.relative_to(raw_dir)
        # rel.parts examples:
        #   flat:    ('0042', 'teleoperation.h5')
        #   nested:  ('gripper_pnp_egg_final', '0042', 'teleoperation.h5')
        #   deep:    ('start-cookie', 'butter-cookie-left_green-cup-middle', 'teleoperation.h5')
        # Check all directory components (exclude filename), deepest first
        dir_parts = rel.parts[:-1]  # strip 'teleoperation.h5'
        for part in reversed(dir_parts):
            if part in task_map:
                return task_map[part]
        # flat layout: use raw_dir's own name as the key
        if raw_dir.name in task_map:
            return task_map[raw_dir.name]
        # per-episode mapping: "dataset_name/NNNN/teleoperation.h5"
        full_key = f"{raw_dir.name}/{rel}"
        if full_key in task_map:
            return task_map[full_key]
        if task is not None:
            return task
        raise ValueError(
            f"No task_map key found for episode {ep_path}. "
            f"Tried {list(reversed(dir_parts))}, '{raw_dir.name}', and '{full_key}'. "
            f"Available keys (first 10): {list(task_map.keys())[:10]}"
        )
    if task is not None:
        return task
    raise ValueError("Either --task or --task-map must be provided.")


def _resolve_physics(raw_dir: Path, physics_map: Optional[dict[str, str]],
                     no_tactile: bool, no_torque: bool) -> tuple[bool, bool]:
    """Determine has_tactile / has_torque for a dataset.

    Lookup order when physics_map is provided:
      1. raw_dir.name (e.g. "dust_clean_up", "tactile_moss")
      2. --no-tactile / --no-torque fallback
    Returns (has_tactile, has_torque).
    """
    if physics_map is not None:
        key = raw_dir.name
        if key in physics_map:
            flags = physics_map[key]
            return ("--no-tactile" not in flags, "--no-torque" not in flags)
    return (not no_tactile, not no_torque)


def populate_dataset(
    dataset: LeRobotDataset,
    h5_files: list[Path],
    raw_dir: Path,
    task: Optional[str],
    task_map: Optional[dict[str, str]],
    cameras: list[str],
    action_type: Literal["joint", "eef"],
    has_velocity: bool = True,
    has_tactile: bool = True,
    has_torque: bool = True,
) -> LeRobotDataset:

    for ep_idx, ep_path in enumerate(tqdm.tqdm(h5_files, desc="Converting episodes")):
        ep = load_episode(ep_path, cameras, action_type, has_tactile=has_tactile, has_torque=has_torque)
        T = ep.state.shape[0]
        if T == 0:
            print(f"[WARN] Skipping {ep_path}: 0 frames after alignment")
            continue
        ep_task = _resolve_task(ep_path, raw_dir, task, task_map)

        for t in range(T):
            frame = {
                "task": ep_task,
                "observation.state": ep.state[t],
                "action": ep.action[t],
            }
            if action_type == "joint" and ep.cartesian is not None:
                frame["observation.cartesian_position"] = ep.cartesian[t]
            if has_tactile and ep.tactile_left is not None and ep.tactile_right is not None:
                frame["observation.tactile.left"] = ep.tactile_left[t]
                frame["observation.tactile.right"] = ep.tactile_right[t]
            if has_torque and ep.torque is not None:
                frame["observation.torque"] = ep.torque[t]
            if has_velocity and ep.velocity is not None:
                frame["observation.velocity"] = ep.velocity[t]

            for cam, frames in ep.images_per_cam.items():
                img = frames[t]
                chw = torch.from_numpy(np.transpose(img, (2, 0, 1)))
                frame[f"observation.images.{cam}"] = chw
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


# ----------------------------
# Parquet patch (List -> Sequence)
# ----------------------------
def _count_list_nodes(node):
    if isinstance(node, dict):
        return (1 if node.get("_type") == "List" else 0) + sum(_count_list_nodes(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_list_nodes(x) for x in node)
    return 0


def _fix_node(node):
    if isinstance(node, dict):
        if node.get("_type") == "List":
            node["_type"] = "Sequence"
        for k, v in list(node.items()):
            node[k] = _fix_node(v)
    elif isinstance(node, list):
        node = [_fix_node(x) for x in node]
    return node


def _patch_parquet_file(p: Path, compression: str | None = None) -> bool:
    meta = pq.read_metadata(p).metadata or {}
    hf_raw = meta.get(b"huggingface")
    if not hf_raw:
        return False

    obj = json.loads(hf_raw.decode())
    if _count_list_nodes(obj) == 0:
        return False

    fixed = _fix_node(obj)
    table = pq.read_table(p, memory_map=True)
    new_md = dict(table.schema.metadata or {})
    new_md[b"huggingface"] = json.dumps(fixed, separators=(",", ":")).encode()
    table_fixed = table.replace_schema_metadata(new_md)

    tmp = p.with_suffix(".parquet.tmp")
    pq.write_table(table_fixed, tmp, compression=compression)
    tmp.replace(p)
    return True


def _patch_dataset_parquet(data_root: Path, pattern: str = "chunk-*/episode_*.parquet"):
    patched = 0
    for p in sorted(data_root.glob(pattern)):
        if _patch_parquet_file(p, compression=None):
            patched += 1
    print(f"[PATCH] fixed {patched} parquet file(s)")


# ----------------------------
# CLI entry
# ----------------------------
def convert_h5_dir_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    output_path: Path,
    *,
    robot_type: str = "franka_panda",
    action_type: Literal["joint", "eef"] = "joint",
    push_to_hub: bool = False,
    task: Optional[str] = None,
    task_map: Optional[Path] = None,
    physics_map: Optional[Path] = None,
    mode: Literal["video", "image"] = "video",
    fps: Optional[int] = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    episodes: Optional[list[int]] = None,
    no_tactile: bool = False,
    no_torque: bool = False,
):
    """
    Convert a directory that contains numbered subdirs (e.g., 0000..NNNN) with teleoperation.h5
    into a LeRobot dataset. Supports both flat and nested directory layouts.

    Args:
        action_type: "joint" for joint-space state/action, "eef" for end-effector cartesian state/action.
        task: Single task string applied to all episodes (for flat layouts).
        task_map: Path to a JSON file mapping subdirectory names to task strings (for nested layouts).
                  If both --task and --task-map are given, --task-map takes priority and --task is used
                  as a fallback for subdirectories not found in the map.
        physics_map: Path to a JSON file mapping dataset names to physics CLI flags
                     (e.g. "", "--no-tactile", "--no-torque", "--no-tactile --no-torque").
                     If provided, overrides --no-tactile / --no-torque flags.
        no_tactile: If True, skip loading tactile data. Overridden by --physics-map if provided.
        no_torque: If True, skip loading torque data. Overridden by --physics-map if provided.
    """
    assert raw_dir.exists(), f"{raw_dir} does not exist"
    assert task is not None or task_map is not None, "Either --task or --task-map must be provided."

    # load task_map JSON if provided
    task_map_dict: Optional[dict[str, str]] = None
    if task_map is not None:
        with open(task_map) as f:
            task_map_dict = json.load(f)
        print(f"[INFO] Loaded task_map with {len(task_map_dict)} entries")

    # load physics_map JSON if provided
    physics_map_dict: Optional[dict[str, str]] = None
    if physics_map is not None:
        with open(physics_map) as f:
            physics_map_dict = json.load(f)
        print(f"[INFO] Loaded physics_map with {len(physics_map_dict)} entries")

    has_tactile, has_torque = _resolve_physics(raw_dir, physics_map_dict, no_tactile, no_torque)
    print(f"[INFO] Physics: has_tactile={has_tactile}, has_torque={has_torque}")

    # discover h5 files (supports both flat and nested directory layouts)
    h5_files = sorted(raw_dir.rglob("teleoperation.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No teleoperation.h5 found under {raw_dir}")

    if episodes is not None:
        h5_files = [h5_files[i] for i in episodes]

    # probe first episode for cameras, fps, image shape
    with h5py.File(h5_files[0], "r") as ep0:
        cameras = _read_camera_names(ep0)

        if cameras:
            first_cam_bytes = ep0[f"/observation/videos/{cameras[0]}"][()]
            first_frames = _decode_video_bytes_to_frames(first_cam_bytes)
            H, W = _probe_image_shape(first_frames)
        else:
            H, W = 480, 640

        if fps is None:
            fps_est = _estimate_fps_from_timestamps(ep0)
            fps = fps_est if fps_est is not None else 30

    # velocity is available by default in joint mode; disabled by default in eef mode
    has_velocity = action_type == "joint"

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type=robot_type,
        output_path=output_path,
        cameras=cameras,
        image_hw=(H, W),
        mode=mode,
        action_type=action_type,
        has_velocity=has_velocity,
        has_tactile=has_tactile,
        has_torque=has_torque,
        dataset_config=dataset_config,
        fps=fps,
    )

    dataset = populate_dataset(
        dataset, h5_files, raw_dir=raw_dir, task=task, task_map=task_map_dict,
        cameras=cameras, action_type=action_type,
        has_velocity=has_velocity, has_tactile=has_tactile, has_torque=has_torque,
    )

    data_root = output_path / "data"
    _patch_dataset_parquet(data_root)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(convert_h5_dir_to_lerobot)
