import os
import sys
import gymnasium as gym
import numpy as np
import pytest
from pathlib import Path
from gymnasium import spaces

import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)


def demo_robocasa_digital_cousin_env(env_id, video_dir):
    env = gym.make(env_id, enable_render=True)
    video_recorder = VideoRecorder.create_h264(
        fps=10,
        codec="h264",
        input_pix_fmt="rgb24",
        crf=22,
        thread_type="FRAME",
        thread_count=1,
    )
    env = VideoRecordingWrapper(
        env,
        video_recorder,
        video_dir=Path(video_dir),
    )

    # raw obs from env is in float64, convert to float32
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
    env.render()
    env.close()


if __name__ == "__main__":
    env_id = sys.argv[1]
    video_dir = os.path.join(os.getcwd(), "video")
    os.makedirs(video_dir, exist_ok=True)
    demo_robocasa_digital_cousin_env(env_id, video_dir=video_dir)
