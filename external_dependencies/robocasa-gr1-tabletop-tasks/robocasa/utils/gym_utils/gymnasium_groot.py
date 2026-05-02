import sys
from typing import Any, Dict

import cv2
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from robocasa.models.robots import GROOT_ROBOCASA_ENVS_ROBOTS
from robocasa.models.robots.manipulators.gr1_robot import GR1ArmsOnly, GR1ArmsAndWaist
from .gymnasium_basic import (
    REGISTERED_ENVS,
    RoboCasaEnv,
)

ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)
FINAL_IMAGE_RESOLUTION = (256, 256)


class GrootRoboCasaEnv(RoboCasaEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.key_converter.deduce_observation_space(self.env)
        mapped_names, _, _, _ = self.key_converter.get_camera_config()
        for mapped_name in mapped_names:
            self.observation_space[mapped_name] = spaces.Box(
                low=0, high=255, shape=(*FINAL_IMAGE_RESOLUTION, 3), dtype=np.uint8
            )
            if mapped_name == "video.ego_view_pad_res256_freq20":
                self.observation_space[
                    "video.ego_view_bg_crop_pad_res256_freq20"
                ] = spaces.Box(
                    low=0, high=255, shape=(*FINAL_IMAGE_RESOLUTION, 3), dtype=np.uint8
                )
        if isinstance(self.env.robots[0].robot_model, GR1ArmsOnly):
            self.observation_space["annotation.human.coarse_action"] = spaces.Text(
                max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
            )
        elif isinstance(self.env.robots[0].robot_model, GR1ArmsAndWaist):
            self.observation_space["annotation.human.coarse_action"] = spaces.Text(
                max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
            )
        else:
            self.observation_space[
                "annotation.human.action.task_description"
            ] = spaces.Text(max_length=256, charset=ALLOWED_LANGUAGE_CHARSET)
        self.action_space = self.key_converter.deduce_action_space(self.env)

        self.verbose = False
        for k, v in self.observation_space.items():
            self.verbose and print("{OBS}", k, v)
        for k, v in self.action_space.items():
            self.verbose and print("{ACTION}", k, v)

    @staticmethod
    def process_img(img):
        h, w, _ = img.shape
        if h != w:
            dim = max(h, w)
            y_offset = (dim - h) // 2
            x_offset = (dim - w) // 2
            img = np.pad(img, ((y_offset, y_offset), (x_offset, x_offset), (0, 0)))
            h, w = dim, dim
        if (h, w) != FINAL_IMAGE_RESOLUTION:
            img = cv2.resize(img, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA)
        return np.copy(img)

    @staticmethod
    def process_img_cotrain(img):
        assert img.shape[0] == 800 and img.shape[1] == 1280

        oh, ow = 256, 256
        crop = (310, 770, 110, 1130)
        img = img[crop[0] : crop[1], crop[2] : crop[3]]

        img_resized = cv2.resize(img, (720, 480), cv2.INTER_AREA)
        width_pad = (img_resized.shape[1] - img_resized.shape[0]) // 2
        img_pad = np.pad(
            img_resized,
            ((width_pad, width_pad), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        img_resized = cv2.resize(img_pad, (oh, ow), cv2.INTER_AREA)
        return img_resized

    def get_groot_observation(self, raw_obs):
        obs = {}
        temp_obs = self.key_converter.map_obs(raw_obs)
        for k, v in temp_obs.items():
            if k.startswith("hand.") or k.startswith("body."):
                obs["state." + k[5:]] = v
            else:
                raise ValueError(f"Unknown key: {k}")
        mapped_names, camera_names, _, _ = self.key_converter.get_camera_config()
        for mapped_name, camera_name in zip(mapped_names, camera_names):
            obs[mapped_name] = GrootRoboCasaEnv.process_img(
                raw_obs[camera_name + "_image"]
            )
            if mapped_name == "video.ego_view_pad_res256_freq20":
                obs[
                    "video.ego_view_bg_crop_pad_res256_freq20"
                ] = GrootRoboCasaEnv.process_img_cotrain(
                    raw_obs[camera_name + "_image"]
                )
        if isinstance(self.env.robots[0].robot_model, GR1ArmsOnly):
            obs[
                "annotation.human.coarse_action"
            ] = f"locked_waist: {raw_obs['language']}"
        elif isinstance(self.env.robots[0].robot_model, GR1ArmsAndWaist):
            obs[
                "annotation.human.coarse_action"
            ] = f"unlocked_waist: {raw_obs['language']}"
        else:
            obs["annotation.human.action.task_description"] = raw_obs["language"]
        return obs

    def reset(self, seed=None, options=None):
        raw_obs, info = super().reset(seed=seed, options=options)
        obs = self.get_groot_observation(raw_obs)
        return obs, info

    def step(self, action):
        for k, v in action.items():
            self.verbose and print("<ACTION>", k, v)

        action = self.key_converter.unmap_action(action)
        raw_obs, reward, terminated, truncated, info = super().step(action)
        obs = self.get_groot_observation(raw_obs)

        for k, v in obs.items():
            self.verbose and print("<OBS>", k, v.shape if k.startswith("video.") else v)
        self.verbose = False

        return obs, reward, terminated, truncated, info


def create_grootrobocasa_env_class(env, robot, robot_alias):
    class_name = f"{env}_{robot}_Env"
    id_name = f"robocasa_{robot_alias}/{class_name}"

    env_class_type = type(
        class_name,
        (GrootRoboCasaEnv,),
        {
            "__init__": lambda self, **kwargs: super(self.__class__, self).__init__(
                env_name=env,
                robots_name=robot,
                **kwargs,
            )
        },
    )

    current_module = sys.modules["robocasa.utils.gym_utils.gymnasium_groot"]
    setattr(current_module, class_name, env_class_type)
    register(
        id=id_name,  # Unique ID for the environment
        entry_point=f"robocasa.utils.gym_utils.gymnasium_groot:{class_name}",  # Path to your environment class
    )

    if robot_alias == "gr1_arms_waist_fourier_hands":
        id_name = f"gr1_unified/{class_name}"
        register(
            id=id_name,  # Unique ID for the environment
            entry_point=f"robocasa.utils.gym_utils.gymnasium_groot:{class_name}",  # Path to your environment class
        )


for ENV in REGISTERED_ENVS:
    for ROBOT, ROBOT_ALIAS in GROOT_ROBOCASA_ENVS_ROBOTS.items():
        create_grootrobocasa_env_class(ENV, ROBOT, ROBOT_ALIAS)
