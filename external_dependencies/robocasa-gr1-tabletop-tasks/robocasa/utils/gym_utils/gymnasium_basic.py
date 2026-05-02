import datetime, uuid
from copy import deepcopy
import gymnasium as gym
import numpy as np
import os
import robocasa  # we need this to register environments  # noqa: F401
import robosuite
from gymnasium import spaces
from robocasa.environments.tabletop.tabletop import Tabletop
from robocasa.models.robots import (
    GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY,
    GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST,
    GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY,
    gather_robot_observations,
    make_key_converter,
)
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.arm.osc import OperationalSpaceController
from robosuite.controllers.composite.composite_controller import HybridMobileBase
from robosuite.environments.base import REGISTERED_ENVS


ALLOWED_LANGUAGE_CHARSET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"
)


def create_env_robosuite(
    env_name,
    # robosuite-related configs
    robots="PandaOmron",
    controller_configs=None,
    camera_names=[
        "egoview",
        "robot0_eye_in_left_hand",
        "robot0_eye_in_right_hand",
    ],
    camera_widths=128,
    camera_heights=128,
    enable_render=True,
    seed=None,
    # robocasa-related configs
    obj_instance_split=None,
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=None,
    layout_ids=None,
    style_ids=None,
):
    if controller_configs is None:
        controller_configs = load_composite_controller_config(
            controller=None,
            robot=robots if isinstance(robots, str) else robots[0],
        )
    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=enable_render,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=enable_render,
        camera_depths=False,
        seed=seed,
        translucent_robot=False,
    )
    env_class = REGISTERED_ENVS[env_name]

    env = robosuite.make(**env_kwargs)
    return env, env_kwargs


class RoboCasaEnv(gym.Env):
    def __init__(
        self,
        env_name=None,
        robots_name=None,
        camera_names=None,
        camera_widths=None,
        camera_heights=None,
        enable_render=True,
        dump_rollout_dataset_dir=None,
        **kwargs,  # Accept additional kwargs
    ):
        self.key_converter = make_key_converter(robots_name)
        (
            _,
            camera_names,
            default_camera_widths,
            default_camera_heights,
        ) = self.key_converter.get_camera_config()

        if camera_widths is None:
            camera_widths = default_camera_widths
        if camera_heights is None:
            camera_heights = default_camera_heights

        controller_configs = load_composite_controller_config(
            controller=None,
            robot=robots_name.split("_")[0],
        )
        if (
            robots_name in GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY
            or robots_name in GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST
            or robots_name in GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY
        ):
            controller_configs["type"] = "BASIC"
            controller_configs["composite_controller_specific_configs"] = {}
            controller_configs["control_delta"] = False

        self.env, self.env_kwargs = create_env_robosuite(
            env_name=env_name,
            robots=robots_name.split("_"),
            controller_configs=controller_configs,
            camera_names=camera_names,
            camera_widths=camera_widths,
            camera_heights=camera_heights,
            enable_render=enable_render,
            **kwargs,  # Forward kwargs to create_env_robosuite
        )

        # TODO: the following info should be output by grootrobocasa
        self.camera_names = camera_names
        self.camera_widths = camera_widths
        self.camera_heights = camera_heights
        self.enable_render = enable_render
        self.render_obs_key = f"{camera_names[0]}_image"
        self.render_cache = None

        # setup spaces
        action_space = spaces.Dict()
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            for part_name, controller in cc.part_controllers.items():
                min_value, max_value = -1, 1
                start_idx, end_idx = cc._action_split_indexes[part_name]
                shape = [end_idx - start_idx]
                this_space = spaces.Box(
                    low=min_value, high=max_value, shape=shape, dtype=np.float32
                )
                action_space[f"{pf}{part_name}"] = this_space
            if isinstance(cc, HybridMobileBase):
                this_space = spaces.Discrete(2)
                action_space[f"{pf}base_mode"] = this_space

            action_space = spaces.Dict(action_space)
            self.action_space = action_space

        obs = (
            self.env.viewer._get_observations(force_update=True)
            if self.env.viewer_get_obs
            else self.env._get_observations(force_update=True)
        )
        obs.update(gather_robot_observations(self.env))
        observation_space = spaces.Dict()
        for obs_name, obs_value in obs.items():
            shape = list(obs_value.shape)
            if obs_name.endswith("_image"):
                continue
            min_value, max_value = -1, 1
            this_space = spaces.Box(
                low=min_value, high=max_value, shape=shape, dtype=np.float32
            )
            observation_space[obs_name] = this_space

        for camera_name in camera_names:
            shape = [camera_heights, camera_widths, 3]
            this_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
            observation_space[f"{camera_name}_image"] = this_space

        observation_space["language"] = spaces.Text(
            max_length=256, charset=ALLOWED_LANGUAGE_CHARSET
        )

        self.observation_space = observation_space

        self.dump_rollout_dataset_dir = dump_rollout_dataset_dir
        self.groot_exporter = None
        self.np_exporter = None

    def get_basic_observation(self, raw_obs):
        raw_obs.update(gather_robot_observations(self.env))

        # Image are in (H, W, C), flip it upside down
        def process_img(img):
            return np.copy(img[::-1, :, :])

        for obs_name, obs_value in raw_obs.items():
            if obs_name.endswith("_image"):
                # image observations
                raw_obs[obs_name] = process_img(obs_value)
            else:
                # non-image observations
                raw_obs[obs_name] = obs_value.astype(np.float32)

        # Return black image if rendering is disabled
        if not self.enable_render:
            for name in self.camera_names:
                raw_obs[f"{name}_image"] = np.zeros(
                    (self.camera_heights, self.camera_widths, 3), dtype=np.uint8
                )

        self.render_cache = raw_obs[self.render_obs_key]
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")

        return raw_obs

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        raw_obs = self.env.reset()
        # return obs
        obs = self.get_basic_observation(raw_obs)

        info = {}
        info["success"] = False
        info["grasp_distractor_obj"] = False

        return obs, info

    def step(self, action_dict):
        env_action = []
        for robot in self.env.robots:
            cc = robot.composite_controller
            pf = robot.robot_model.naming_prefix
            action = np.zeros(cc.action_limits[0].shape)
            for part_name, controller in cc.part_controllers.items():
                start_idx, end_idx = cc._action_split_indexes[part_name]
                act = action_dict.pop(f"{pf}{part_name}")
                action[start_idx:end_idx] = act
            if isinstance(cc, HybridMobileBase):
                action[-1] = action_dict.pop(f"{pf}base_mode")
            env_action.append(action)

        assert len(action_dict) == 0, f"Unprocessed actions: {action_dict}"
        env_action = np.concatenate(env_action)

        raw_obs, reward, done, info = self.env.step(env_action)

        obs = self.get_basic_observation(raw_obs)

        truncated = False

        info["success"] = reward > 0
        info["grasp_distractor_obj"] = False
        if hasattr(self, "_check_grasp_distractor_obj"):
            info["grasp_distractor_obj"] = self._check_grasp_distractor_obj()

        return obs, reward, done, truncated, info

    def render(self):
        if self.render_cache is None:
            raise RuntimeError("Must run reset or step before render.")
        return self.render_cache

    def close(self):
        self.env.close()
