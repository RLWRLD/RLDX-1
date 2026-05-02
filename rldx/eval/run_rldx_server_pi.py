from dataclasses import dataclass
import os
import socket
import warnings

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.eval.serving import websocket_policy_server
from rldx.policy.rldx_policy import RLDXPolicy
import tyro


warnings.simplefilter("ignore", category=FutureWarning)


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    port: int = 5555
    """Port to connect to."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.
    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See rldx/experiment/data_config.py for more details.
    """

    embodiment_tag: EmbodimentTag = EmbodimentTag.GENERAL_EMBODIMENT
    """Embodiment tag"""

    model_path: str = None
    """Path to the model checkpoint."""

    concat_frames: bool = False
    """If True, concatenate multiple views horizontally before passing to backbone."""

    sample_timestep_from_beta_dist: bool = False
    """Whether to sample timestep from beta distribution. If False, sample uniformly."""


def main(args: ArgsConfig):
    # check if the model path exists
    if args.model_path.startswith("/") and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} does not exist")

    # Create and start the server
    if args.model_path is not None:
        policy = RLDXPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=args.model_path,
            device="cuda",
            strict=True,
            sample_timestep_from_beta_dist=args.sample_timestep_from_beta_dist,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("Creating server (host: %s, ip: %s)", hostname, local_ip)
    print(f"policy path: {args.model_path}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=None,
    )
    server.serve_forever()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
