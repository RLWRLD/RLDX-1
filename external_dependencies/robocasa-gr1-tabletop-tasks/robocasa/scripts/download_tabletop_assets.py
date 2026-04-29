import argparse

from download_groot_assets import (
    download_groot_assets,
    DownloadConfig,
    DEFAULT_ASSETS_VERSION,
    Registry,
)


def download_dc_assets(bypass: bool):
    cfg = DownloadConfig(registries={Registry.SKETCHFAB, Registry.LIGHTWHEEL})
    download_groot_assets(
        download_config=cfg,
        version=DEFAULT_ASSETS_VERSION,
        bypass=bypass,
        update_registry=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Download assets directly without prompting to screen",
    )
    args = parser.parse_args()
    download_dc_assets(args.yes)
