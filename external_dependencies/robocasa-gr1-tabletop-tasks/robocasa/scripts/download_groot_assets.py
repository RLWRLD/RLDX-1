import argparse
import dataclasses
import enum
import json
import logging
import os
from pathlib import Path
from typing import Optional
import urllib.request

from download_kitchen_assets import download_kitchen_assets as download

DEFAULT_ASSETS_VERSION = "1b018839a6da865dffecd3185fe054211bc71270"
REPO_ID = "nvidia/PhysicalAI-DigitalCousin-Assets"
API_URL = f"https://huggingface.co/api/datasets/{REPO_ID}"
SOURCE_URL = f"https://huggingface.co/datasets/{REPO_ID}"


class Registry(enum.Enum):
    SKETCHFAB = "sketchfab"
    LIGHTWHEEL = "lightwheel"


@dataclasses.dataclass(frozen=True)
class DownloadConfig:
    registries: set[Registry]

    @classmethod
    def from_args(cls, registry_names: Optional[list[str]] = None) -> "DownloadConfig":
        valid_registries = (
            {r for r in Registry if r.value in registry_names}
            if registry_names
            else set(Registry)
        )
        return cls(registries=valid_registries)


def get_available_registries(revision: str) -> set[Registry]:
    url = f"{API_URL}?revision={revision}"
    with urllib.request.urlopen(url) as resp:
        resp = json.load(resp)
        tree = resp["siblings"]
    stems = {Path(item["rfilename"]).stem for item in tree}
    return {r for r in Registry if r.value in stems}


def get_groot_asset_registry(
    download_config: DownloadConfig, version: Optional[str]
) -> dict:
    logging.info(
        f"Initiating download for the specified assets: {', '.join(r.value for r in download_config.registries)}..."
    )

    if version:
        logging.info(f"Initiating download for assets at revision '{version}'...")
    else:
        version = DEFAULT_ASSETS_VERSION
        logging.info(
            f"No revision specified; proceeding with the default version '{version}'..."
        )

    available_registries = get_available_registries(version)
    registry = {}
    for r in download_config.registries:
        if r not in available_registries:
            logging.warning(
                f"Requested registry '{r.value}' is not available at revision '{version}', skipping..."
            )
            continue
        registry[r.value] = dict(
            message=f"Downloading {r.value} assets",
            url=f"{SOURCE_URL}/resolve/{version}/{r.value}.zip",
            folder=os.path.join(
                os.path.dirname(__file__), f"../models/assets/objects/{r.value}"
            ),
            check_folder_exists=False,
        )
    return registry


def download_groot_assets(
    download_config: DownloadConfig,
    version: str,
    bypass: bool,
    update_registry: bool = False,
):
    import download_kitchen_assets

    groot_asset_registry = get_groot_asset_registry(
        download_config=download_config, version=version
    )
    if update_registry:
        download_kitchen_assets.DOWNLOAD_ASSET_REGISTRY.update(groot_asset_registry)
    else:
        download_kitchen_assets.DOWNLOAD_ASSET_REGISTRY = groot_asset_registry
    download(bypass=bypass)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Download assets directly without prompting to screen",
    )
    parser.add_argument(
        "-a",
        "--assets",
        nargs="*",
        help="Specify a list of asset names to download (leave empty to download all assets)",
    )
    parser.add_argument(
        "-r",
        "--revision",
        help="Specify the release tag or revision number of the assets to download",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    cfg = DownloadConfig.from_args(args.assets)

    download_groot_assets(
        download_config=cfg,
        version=args.revision,
        bypass=args.yes,
        update_registry=False,
    )
