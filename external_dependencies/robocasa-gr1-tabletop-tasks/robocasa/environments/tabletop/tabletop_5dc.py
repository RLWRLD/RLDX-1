import numpy as np
from typing import Tuple
from robocasa.environments.tabletop.tabletop_24dc import (
    PositionSampler as PositionSamplerOld,
)
from robocasa.environments.tabletop.tabletop_24dc import generate_task_classes

# generate tasks with a specific object category as target object
obj_categories = ["can", "apple", "cucumber", "lemon", "bottled_water"]
container_combos = [
    ("plate", "bowl"),
    ("placemat", "basket"),
    ("cutting_board", "basket"),
    ("tray", "plate"),
    ("cutting_board", "pan"),
]
dc5_tasks_info = generate_task_classes(
    obj_categories,
    container_combos,
    prefix="PnP5",
    randomize_distractor_configs=False,  # no distractor
    distractor_configs=None,
    postfix="NoDistractorSplitA",
)


if __name__ == "__main__":

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"  # Resets color to terminal default

    # print task names
    print(f"{GREEN}DC5 tasks: {len(dc5_tasks_info)} {RESET}")
    for task_info in dc5_tasks_info:
        print(task_info["class_name"])
