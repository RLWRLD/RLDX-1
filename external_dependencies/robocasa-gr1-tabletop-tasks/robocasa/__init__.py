from robosuite.environments.base import make

# Manipulation environments
from robocasa.environments.tabletop.tabletop import Tabletop
from robocasa.environments.tabletop.tabletop_pnp import (
    PnPOnionToBowl,
    PnPCanToBowl,
    PnPCupToPlate,
    PnPAppleToPlate,
    PnPMilkToBasket,
    PnPKettleToPlate,
    PnPFruitToPlacemat,
    PnPCounterToPlate,
    PnPCounterToBowl,
    PnPCounterToCuttingBoard,
    PnPCounterToPot,
    PnPCounterToPan,
    PnPPlateToPlate,
    PnPMilkPlateToPlate,
    PnPVegetableBowlToPlate,
    PnPObjectsToShelf,
    PnPObjectsToShelfLevel,
    PnPObjectsShelfToCounter,
    PnPObjectsShelfLevelToLevel,
    PnPObjectsToTieredBasket,
    PnPObjectsToTieredBasketLevel,
    PnPObjectsTieredBasketToCounter,
    PnPObjectsTieredBasketLevelToLevel,
    PnPRubixCubeBasketToCounter,
    PnPCupToPlateNoDistractors,
    PnPCupToDishRackUpperLevel,
    PnPBreadBasketToBowl,
    PnPPouring,
    PnPFruitToPlate,
    PnPFruitToPlateSplitA,
    PnPFruitToPlateSplitB,
    PnPCylindricalToPlate,
    PnPMilkPlateToPlateCotrain,
    PnPAppleToPlateCotrain,
)
from robocasa.environments.tabletop.tabletop_cabinet_door import (
    TabletopCabinetDoor,
    TabletopOpenCabinetDoor,
    TabletopCloseCabinetDoor,
)
from robocasa.environments.tabletop.tabletop_drawer_door import (
    TabletopDrawerDoor,
    TabletopOpenDrawerDoor,
    TabletopCloseDrawerDoor,
)
from robocasa.environments.tabletop.tabletop_drawer_pnp import (
    PnPCupToDrawerClose,
    PnPAppleToDrawerClose,
    PnPBottleToDrawerClose,
    PnPCanToDrawerClose,
    PnPWineToDrawerClose,
)
from robocasa.environments.tabletop.tabletop_microwave_pnp import (
    PnPCupToMicrowaveClose,
    PnPCornToMicrowaveClose,
    PnPPotatoToMicrowaveClose,
    PnPEggplantToMicrowaveClose,
    PnPMilkToMicrowaveClose,
)
from robocasa.environments.tabletop.tabletop_cabinet_pnp import (
    PnPCupToCabinetClose,
    PnPAppleToCabinetClose,
    PnPBottleToCabinetClose,
    PnPCanToCabinetClose,
    PnPWineToCabinetClose,
)
from robocasa.environments.tabletop.tabletop_microwave import (
    TabletopTurnOffMicrowave,
    TabletopTurnOnMicrowave,
)
from robocasa.environments.tabletop.tabletop_microwave_door import (
    TabletopOpenMicrowaveDoor,
    TabletopCloseMicrowaveDoor,
)
from robocasa.environments.tabletop.tabletop_multi_pnp import (
    PutAllObjectsInBasket,
    PutAllObjectsOnPlate,
)
from robocasa.environments.tabletop.tabletop_object_showcase import (
    TabletopObjectShowcase,
)
from robocasa.environments.tabletop.tabletop_5dc import *
from robocasa.environments.tabletop.tabletop_24dc import *
from robocasa.environments.tabletop.tabletop_laptop import (
    TabletopLaptopOpen,
    TabletopLaptopClose,
)

try:
    import mimicgen
except ImportError:
    print(
        "WARNING: mimicgen environments not imported since mimicgen is not installed!"
    )

# from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.controllers import ALL_PART_CONTROLLERS, load_composite_controller_config
from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.models.grippers import ALL_GRIPPERS
from robosuite.robots import ALL_ROBOTS

import mujoco

assert (
    mujoco.__version__ == "3.2.6"
), "MuJoCo version must be 3.2.6. Please run pip install mujoco==3.2.6"

import numpy

assert numpy.__version__ in [
    "1.23.2",
    "1.23.3",
    "1.23.5",
    "1.26.4",
], "numpy version must be either 1.23.{2,3,5} or 1.26.4. Please install one of these versions."

import robosuite

assert robosuite.__version__ in [
    "1.5.0",
    "1.5.1",
], "robosuite version must be 1.5.{0,1}. Please install the correct version"

__version__ = "0.2.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
