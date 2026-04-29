from robocasa.environments.tabletop.tabletop import *


class TabletopObjectShowcase(Tabletop):
    """
    Class for displaying a single object on the tabletop.

    Args:
        obj_groups (str): Object groups to sample the target object from.
        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """

    VALID_LAYOUTS = [0]

    def __init__(self, obj_groups="all", exclude_obj_groups=None, *args, **kwargs):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        super().__init__(*args, **kwargs)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"observe the {obj_lang}. the task cannot be completed."
        return ep_meta

    def _setup_table_references(self):
        super()._setup_table_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55))
        )
        self.init_robot_base_pos = self.counter

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 1),
                    pos=(0.0, 0.0),  # center of table
                ),
            )
        )
        return cfgs

    def _check_success(self):
        return False
