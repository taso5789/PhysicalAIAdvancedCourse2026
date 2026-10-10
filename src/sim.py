from pathlib import Path

import mujoco
import numpy as np

WORLD = Path(__file__).resolve().parents[1] / "assets" / "room.xml"


class Simulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(WORLD))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def drive(self, linear=0.0, angular=0.0):
        # Wheel radius 0.12 m; wheel separation 0.38 m.
        self.data.ctrl[:] = np.clip(
            [(linear - angular * 0.19) / 0.12,
             (linear + angular * 0.19) / 0.12], -10, 10
        )

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
