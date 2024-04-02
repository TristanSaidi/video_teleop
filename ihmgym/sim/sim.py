""" Base class for simulation """

from collections import namedtuple
from typing import Callable, Union

import glfw
import mujoco_py
import numpy as np
from logger import getlogger

logger = getlogger(__name__)

# __all__ = ["Sim", "State"]

State = namedtuple(
    "State",
    [
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
        "ctrl",
    ],
)


class Sim:

    """
    Base class that wraps around MuJoCo simulation objects and implements commonly used fuctionality
    such as stepping, rendering and callbacks.
    """

    def __init__(
        self,
        *,
        model: Union[str, bytes],
        state: State = None,
    ) -> None:
        """
        Simulation from xml string or bytes and optionally set state
        """
        if isinstance(model, str):
            self._model = mujoco_py.load_model_from_xml(model)
        elif isinstance(model, bytes):
            self._model = mujoco_py.load_model_from_mjb(model)
        self._sim = mujoco_py.MjSim(self._model)
        if state is not None:
            self.set_state(state)
        self._step = 0

        # Rendering
        self._viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        # Callbacks
        self._callbacks = []

        # Set flag to run self._sim.forward() before next call to
        # advance()
        self._sim_data_dirty = False

        logger.info("Initialized sim")

    @property
    def dt(self) -> None:
        """Return simulation timestep"""
        return self._sim.model.opt.timestep * self._sim.nsubsteps

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._sim.data

    def advance(self, dt: float) -> None:
        """
        Advance the simulation by dt and call the registered callback
        methods
        """
        if self._sim_data_dirty:
            self._sim.forward()
            self._sim_data_dirty = False
        for _ in range(int(np.ceil(dt / self.dt))):
            # Step the simulation
            self._sim.step()
            # Callbacks
            if hasattr(self, "_callbacks"):
                for (callback, period) in self._callbacks:
                    if self._step % period == 0:
                        callback()
            self._step += 1

    def step(self) -> None:
        """Step the simulation by dt"""
        self.advance(self.dt)

    def set_ctrl(self, ctrl: np.ndarray = None):
        """Set control"""
        self._sim.data.ctrl[:] = ctrl[:]

    def render(self, mode="human", width=500, height=500):
        """Render simulation"""
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self.register_callback(
                self._get_viewer(mode).render, np.ceil(int(1 / self.dt))
            )

    def close(self) -> None:
        """closes resources like viewer used for rendering"""
        del self._sim
        del self._model
        if self._viewer is not None:
            glfw.destroy_window(self._viewer.window)
            self._viewer = None
            self._viewers = {}
        logger.info("closing viewer")

    def register_callback(self, callback: Callable[[], None], frequency: int) -> None:
        """Register callback method

        Register callback method to be called with the specified
        frequency
        """
        period = int(1 / (frequency * self.dt))
        assert period > 0
        if (callback, period) not in self._callbacks:
            self._callbacks.append((callback, period))

    def _get_viewer(
        self, mode: str
    ) -> mujoco_py.MjViewer or mujoco_py.MjRenderContextOffscreen:
        self._viewer = self._viewers.get(mode)
        if self._viewer is None:
            if mode == "human":
                self._viewer = mujoco_py.MjViewer(self._sim)
                self._viewer._paused = True
            elif mode == "rgb_array":
                self._viewer = mujoco_py.MjRenderContextOffscreen(self._sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self._viewer
        return self._viewer

    def _viewer_setup(self) -> None:
        """Setup viewer camera"""
        pass
        # body_id = self.sim.model.body_name2id("world")
        # lookat = self.sim.data.body_xpos[body_id]
        # for idx, value in enumerate(lookat):
        #     self._viewer.cam.lookat[idx] = value
        # self._viewer.cam.distance = 0.5
        # self._viewer.cam.azimuth = 55.0
        # self._viewer.cam.elevation = -25.0

    # Sim state:
    # MuJoCo provides getstate and setstate methods where the state
    # consists only of the dynamics state i.e qpos and qvel without
    # capturing actuation state or external forces. The following
    # methods deal with the full state of the simulator
    def get_state(self) -> dict:
        """Extract and return simulation state"""
        return State(*[getattr(self._sim.data, key) for key in State._fields])

    def set_state(self, state: State) -> None:
        """Set simulation state"""
        for i in range(self._model.nq):
            self._sim.data.qpos[i] = state.qpos[i]
        for i in range(self._model.nv):
            self._sim.data.qvel[i] = state.qvel[i]
            self._sim.data.qacc[i] = state.qacc[i]
            self._sim.data.qacc_warmstart[i] = state.qacc_warmstart[i]
            self._sim.data.qfrc_applied[i] = state.qfrc_applied[i]
        for i in range(self._model.nbody):
            self._sim.data.xfrc_applied[i] = state.xfrc_applied[i]
        for i in range(self._model.nu):
            self._sim.data.ctrl[i] = state.ctrl[i]
        self._sim.forward()

    def __getstate__(self):
        return {
            "model": self._model.get_mjb(),
            "state": self.get_state(),
        }

    def __setstate__(self, state: dict):
        self.__init__(**state)