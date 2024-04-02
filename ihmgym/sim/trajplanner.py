""" Implements trajectory planner """


class TrajPlanner:
    """
    Trajectory planner to compute intermediate setpoints from initial to
    target setpoint. Essentially the controller interpolates between
    initial and target setpoint using one of the "step", "rectangle" or
    "trapezoid" velocity profile.

            control = interpolate(initial, setpoint, time)

    NOTE: (1) The implementation is not vectorized and hence we need to
    one controller instance per actuator.

          (2) Trapezoid profile notation:

                        _______v_______
                       /               \
                    a /                 \
            _________/                   \____________
                    0   t1           t2  t3

    - To start planning a new trajectory from current value to target
    value, first set the current and target values using set_initial()
    and set_setpoint().
    - To compute intermediate setpoints, first call step(). Call
    get_control() to retrieve it thereafter.

    """

    def __init__(
        self,
        *,
        type: str = "velocity_based",
        frequency: int,
        profile: str = "trapezoid",
        vel: float = None,
        acc: float = None,
        t_total: float = None,
        acc_frac: float = 0.25,
    ) -> None:
        """
        Args:
            type:
            frequency: Frequency of the controller in Hz
            profile: 'step' or 'rectangle' or 'trapezoid'
            vel: Velocity of the profile, required when profile
                is 'rectange' or 'trapezoid'
            acc: Acceleration of the profile

        """

        self._frequency = frequency
        self._type = type
        self._profile = profile
        self._vel = vel
        self._acc = acc
        self._t_total = t_total
        self._acc_frac = acc_frac
        self._xi = None
        self._xf = None
        self._step = 0

        self._dir = 0
        self._control = 0

    @property
    def frequency(self) -> int:
        """Return frequency"""
        return self._frequency

    def plan(self, initial, setpoint) -> None:
        """Update the setpoint and compute profile parameters are
        required"""
        self._step = 0
        self._xi = initial
        self._xf = setpoint

        # Find direction
        if abs(self._xf - self._xi) > 0:
            self._dir = (self._xf - self._xi) / abs(self._xf - self._xi)
        else:
            self._dir = 0

        # Compute profile parameters
        if self._profile == "step":
            pass  # Nothing to do
        elif self._profile == "rectangle":
            # Compute the profile duration in steps
            if self._type == "velocity_based":
                self._t_total = abs(self._xf - self._xi) / self._vel
            elif self._type == "time_based":
                self._vel = abs(self._xf - self._xi) / self._t_total
        elif self._profile == "trapezoid":
            if self._type == "velocity_based":
                self._t1 = self._vel / self._acc
                self._t2 = abs(self._xf - self._xi) / self._vel
                self._t3 = self._t1 + self._t2
            elif self._type == "time_based":
                self._t1 = self._acc_frac * self._t_total
                self._t2 = self._t_total * (1 - self._acc_frac)
                self._t3 = self._t_total
                self._vel = abs(self._xf - self._xi) / self._t2
                self._acc = self._vel / self._t1

    def get_setpoint(self) -> float:
        return self._xf

    def get_control(self) -> float:
        """Returns control compute during step()"""
        return self._control

    def step(self) -> None:
        """Compute and return control at time t based on profile
        parameters"""
        self._step += 1
        t = self._step / self._frequency

        if self._profile == "step":
            control = self._xf
        if self._profile == "rectangle":
            vel = self._dir * self._vel
            if t < self._t_total:
                control = self._xi + vel * t
            else:
                control = self._xi + vel * self._t_total
        if self._profile == "trapezoid":
            vel = self._dir * self._vel
            acc = self._dir * self._acc
            if 0 <= t < self._t1:
                control = self._xi + 0.5 * acc * t * t
            elif self._t1 <= t < self._t2:
                control = self._xi + 0.5 * acc * self._t1 * self._t1 + (t - self._t1) * vel
            elif self._t2 <= t <= self._t3:
                control = (
                    self._xi
                    + 0.5 * acc * self._t1 * self._t1
                    + (self._t2 - self._t1) * vel
                    + vel * (t - self._t2)
                    + 0.5 * (-1.0 * acc) * (t - self._t2) * (t - self._t2)
                )
            elif self._t3 < t:
                control = (
                    self._xi
                    + 0.5 * acc * self._t1 * self._t1
                    + (self._t2 - self._t1) * vel
                    + vel * (self._t3 - self._t2)
                    + 0.5 * (-1.0 * acc) * (self._t3 - self._t2) * (self._t3 - self._t2)
                )
        self._control = control