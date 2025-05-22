
from auto_typing.motor_control import motor_cpp
from auto_typing.motor_control.motion_model import MotorMotionModel
from auto_typing.utils.timing import run_for_duration

class DistanceMotorController:
    def __init__(self, channel=2, default_feedforward=50):
        self.controller = motor_cpp.MotorController(channel)
        self.controller.init()
        self.model = MotorMotionModel()
        self.feedforward = default_feedforward  # ms

    def move(self, direction: str, distance_mm: float, duty=1.0, rate_hz=20):
        if direction not in ('left', 'right'):
            raise ValueError("Direction must be 'left' or 'right'")

        # Adjust duty sign based on convention
        duty = -abs(duty) if direction == 'left' else abs(duty)

        duration = self.model.estimate_time_to_travel(direction, distance_mm)
        print(f"🔁 Moving {direction} for {duration:.2f} seconds to cover ~{distance_mm}mm")

        def command():
            self.controller.run(duty, self.feedforward)

        run_for_duration(duration, command, rate_hz=rate_hz)
