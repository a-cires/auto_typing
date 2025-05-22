
from auto_typing.motor_control.distance_controller import DistanceMotorController

class PositionAwareMotor:
    def __init__(self, initial_position_mm=0.0, channel=2):
        self.current_position = initial_position_mm
        self.motor = DistanceMotorController(channel=channel)

    def move_relative(self, direction, distance_mm, duty=1.0):
        if direction == 'left':
            self.current_position += distance_mm
        elif direction == 'right':
            self.current_position -= distance_mm
        else:
            raise ValueError("Direction must be 'left' or 'right'")

        self.motor.move(direction, distance_mm, duty=duty)

    def move_to(self, target_position_mm, duty=1.0):
        delta = target_position_mm - self.current_position
        if abs(delta) < 1e-2:
            print("🟢 Already at target position.")
            return
        direction = 'left' if delta > 0 else 'right'
        self.move_relative(direction, abs(delta), duty=duty)

    def get_position(self):
        return self.current_position
