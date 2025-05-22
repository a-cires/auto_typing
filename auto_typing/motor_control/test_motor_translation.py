
from auto_typing.motor_control import motor_cpp
from auto_typing.utils.timing import run_for_duration

def test_motor_translation(channel=2, duty=-1.0, duration_sec=1, rate_hz=20):
    controller = motor_cpp.MotorController(channel)
    print("Initializing motor...")
    controller.init()

    def run_motor():
        controller.run(duty, 50)  # 50ms feed_forward

    run_for_duration(duration_sec, run_motor, rate_hz=rate_hz)

if __name__ == "__main__":
    test_motor_translation()
