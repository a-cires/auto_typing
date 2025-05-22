from auto_typing.motor_control import motor_cpp
from time import sleep

controller = motor_cpp.MotorController(2)

print("Initializing motor...")
controller.init()
sleep(8)

# print("Spinning motor at 30%...")
# controller.set_speed(100)

print("Running motor at 50% for 1000 ms...")
controller.run(0.1, 5000)
sleep(6)