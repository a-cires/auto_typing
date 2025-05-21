import can
import time

# Constants
DEVICE_ID = 0x02  # Change to match your Talon SRX CAN ID
ENCODER_TICKS_PER_REV = 4096

# Talon SRX CAN arbitration ID constants
TALON_API_SET = 0x0400  # Set demand (control mode 0x1 is position)
TALON_API_CONTROL = 0x0200  # Control Frame for enabling the controller

def ticks_for_revolutions(revolutions):
    return int(revolutions * ENCODER_TICKS_PER_REV)

def build_control_frame(device_id):
    # Send a control frame to enable motor output
    arbitration_id = 0x02040000 | (device_id << 16)
    data = bytearray([0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    return can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True)

def build_position_command(device_id, position_ticks):
    arbitration_id = 0x04000000 | (device_id << 16)
    demand = position_ticks.to_bytes(4, byteorder='little', signed=True)
    data = bytearray([0x01]) + demand + bytearray(3)  # Mode 0x01 = Position
    return can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=True)

def send_motor_command(revolutions, can_interface='can0'):
    with can.interface.Bus(can_interface, bustype='socketcan') as bus:
        ticks = ticks_for_revolutions(revolutions)
        
        control_msg = build_control_frame(DEVICE_ID)
        position_msg = build_position_command(DEVICE_ID, ticks)

        print(f"[INFO] Sending control enable frame to Talon SRX ID {DEVICE_ID}")
        bus.send(control_msg)
        time.sleep(0.1)

        print(f"[INFO] Sending position command: {revolutions} rev -> {ticks} ticks")
        bus.send(position_msg)

        print("[INFO] Command sent.")

if __name__ == "__main__":
    revs = float(input("Enter number of revolutions: "))
    send_motor_command(revs)
