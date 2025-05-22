#pragma once

#define Phoenix_No_WPI
#include "ctre/Phoenix.h"

#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

class MotorController {
public:
    explicit MotorController(int can_id);  // Constructor with CAN ID
    void init();                           // Initialize the motor config
    void setSpeed(double dutyCycle);       // Set motor output [-1.0, 1.0]
    void run(double speed, int duration_ms); // Set speed and feed enable

private:
    TalonSRX motor;
};

void sendCanFrame(const std::string& ifname, canid_t can_id, uint8_t data[8], uint8_t dlc);
