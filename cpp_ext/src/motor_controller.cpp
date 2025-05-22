#include "motor_control/motor_controller.hpp"
#include "ctre/phoenix/unmanaged/Unmanaged.h"
#include <cstdlib>
#include <unistd.h>

#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

void sendCanFrame(const std::string& ifname, canid_t can_id, uint8_t data[8], uint8_t dlc) {
    struct ifreq ifr {};
    struct sockaddr_can addr {};
    struct can_frame frame {};
    
    // Open a raw CAN socket
    int sock = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock < 0) {
        perror("Socket");
        return;
    }

    std::strncpy(ifr.ifr_name, ifname.c_str(), IFNAMSIZ);
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0) {
        perror("ioctl");
        close(sock);
        return;
    }

    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(sock);
        return;
    }

    frame.can_id = can_id;
    frame.can_dlc = dlc;
    std::memcpy(frame.data, data, dlc);

    if (write(sock, &frame, sizeof(frame)) != sizeof(frame)) {
        perror("write");
    }

    close(sock);
}


MotorController::MotorController(int can_id)
    : motor(can_id) {}

void MotorController::init() {
    // Sends a custom CAN frame (initialization kick)
    uint8_t zero_data[8] = {0};
    sendCanFrame("can0", 0x123, zero_data, 8);
    sleep(5);

    motor.ConfigFactoryDefault(100);
    motor.SetInverted(true);

    motor.ConfigSelectedFeedbackSensor(FeedbackDevice::QuadEncoder, 0, 100);
    motor.SetSensorPhase(true);

    motor.Config_kP(0, 10.0, 100);
    motor.Config_kD(0, 0.0, 100);
    motor.Config_kF(0, 0.0, 100);
}

void MotorController::setSpeed(double dutyCycle) {
    motor.Set(ControlMode::PercentOutput, dutyCycle);
}

void MotorController::run(double speed, int duration_ms) {
    setSpeed(speed);
    ctre::phoenix::unmanaged::Unmanaged::FeedEnable(duration_ms);
}
