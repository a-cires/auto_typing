#include <pybind11/pybind11.h>
#include "motor_control/motor_controller.hpp"

namespace py = pybind11;

PYBIND11_MODULE(motor_cpp, m) {
    py::class_<MotorController>(m, "MotorController")
        .def(py::init<int>(), py::arg("can_id"))
        .def("init", &MotorController::init)
        .def("set_speed", &MotorController::setSpeed, py::arg("duty_cycle"))
        .def("run", &MotorController::run, py::arg("speed"), py::arg("duration_ms"));
}
