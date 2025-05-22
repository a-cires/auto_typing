Imports:
import cv2
import pygame
import keyboard
import numpy as np

Setup:
gui = TypingGUI()

Running:
gui.plot_from_raw_motion(frame, tx, ty, tz, yaw, pitch, roll)
frame - the frame to display onto
tx - translation in the x direction, postive is forwards, negative is backwards
ty - translation in the y direction, positive is upwards, negative is downwards
tz - translation in the z direction, positive is left, negative is right
yaw - rotation around the y axis, positive is turning left, negative is turning right
pitch - rotation around the z axis, positive is pitch up. negative is pitch downwards
roll - rotation around the x axis, positive turns the camera CCW, negative turns the camera CW
