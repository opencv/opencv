#!/usr/bin/python
"""
   Tracking of rotating point.
   Rotation speed is constant.
   Both state and measurements vectors are 1D (a point angle),
   Measurement is the real point angle + gaussian noise.
   The real and the estimated points are connected with yellow line segment,
   the real and the measured points are connected with red line segment.
   (if Kalman filter works correctly,
    the yellow segment should be shorter than the red one).
   Pressing any key (except ESC) will reset the tracking with a different speed.
   Pressing ESC will stop the program.
"""
import urllib2
import cv2
from math import cos, sin, sqrt
import sys
import numpy as np

if __name__ == "__main__":

    img_height = 500
    img_width = 500
    img = np.array((img_height, img_width, 3), np.uint8)
    kalman = cv2.KalmanFilter(2, 1, 0)
    state = np.zeros((2, 1))  # (phi, delta_phi)
    process_noise = np.zeros((2, 1))
    measurement = np.zeros((1, 1))

    code = -1L

    cv2.namedWindow("Kalman")

    while True:
        state = 0.1 * np.random.randn(2, 1)

        transition_matrix = np.array([[1., 1.], [0., 1.]])
        kalman.setTransitionMatrix(transition_matrix)
        measurement_matrix = 1. * np.ones((1, 2))
        kalman.setMeasurementMatrix(measurement_matrix)

        process_noise_cov = 1e-5
        kalman.setProcessNoiseCov(process_noise_cov * np.eye(2))

        measurement_noise_cov = 1e-1
        kalman.setMeasurementNoiseCov(measurement_noise_cov * np.ones((1, 1)))

        kalman.setErrorCovPost(1. * np.ones((2, 2)))

        kalman.setStatePost(0.1 * np.random.randn(2, 1))

        while True:
            def calc_point(angle):
                return (np.around(img_width/2 + img_width/3*cos(angle), 0).astype(int),
                         np.around(img_height/2 - img_width/3*sin(angle), 1).astype(int))

            state_angle = state[0, 0]
            state_pt = calc_point(state_angle)

            prediction = kalman.predict()
            predict_angle = prediction[0, 0]
            predict_pt = calc_point(predict_angle)


            measurement = measurement_noise_cov * np.random.randn(1, 1) 

            # generate measurement
            measurement = np.dot(measurement_matrix, state) + measurement

            measurement_angle = measurement[0, 0]
            measurement_pt = calc_point(measurement_angle)

            # plot points
            def draw_cross(center, color, d):
                cv2.line(img, (center[0] - d, center[1] - d),
                              (center[0] + d, center[1] + d), color, 1, cv2.LINE_AA, 0)
                cv2.line(img, (center[0] + d, center[1] - d),
                              (center[0] - d, center[1] + d), color, 1, cv2.LINE_AA, 0)

            img = np.zeros((img_height, img_width, 3), np.uint8)
            draw_cross(np.int32(state_pt), (255, 255, 255), 3)
            draw_cross(np.int32(measurement_pt), (0, 0, 255), 3)
            draw_cross(np.int32(predict_pt), (0, 255, 0), 3)

            cv2.line(img, state_pt, measurement_pt, (0, 0, 255), 3, cv2.LINE_AA, 0)
            cv2.line(img, state_pt, predict_pt, (0, 255, 255), 3, cv2.LINE_AA, 0)

            kalman.correct(measurement)

            process_noise = process_noise_cov * np.random.randn(2, 1)
            
            state = np.dot(transition_matrix, state) + process_noise

            cv2.imshow("Kalman", img)

            code = cv2.waitKey(100) % 0x100
            if code != -1:
                break

        if code in [27, ord('q'), ord('Q')]:
            break

    cv2.destroyWindow("Kalman")
