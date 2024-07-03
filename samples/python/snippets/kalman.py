#!/usr/bin/env python
"""
   Tracking of rotating point.
   Point moves in a circle and is characterized by a 1D state.
   state_k+1 = state_k + speed + process_noise N(0, 1e-5)
   The speed is constant.
   Both state and measurements vectors are 1D (a point angle),
   Measurement is the real state + gaussian noise N(0, 1e-1).
   The real and the measured points are connected with red line segment,
   the real and the estimated points are connected with yellow line segment,
   the real and the corrected estimated points are connected with green line segment.
   (if Kalman filter works correctly,
    the yellow segment should be shorter than the red one and
    the green segment should be shorter than the yellow one).
   Pressing any key (except ESC) will reset the tracking.
   Pressing ESC will stop the program.
"""

import numpy as np
import cv2 as cv

from math import cos, sin, sqrt, pi

def main():
    img_height = 500
    img_width = 500
    kalman = cv.KalmanFilter(2, 1, 0)

    code = -1
    num_circle_steps = 12
    while True:
        img = np.zeros((img_height, img_width, 3), np.uint8)
        state = np.array([[0.0],[(2 * pi) / num_circle_steps]])   # start state
        kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]])  # F. input
        kalman.measurementMatrix = 1. * np.eye(1, 2)              # H. input
        kalman.processNoiseCov = 1e-5 * np.eye(2)                 # Q. input
        kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1))       # R. input
        kalman.errorCovPost = 1. * np.eye(2, 2)                   # P._k|k  KF state var
        kalman.statePost = 0.1 * np.random.randn(2, 1)            # x^_k|k  KF state var

        while True:
            def calc_point(angle):
                return (np.around(img_width / 2. + img_width / 3.0 * cos(angle), 0).astype(int),
                        np.around(img_height / 2. - img_width / 3.0 * sin(angle), 1).astype(int))
            img = img * 1e-3
            state_angle = state[0, 0]
            state_pt = calc_point(state_angle)
            # advance Kalman filter to next timestep
            # updates statePre, statePost, errorCovPre, errorCovPost
            # k-> k+1, x'(k) = A*x(k)
            # P'(k) = temp1*At + Q
            prediction = kalman.predict()

            predict_pt = calc_point(prediction[0, 0])  # equivalent to calc_point(kalman.statePre[0,0])
            # generate measurement
            measurement = kalman.measurementNoiseCov * np.random.randn(1, 1)
            measurement = np.dot(kalman.measurementMatrix, state) + measurement

            measurement_angle = measurement[0, 0]
            measurement_pt = calc_point(measurement_angle)

            # correct the state estimates based on measurements
            # updates statePost & errorCovPost
            kalman.correct(measurement)
            improved_pt = calc_point(kalman.statePost[0, 0])

            # plot points
            cv.drawMarker(img, measurement_pt, (0, 0, 255), cv.MARKER_SQUARE, 5, 2)
            cv.drawMarker(img, predict_pt, (0, 255, 255), cv.MARKER_SQUARE, 5, 2)
            cv.drawMarker(img, improved_pt, (0, 255, 0), cv.MARKER_SQUARE, 5, 2)
            cv.drawMarker(img, state_pt, (255, 255, 255), cv.MARKER_STAR, 10, 1)
            # forecast one step
            cv.drawMarker(img, calc_point(np.dot(kalman.transitionMatrix, kalman.statePost)[0, 0]),
                          (255, 255, 0), cv.MARKER_SQUARE, 12, 1)

            cv.line(img, state_pt, measurement_pt, (0, 0, 255), 1, cv.LINE_AA, 0)  # red measurement error
            cv.line(img, state_pt, predict_pt, (0, 255, 255), 1, cv.LINE_AA, 0)  # yellow pre-meas error
            cv.line(img, state_pt, improved_pt, (0, 255, 0), 1, cv.LINE_AA, 0)  # green post-meas error

            # update the real process
            process_noise = sqrt(kalman.processNoiseCov[0, 0]) * np.random.randn(2, 1)
            state = np.dot(kalman.transitionMatrix, state) + process_noise  # x_k+1 = F x_k + w_k

            cv.imshow("Kalman", img)
            code = cv.waitKey(1000)
            if code != -1:
                break

        if code in [27, ord('q'), ord('Q')]:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
