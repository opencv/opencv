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
import cv2.cv as cv
from math import cos, sin, sqrt
import sys

if __name__ == "__main__":
    A = [ [1, 1], [0, 1] ]

    img = cv.CreateImage((500, 500), 8, 3)
    kalman = cv.CreateKalman(2, 1, 0)
    state = cv.CreateMat(2, 1, cv.CV_32FC1)  # (phi, delta_phi)
    process_noise = cv.CreateMat(2, 1, cv.CV_32FC1)
    measurement = cv.CreateMat(1, 1, cv.CV_32FC1)
    rng = cv.RNG(-1)
    code = -1L

    cv.Zero(measurement)
    cv.NamedWindow("Kalman", 1)

    while True:
        cv.RandArr(rng, state, cv.CV_RAND_NORMAL, cv.RealScalar(0), cv.RealScalar(0.1))

        kalman.transition_matrix[0,0] = 1
        kalman.transition_matrix[0,1] = 1
        kalman.transition_matrix[1,0] = 0
        kalman.transition_matrix[1,1] = 1

        cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
        cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-5))
        cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-1))
        cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(1))
        cv.RandArr(rng, kalman.state_post, cv.CV_RAND_NORMAL, cv.RealScalar(0), cv.RealScalar(0.1))


        while True:
            def calc_point(angle):
                return (cv.Round(img.width/2 + img.width/3*cos(angle)),
                         cv.Round(img.height/2 - img.width/3*sin(angle)))

            state_angle = state[0,0]
            state_pt = calc_point(state_angle)

            prediction = cv.KalmanPredict(kalman)
            predict_angle = prediction[0, 0]
            predict_pt = calc_point(predict_angle)

            cv.RandArr(rng, measurement, cv.CV_RAND_NORMAL, cv.RealScalar(0),
                       cv.RealScalar(sqrt(kalman.measurement_noise_cov[0, 0])))

            # generate measurement
            cv.MatMulAdd(kalman.measurement_matrix, state, measurement, measurement)

            measurement_angle = measurement[0, 0]
            measurement_pt = calc_point(measurement_angle)

            # plot points
            def draw_cross(center, color, d):
                cv.Line(img, (center[0] - d, center[1] - d),
                             (center[0] + d, center[1] + d), color, 1, cv.CV_AA, 0)
                cv.Line(img, (center[0] + d, center[1] - d),
                             (center[0] - d, center[1] + d), color, 1, cv.CV_AA, 0)

            cv.Zero(img)
            draw_cross(state_pt, cv.CV_RGB(255, 255, 255), 3)
            draw_cross(measurement_pt, cv.CV_RGB(255, 0,0), 3)
            draw_cross(predict_pt, cv.CV_RGB(0, 255, 0), 3)
            cv.Line(img, state_pt, measurement_pt, cv.CV_RGB(255, 0,0), 3, cv. CV_AA, 0)
            cv.Line(img, state_pt, predict_pt, cv.CV_RGB(255, 255, 0), 3, cv. CV_AA, 0)

            cv.KalmanCorrect(kalman, measurement)

            cv.RandArr(rng, process_noise, cv.CV_RAND_NORMAL, cv.RealScalar(0),
                       cv.RealScalar(sqrt(kalman.process_noise_cov[0, 0])))
            cv.MatMulAdd(kalman.transition_matrix, state, process_noise, state)

            cv.ShowImage("Kalman", img)

            code = cv.WaitKey(100) % 0x100
            if code != -1:
                break

        if code in [27, ord('q'), ord('Q')]:
            break

    cv.DestroyWindow("Kalman")
