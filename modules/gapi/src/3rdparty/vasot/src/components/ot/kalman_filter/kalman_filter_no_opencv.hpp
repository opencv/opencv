/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef VAS_OT_KALMAN_FILTER_NO_OPENCV_HPP
#define VAS_OT_KALMAN_FILTER_NO_OPENCV_HPP

#include <vas/common.hpp>

#include <opencv2/core.hpp>

const float kMeasurementNoiseCoordinate = 0.001f;

const float kMeasurementNoiseRectSize = 0.002f;

namespace vas {

/*
 * This class implements a kernel of a standard kalman filter without using of OpenCV.
 * It supplies simple and common APIs to be use by all components.
 *
 */
class KalmanFilterNoOpencv {
  public:
    /** @brief Create & initialize KalmanFilterNoOpencv
     *      This function initializes Kalman filter with a spectific value of the ratio for measurement noise covariance
     * matrix. If you consider the detection method is enough reliable, it is recommended to use lower ratio value than
     * the default value.
     * @code
     *      cv::Rect2f input_rect(50.f, 50.f, 100.f, 100.f);
     *      cv::Rect2f predicted, corrected;
     *      vas::KalmanFilter kalman_filter = new vas::KalmanFilter(input_rect);
     *      predicted = kalman_filter->Predict();
     *      corrected = kalman_filter->Correct(cv::Rect(52, 52, 105, 105));
     *      delete kalman_filter;
     * @endcode
     * @param
     *      initial_rect                        Initial rectangular coordinates
     */
    explicit KalmanFilterNoOpencv(const cv::Rect2f &initial_rect);
    KalmanFilterNoOpencv() = delete;

    KalmanFilterNoOpencv(const KalmanFilterNoOpencv &) = delete;
    KalmanFilterNoOpencv &operator=(const KalmanFilterNoOpencv &) = delete;

    /* @brief Destroy Kalman filter kernel
     */
    ~KalmanFilterNoOpencv() = default;

    /*
     * This function computes a predicted state.
     * input 'delta_t' is not used.
     */
    cv::Rect2f Predict(float delta_t = 0.033f);

    /*
     * This function updates the predicted state from the measurement.
     */
    cv::Rect2f Correct(const cv::Rect2f &detect_rect);

  private:
    struct kalmanfilter1d32i {
        int32_t X[2];
        int32_t P[2][2];
        int32_t Q[2][2];
        int32_t R;

        int32_t Pk[2][2]; // buffer to copy from Pk-1 to Pk
        int32_t Xk[2];    // buffer to copy form Xk-1 to Xk
    };

    void kalmanfilter1d32i_init(kalmanfilter1d32i *kf, int32_t *z, int32_t var);
    void kalmanfilter1d32i_predict_phase(kalmanfilter1d32i *kf, float dt);
    void kalmanfilter1d32i_update_phase(kalmanfilter1d32i *kf, int32_t z, int32_t *x);
    void kalmanfilter1d32i_filter(kalmanfilter1d32i *kf, int32_t *z, int32_t dt, int32_t *x);

    kalmanfilter1d32i kfX;
    kalmanfilter1d32i kfY;
    kalmanfilter1d32i kfRX;
    kalmanfilter1d32i kfRY;

    float noise_ratio_coordinates_;
    float noise_ratio_rect_size_;
    float delta_t_;
};

}; // namespace vas

#endif // VAS_OT_KALMAN_FILTER_NO_OPENCV_HPP
