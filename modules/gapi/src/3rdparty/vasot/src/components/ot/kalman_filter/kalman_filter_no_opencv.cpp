/*******************************************************************************
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "kalman_filter_no_opencv.hpp"
#include "../../../common/exception.hpp"

#include <cstring>

#define KALMAN_FILTER_NSHIFT 4
#define KF_USE_PARTIAL_64F

namespace vas {

// R
const int32_t kNoiseCovarFactor = 8;
const float kDefaultDeltaT = 0.033f;
const int32_t kDefaultErrCovFactor = 1;

// lower noise = slowly changed
KalmanFilterNoOpencv::KalmanFilterNoOpencv(const cv::Rect2f &initial_rect) : delta_t_(kDefaultDeltaT) {
    int32_t left = static_cast<int32_t>(initial_rect.x);
    int32_t right = static_cast<int32_t>(initial_rect.x + initial_rect.width);
    int32_t top = static_cast<int32_t>(initial_rect.y);
    int32_t bottom = static_cast<int32_t>(initial_rect.y + initial_rect.height);
    int32_t cX = (left + right) << (KALMAN_FILTER_NSHIFT - 1);
    int32_t cY = (top + bottom) << (KALMAN_FILTER_NSHIFT - 1);
    int32_t cRX = (right - left) << (KALMAN_FILTER_NSHIFT - 1);
    int32_t cRY = (bottom - top) << (KALMAN_FILTER_NSHIFT - 1);
    kalmanfilter1d32i_init(&kfX, &cX, 0);
    kalmanfilter1d32i_init(&kfY, &cY, 0);
    kalmanfilter1d32i_init(&kfRX, &cRX, 0);
    kalmanfilter1d32i_init(&kfRY, &cRY, 0);

    int32_t object_size = std::max(64, (cRX * cRY));
    noise_ratio_coordinates_ = kMeasurementNoiseCoordinate;
    noise_ratio_rect_size_ = kMeasurementNoiseRectSize;

    // Set default Q
    int32_t cood_cov = static_cast<int32_t>(object_size * noise_ratio_coordinates_);
    int32_t size_cov = static_cast<int32_t>(object_size * noise_ratio_rect_size_);
    kfX.Q[0][0] = cood_cov;
    kfX.Q[1][1] = cood_cov;
    kfY.Q[0][0] = cood_cov;
    kfY.Q[1][1] = cood_cov;
    kfRX.Q[0][0] = size_cov;
    kfRY.Q[0][0] = size_cov;
}

cv::Rect2f KalmanFilterNoOpencv::Predict(float delta_tf) {
    delta_t_ = delta_tf;

    kalmanfilter1d32i_predict_phase(&kfX, delta_tf);
    kalmanfilter1d32i_predict_phase(&kfY, delta_tf);
    kalmanfilter1d32i_predict_phase(&kfRX, 0);
    kalmanfilter1d32i_predict_phase(&kfRY, 0);

    int32_t cp_x = kfX.Xk[0] >> KALMAN_FILTER_NSHIFT;
    int32_t cp_y = kfY.Xk[0] >> KALMAN_FILTER_NSHIFT;
    int32_t rx = kfRX.Xk[0] >> KALMAN_FILTER_NSHIFT;
    int32_t ry = kfRY.Xk[0] >> KALMAN_FILTER_NSHIFT;

    int32_t pre_x = cp_x - rx;
    int32_t pre_y = cp_y - ry;
    auto width = 2 * rx;
    auto height = 2 * ry;

    // printf(" - In Predict: result (%d, %d  %dx%d)\n", pre_x, pre_y, width, height);
    return cv::Rect2f(float(pre_x), float(pre_y),
                      float(width), float(height));
}

cv::Rect2f KalmanFilterNoOpencv::Correct(const cv::Rect2f &measured_region) {
    int32_t pX = static_cast<int32_t>(measured_region.x + (measured_region.x + measured_region.width))
                 << (KALMAN_FILTER_NSHIFT - 1);
    int32_t pY = static_cast<int32_t>(measured_region.y + (measured_region.y + measured_region.height))
                 << (KALMAN_FILTER_NSHIFT - 1);
    int32_t pRX = static_cast<int32_t>(measured_region.width) << (KALMAN_FILTER_NSHIFT - 1);
    int32_t pRY = static_cast<int32_t>(measured_region.height) << (KALMAN_FILTER_NSHIFT - 1);
    int32_t cX = 0;
    int32_t cY = 0;
    int32_t cRX = 0;
    int32_t cRY = 0;

    int32_t delta_t = static_cast<int32_t>(delta_t_ * 31.3f);
    if (delta_t < kDefaultErrCovFactor)
        delta_t = kDefaultErrCovFactor;

    // Set rect-size-adaptive process/observation noise covariance
    int32_t object_size = std::max(64, (pRX * pRY));
    // Q
    int32_t cood_cov = static_cast<int32_t>(object_size * noise_ratio_coordinates_ * delta_t);
    int32_t size_cov = static_cast<int32_t>(object_size * noise_ratio_rect_size_ * delta_t);

    kfX.Q[0][0] = cood_cov;
    kfX.Q[1][1] = cood_cov;
    kfY.Q[0][0] = cood_cov;
    kfY.Q[1][1] = cood_cov;
    kfRX.Q[0][0] = size_cov;
    kfRY.Q[0][0] = size_cov;

    if (kfX.Xk[0] == 0 && kfY.Xk[0] == 0) {
        kalmanfilter1d32i_predict_phase(&kfX, delta_t_);
        kalmanfilter1d32i_predict_phase(&kfY, delta_t_);
        kalmanfilter1d32i_predict_phase(&kfRX, 0);
        kalmanfilter1d32i_predict_phase(&kfRY, 0);
    }

    // R
    int32_t noise_covariance = object_size >> (kNoiseCovarFactor + delta_t);
    kfX.R = noise_covariance;
    kfY.R = noise_covariance;
    kfRX.R = noise_covariance;
    kfRY.R = noise_covariance;

    kalmanfilter1d32i_update_phase(&kfX, pX, &cX);
    kalmanfilter1d32i_update_phase(&kfY, pY, &cY);
    kalmanfilter1d32i_update_phase(&kfRX, pRX, &cRX);
    kalmanfilter1d32i_update_phase(&kfRY, pRY, &cRY);

    auto x = (cX - cRX) >> KALMAN_FILTER_NSHIFT;
    auto y = (cY - cRY) >> KALMAN_FILTER_NSHIFT;
    auto width = (cRX >> (KALMAN_FILTER_NSHIFT - 1));
    auto height = (cRY >> (KALMAN_FILTER_NSHIFT - 1));

    // printf(" - In Correct: result (%d, %d  %dx%d)\n", x, y, width, height);
    return cv::Rect2f(float(x), float(y),
                      float(width), float(height));
}

void KalmanFilterNoOpencv::kalmanfilter1d32i_init(kalmanfilter1d32i *kf, int32_t *z, int32_t var) {
    std::memset(kf, 0, sizeof(kalmanfilter1d32i));
    if (z) {
        kf->X[0] = *z;
    }

    kf->P[0][0] = var;
    kf->P[1][1] = 0;
}

static void mul_matvec_32f(int32_t Ab[2], float A[2][2], int32_t b[2]) {
    Ab[0] = static_cast<int32_t>(A[0][0] * b[0] + A[0][1] * b[1]); // b[0] + dt * b[1]
    Ab[1] = static_cast<int32_t>(A[1][0] * b[0] + A[1][1] * b[1]); // b[1] ( A[1][0] == 0)
}

static void mul_matmat_32f(int32_t AB[2][2], float trans_mat[2][2], int32_t B[2][2]) {
    AB[0][0] =
        static_cast<int32_t>(trans_mat[0][0] * B[0][0] + trans_mat[0][1] * B[1][0]); // kf->P[0][0] + dt * kf->P[1][0]
    AB[0][1] =
        static_cast<int32_t>(trans_mat[0][0] * B[0][1] + trans_mat[0][1] * B[1][1]); // kf->P[0][1] + dt * kf->P[1][1]
    AB[1][0] = static_cast<int32_t>(trans_mat[1][1] * B[1][0]);
    AB[1][1] = static_cast<int32_t>(trans_mat[1][1] * B[1][1]);
}

#ifndef KF_USE_PARTIAL_64F
static void mul_matmat_32i(int32_t AB[2][2], int32_t A[2][2], int32_t B[2][2]) {
    AB[0][0] = A[0][0] * B[0][0] + A[0][1] * B[1][0];
    AB[0][1] = A[0][0] * B[0][1] + A[0][1] * B[1][1];
    AB[1][0] = A[1][0] * B[0][0] + A[1][1] * B[1][0];
    AB[1][1] = A[1][0] * B[0][1] + A[1][1] * B[1][1];
}

static void mul_matmatT_32i(int32_t ABt[2][2], int32_t A[2][2], int32_t B[2][2]) {
    ABt[0][0] = A[0][0] * B[0][0] + A[0][1] * B[0][1];
    ABt[0][1] = A[0][0] * B[1][0] + A[0][1] * B[1][1];
    ABt[1][0] = A[1][0] * B[0][0] + A[1][1] * B[0][1];
    ABt[1][1] = A[1][0] * B[1][0] + A[1][1] * B[1][1];
}
#endif

static void mul_matmatT_32f(int32_t ABt[2][2], int32_t A[2][2], float B[2][2]) {
    ABt[0][0] = static_cast<int32_t>(A[0][0] * B[0][0] + A[0][1] * B[0][1]);
    ABt[0][1] = static_cast<int32_t>(A[0][0] * B[1][0] + A[0][1] * B[1][1]);
    ABt[1][0] = static_cast<int32_t>(A[1][0] * B[0][0] + A[1][1] * B[0][1]);
    ABt[1][1] = static_cast<int32_t>(A[1][0] * B[1][0] + A[1][1] * B[1][1]);
}

static void add_matmat_32i(int32_t A_B[2][2], int32_t A[2][2], int32_t B[2][2]) {
    A_B[0][0] = A[0][0] + B[0][0];
    A_B[0][1] = A[0][1] + B[0][1];
    A_B[1][0] = A[1][0] + B[1][0];
    A_B[1][1] = A[1][1] + B[1][1];
}

void KalmanFilterNoOpencv::kalmanfilter1d32i_predict_phase(kalmanfilter1d32i *kf, float dt) {
    float F[2][2] = {{1.f, 1.f}, {0.f, 1.f}};
    float A[2][2] = {{1.f, 1.f}, {0.f, 1.f}};
    int32_t AP[2][2];
    int32_t APAt[2][2];

    float weight = 8.f; // 2^(KALMAN_FILTER_NSHIFT - 1)
    float delta_t = dt * weight;

    F[0][1] = delta_t;

    // Predict state
    //  - [x(k) = F x(k-1)]
    mul_matvec_32f(kf->Xk, F, kf->X);

    // Predict error estimate covariance matrix (Predicted estimate covariance) : P(k)
    //  - [P(k) = F P(k-1) Ft + Q]
    mul_matmat_32f(AP, A, kf->P);
    mul_matmatT_32f(APAt, AP, A);
    add_matmat_32i(kf->Pk, APAt, kf->Q);

    // Update kf->x from x(k-1) to x(k)
    kf->X[0] = kf->Xk[0];
    kf->X[1] = kf->Xk[1];

    // Update kf->P from P(k-1) to P(k)
    kf->P[0][0] = kf->Pk[0][0];
    kf->P[0][1] = kf->Pk[0][1];
    kf->P[1][0] = kf->Pk[1][0];
    kf->P[1][1] = kf->Pk[1][1];
}

void KalmanFilterNoOpencv::kalmanfilter1d32i_update_phase(kalmanfilter1d32i *kf, int32_t z, int32_t *x) {
    int32_t y;
    int32_t S;
    int32_t K[2];
    int32_t I_KH[2][2];

    if (kf->Xk[0] == 0 && kf->Pk[0][0] == 0) {
        (*x) = z;
        return;
    }

    // Compute measurement pre-fit residual : Y
    // H    : measurement matrix
    // z(k) : actual reading(observed) result of k
    //  - [ Y(k) = z(k) - H * X(k) ]
    y = z - kf->Xk[0];

    // Compute residual covariance : S
    //  - [ S = H*P(k)*Ht + R]
    S = kf->Pk[0][0] + kf->R;

    if (S == 0) {
        (*x) = z;
        return;
    }

    // Compute optimal kalman gain : K(k)
    //  - [ K(k) = P(k)*Ht*inv(S)]
    // K[0] = kf->P[0][0]/S;
    // K[1] = kf->P[1][0]/S;
    K[0] = kf->Pk[0][0];
    K[1] = kf->Pk[1][0];

    // Get updated state
    //  - [ x'(k) = x(k) + K'*Y )]
    kf->X[0] = kf->Xk[0] + K[0] * y / S;
    kf->X[1] = kf->Xk[1] + K[1] * y / S;

    // 7. Get updated estimate covariance : P'(k)
    //  - [ P'(k) = (I - K(k) * H) * P(k)]
    I_KH[0][0] = S - K[0];
    I_KH[0][1] = 0;
    I_KH[1][0] = -K[1];
    I_KH[1][1] = S;

    // modified by chan - 20110329 - start
    // Here, INTEGER is 32bit.
    // To avoid overflow in the below matrix multiplecation, this code is modified.
#ifdef KF_USE_PARTIAL_64F
    {
        kf->P[0][0] = static_cast<int32_t>(
            (I_KH[0][0] * static_cast<double>(kf->Pk[0][0]) + I_KH[0][1] * static_cast<double>(kf->Pk[1][0])) / S);
        kf->P[0][1] = static_cast<int32_t>(
            (I_KH[0][0] * static_cast<double>(kf->Pk[0][1]) + I_KH[0][1] * static_cast<double>(kf->Pk[1][1])) / S);
        kf->P[1][0] = static_cast<int32_t>(
            (I_KH[1][0] * static_cast<double>(kf->Pk[0][0]) + I_KH[1][1] * static_cast<double>(kf->Pk[1][0])) / S);
        kf->P[1][1] = static_cast<int32_t>(
            (I_KH[1][0] * static_cast<double>(kf->Pk[0][1]) + I_KH[1][1] * static_cast<double>(kf->Pk[1][1])) / S);
    }
#else  // KF_USE_PARTIAL_64F
    {
        mul_matmat_32i(kf->P, I_KH, kf->Pk);
        kf->P[0][0] /= S;
        kf->P[0][1] /= S;
        kf->P[1][0] /= S;
        kf->P[1][1] /= S;
    }
#endif // KF_USE_PARTIAL_64F

    // 9. return result
    (*x) = kf->X[0];
}

}; // namespace vas
