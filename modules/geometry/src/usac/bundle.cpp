// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "../precomp.hpp"
#include "../usac.hpp"

namespace cv { namespace usac {
class MlesacLoss {
  public:
    MlesacLoss(double threshold) : squared_thr(threshold * threshold), norm_thr(squared_thr*3), one_over_thr(1/norm_thr), inv_sq_thr(1/squared_thr) {}
    double loss(double r2) const {
        return r2 < norm_thr ? r2 * one_over_thr - 1 : 0;
    }
    double weight(double r2) const {
        // use Cauchly weight
        return 1.0 / (1.0 + r2 * inv_sq_thr);
    }
    const double squared_thr;
    private:
    const double norm_thr, one_over_thr, inv_sq_thr;
};

class RelativePoseJacobianAccumulator {
private:
    const Mat* correspondences;
    const std::vector<int> &sample;
    const int sample_size;
    const MlesacLoss &loss_fn;
    const double *weights;

public:
    RelativePoseJacobianAccumulator(
        const Mat& correspondences_,
        const std::vector<int> &sample_,
        const int sample_size_,
        const MlesacLoss &l,
        const double *w = nullptr) :
            correspondences(&correspondences_),
            sample(sample_),
            sample_size(sample_size_),
            loss_fn(l),
            weights(w) {}

    Matx33d essential_from_motion(const CameraPose &pose) const {
        return Matx33d(0.0, -pose.t(2), pose.t(1),
            pose.t(2), 0.0, -pose.t(0),
            -pose.t(1), pose.t(0), 0.0) * pose.R;
    }

    double residual(const CameraPose &pose) const {
        const Matx33d E = essential_from_motion(pose);
        const float m11=static_cast<float>(E(0,0)), m12=static_cast<float>(E(0,1)), m13=static_cast<float>(E(0,2));
        const float m21=static_cast<float>(E(1,0)), m22=static_cast<float>(E(1,1)), m23=static_cast<float>(E(1,2));
        const float m31=static_cast<float>(E(2,0)), m32=static_cast<float>(E(2,1)), m33=static_cast<float>(E(2,2));
        const auto * const pts = (float *) correspondences->data;
        double cost = 0.0;
        for (int k = 0; k < sample_size; ++k) {
            const int idx = 4*sample[k];
            const float x1=pts[idx], y1=pts[idx+1], x2=pts[idx+2], y2=pts[idx+3];
            const float F_pt1_x = m11 * x1 + m12 * y1 + m13,
                        F_pt1_y = m21 * x1 + m22 * y1 + m23;
            const float pt2_F_x = x2 * m11 + y2 * m21 + m31,
                        pt2_F_y = x2 * m12 + y2 * m22 + m32;
            const float pt2_F_pt1 = x2 * F_pt1_x + y2 * F_pt1_y + m31 * x1 + m32 * y1 + m33;
            const float r2 = pt2_F_pt1 * pt2_F_pt1 / (F_pt1_x * F_pt1_x + F_pt1_y * F_pt1_y +
                                                      pt2_F_x * pt2_F_x + pt2_F_y * pt2_F_y);
            if (weights == nullptr)
                 cost += loss_fn.loss(r2);
            else cost += weights[k] * loss_fn.loss(r2);
        }
        return cost;
    }

    void accumulate(const CameraPose &pose, Matx<double, 5, 5> &JtJ, Matx<double, 5, 1> &Jtr, Matx<double, 3, 2> &tangent_basis) const {
        const auto * const pts = (float *) correspondences->data;
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        Vec3d tangent_basis_col0;
        if (std::abs(pose.t(0)) < std::abs(pose.t(1))) {
            // x < y
            if (std::abs(pose.t(0)) < std::abs(pose.t(2))) {
                tangent_basis_col0 = pose.t.cross(Vec3d(1,0,0));
            } else {
                tangent_basis_col0 = pose.t.cross(Vec3d(0,0,1));
            }
        } else {
            // x > y
            if (std::abs(pose.t(1)) < std::abs(pose.t(2))) {
                tangent_basis_col0 = pose.t.cross(Vec3d(0,1,0));
            } else {
                tangent_basis_col0 = pose.t.cross(Vec3d(0,0,1));
            }
        }
        tangent_basis_col0 /= norm(tangent_basis_col0);
        Vec3d tangent_basis_col1 = pose.t.cross(tangent_basis_col0);
        tangent_basis_col1 /= norm(tangent_basis_col1);
        for (int i = 0; i < 3; i++) {
            tangent_basis(i,0) = tangent_basis_col0(i);
            tangent_basis(i,1) = tangent_basis_col1(i);
        }

        const Matx33d E = essential_from_motion(pose);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        const Matx<double, 9, 3> dR = {0., -E(0,2), E(0,1),
                                       0., -E(1,2), E(1,1),
                                       0., -E(2,2), E(2,1),
                                       E(0,2), 0., -E(0,0),
                                       E(1,2), 0., -E(1,0),
                                       E(2,2), 0., -E(2,0),
                                       -E(0,1), E(0,0), 0.,
                                       -E(1,1), E(1,0), 0.,
                                       -E(2,1), E(2,0), 0.};

        Matx<double, 9, 2> dt;
        // Each column is vec(skew(tangent_basis[k])*R)
        for (int i = 0; i <= 2; i+=1) {
            const Vec3d r_i(pose.R(0,i), pose.R(1,i), pose.R(2,i));
            for (int j = 0; j <= 1; j+= 1) {
                const Vec3d v = (j == 0 ? tangent_basis_col0 : tangent_basis_col1).cross(r_i);
                for (int k = 0; k < 3; k++) {
                    dt(3*i+k,j) = v[k];
                }
            }
        }

        for (int k = 0; k < sample_size; ++k) {
            const auto point_idx = 4*sample[k];
            const Vec3d pt1 (pts[point_idx], pts[point_idx+1], 1), pt2 (pts[point_idx+2], pts[point_idx+3], 1);
            const double C = pt2.dot(E * pt1);

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            const Vec4d J_C ((E.col(0).t() * pt2)[0], (E.col(1).t() * pt2)[0], (E.row(0) * pt1)[0], (E.row(1) * pt1)[0]);
            const double nJ_C = norm(J_C);
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            if (r*r > loss_fn.squared_thr) continue;

            // Compute weight from robust loss function (used in the IRLS)
            double weight = loss_fn.weight(r * r) / sample_size;
            if (weights != nullptr)
                weight = weights[k] * weight;

            if(weight < DBL_EPSILON)
                continue;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Matx<double, 1, 9> dF (pt1(0) * pt2(0), pt1(0) * pt2(1), pt1(0), pt1(1) * pt2(0), pt1(1) * pt2(1), pt1(1), pt2(0), pt2(1), 1.0);
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * pt1(0) + J_C(0) * pt2(0));
            dF(1) -= s * (J_C(3) * pt1(0) + J_C(0) * pt2(1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * pt1(1) + J_C(1) * pt2(0));
            dF(4) -= s * (J_C(3) * pt1(1) + J_C(1) * pt2(1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            const Matx13d dFdR = dF * dR;
            const Matx12d dFdt = dF * dt;
            const Matx<double, 1, 5> J (dFdR(0), dFdR(1), dFdR(2), dFdt(0), dFdt(1));

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.t();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
        }
    }
};

bool satisfyCheirality (const Matx33d& R, const Vec3d &t, const Vec3d &x1, const Vec3d &x2) {
    // This code assumes that x1 and x2 are unit vectors
     const auto Rx1 = R * x1;
    // lambda_2 * x2 = R * ( lambda_1 * x1 ) + t
    // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
    // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
    const double a = -Rx1.dot(x2), b1 = -Rx1.dot(t), b2 = x2.dot(t);
    // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
    return (b1 - a * b2 > 0) && (-a * b1 + b2 > 0);
}

int refine_relpose(const Mat &correspondences_,
                    const std::vector<int> &sample_,
                    const int sample_size_,
                    CameraPose *pose,
                    const BundleOptions &opt,
                    const double* weights) {
    MlesacLoss loss_fn(opt.loss_scale);
    RelativePoseJacobianAccumulator accum(correspondences_, sample_, sample_size_, loss_fn, weights);
    // return lm_5dof_impl(accum, pose, opt);

    Matx<double, 5, 5> JtJ;
    Matx<double, 5, 1> Jtr;
    Matx<double, 3, 2> tangent_basis;
    Matx33d sw = Matx33d::zeros();
    double lambda = opt.initial_lambda;

    // Compute initial cost
    double cost = accum.residual(*pose);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            std::fill(JtJ.val, JtJ.val+25, 0);
            std::fill(Jtr.val, Jtr.val +5, 0);
            accum.accumulate(*pose, JtJ, Jtr, tangent_basis);
            if (norm(Jtr) < opt.gradient_tol)
                break;
        }

        // Add dampening
        JtJ(0, 0) += lambda;
        JtJ(1, 1) += lambda;
        JtJ(2, 2) += lambda;
        JtJ(3, 3) += lambda;
        JtJ(4, 4) += lambda;

        Matx<double, 5, 1> sol;
        Matx<double, 5, 5> JtJ_symm = JtJ;
        for (int i = 0; i < 5; i++)
            for (int j = i+1; j < 5; j++)
                JtJ_symm(i,j) = JtJ(j,i);

        const bool success = solve(-JtJ_symm, Jtr, sol);
        if (!success || norm(sol) < opt.step_tol)
            break;

        Vec3d w (sol(0,0), sol(1,0), sol(2,0));
        const double theta = norm(w);
        w /= theta;
        const double a = std::sin(theta);
        const double b = std::cos(theta);
        sw(0, 1) = -w(2);
        sw(0, 2) = w(1);
        sw(1, 2) = -w(0);
        sw(1, 0) = w(2);
        sw(2, 0) = -w(1);
        sw(2, 1) = w(0);

        CameraPose pose_new;
        pose_new.R = pose->R + pose->R * (a * sw + (1 - b) * sw * sw);
        // In contrast to the 6dof case, we don't apply R here
        // (since this can already be added into tangent_basis)
        pose_new.t = pose->t + Vec3d(Mat(tangent_basis * Matx21d(sol(3,0), sol(4,0))));
        double cost_new = accum.residual(pose_new);

        if (cost_new < cost) {
            *pose = pose_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            JtJ(0, 0) -= lambda;
            JtJ(1, 1) -= lambda;
            JtJ(2, 2) -= lambda;
            JtJ(3, 3) -= lambda;
            JtJ(4, 4) -= lambda;
            lambda *= 10;
            recompute_jac = false;
        }
    }
    return iter;
}
}}