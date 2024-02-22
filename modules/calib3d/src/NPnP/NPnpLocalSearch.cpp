/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "../Utils_Npnp/Definitions.h"
#include "NPnpProblemSolver.h"
#include "QuaternionVector.h"
#include <functional>
#include <iostream>
#include <memory>

namespace NPnP
{

    Eigen::Vector4d PnpProblemSolver::perform_local_search(
        std::shared_ptr<PnpObjective> pnp_objective, Eigen::Vector4d quats)
    {
        const auto eps = 0.0000001;
        auto t = 10.0;
        auto mu = 1.5;

        std::function<double(const QuaternionVector &)> func =
            [&t, pnp_objective](const QuaternionVector &q) -> double
        {
            double barrier_func = (1 - q.q_norm_squared) * (1 - q.q_norm_squared);
            return q.obj_func(pnp_objective->b) + t * barrier_func;
        };
        std::function<Eigen::Vector4d(const QuaternionVector &)> grad =
            [&t, pnp_objective](const QuaternionVector &q) -> Eigen::Vector4d
        {
            Eigen::Vector4d barrier_grad = -2 * (1 - q.q_norm_squared) * q.quaternions;
            return q.obj_grad(pnp_objective->b) + t * barrier_grad;
        };
        std::function<Eigen::Matrix4d(const QuaternionVector &)> hess =
            [&t, pnp_objective](const QuaternionVector &q) -> Eigen::Matrix4d
        {
            Eigen::Matrix4d barrier_hess = q.quaternions * q.quaternions.transpose();
            barrier_hess *= 4;
            barrier_hess -= 2 * (1 - q.q_norm_squared) * Eigen::Matrix4d::Identity();
            barrier_hess *= t;
            barrier_hess += q.obj_hess(pnp_objective->b);
            return barrier_hess;
        };
        QuaternionVector q(quats);
        while (t < 1.0 / eps)
        {
            Eigen::Matrix4d h = hess(q);
            Eigen::Vector4d g = grad(q);
            Eigen::Matrix4d h_inv = h.inverse();
            auto newton_step = -h_inv * g;

            Eigen::Vector4d new_q = q.quaternions + newton_step;
            new_q.normalize();
            q = QuaternionVector(new_q);
            t *= mu;
        }
        return q.quaternions;
    }
} // namespace NPnP
