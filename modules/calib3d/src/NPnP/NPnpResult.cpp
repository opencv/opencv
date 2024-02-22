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
#include "NPnpResult.h"
#include "../Utils_Npnp/Definitions.h"

namespace NPnP
{
    double PnpResult::cost()
    {
        auto rotation_matrix = this->rotation_matrix();
        Eigen::Map<ColVector<9>> rotation_vector(rotation_matrix.data(), 9);
        auto sum_error =
            (rotation_vector.transpose() * pnp_objective->M * rotation_vector)(0, 0);
        return sum_error / pnp_objective->sum_weights;
    }

    Eigen::Matrix3d PnpResult::rotation_matrix()
    {
        auto q1 = this->quaternions.x();
        auto q2 = this->quaternions.y();
        auto q3 = this->quaternions.z();
        auto q4 = this->quaternions.w();
        auto q11 = q1 * q1;
        auto q12 = q1 * q2;
        auto q13 = q1 * q3;
        auto q14 = q1 * q4;
        auto q22 = q2 * q2;
        auto q23 = q2 * q3;
        auto q24 = q2 * q4;
        auto q33 = q3 * q3;
        auto q34 = q3 * q4;
        auto q44 = q4 * q4;
        Eigen::Matrix3d rotation_matrix;
        rotation_matrix << q11 + q22 - q33 - q44, 2 * q23 - 2 * q14,
            2 * q24 + 2 * q13, 2 * q23 + 2 * q14, q11 + q33 - q22 - q44,
            2 * q34 - 2 * q12, 2 * q24 - 2 * q13, 2 * q34 + 2 * q12,
            q11 + q44 - q22 - q33;
        return rotation_matrix;
    }

    Eigen::Vector3d PnpResult::translation_vector()
    {
        auto rotation_matrix = this->rotation_matrix();
        Eigen::Map<ColVector<9>> rotation_vector(rotation_matrix.data(), 9);
        return pnp_objective->T * rotation_vector;
    }
} // namespace NPnP
