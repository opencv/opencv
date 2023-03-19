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
#ifndef PNP_USING_EIGEN_LIBRARY_PNPPROBLEMSOLVER_H
#define PNP_USING_EIGEN_LIBRARY_PNPPROBLEMSOLVER_H

#include "BarrierMethodSettings.h"
#include "../Utils_Npnp/Definitions.h"
#include "DualVar.h"
#include "NPnpInput.h"
#include "NPnpObjective.h"
#include "NPnpResult.h"

namespace NPnP
{

    class PnpProblemSolver
    {
    public:
        SparseRowMatrix A_rows;
        SparseColMatrix A_cols;
        SparseColMatrix c_vec;
        RowMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_rows;
        ColMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_cols;
        ColMatrix<NUM_CONSTRAINTS, NUM_CONSTRAINTS> zero_mat_15_15;
        ColVector<Y_SIZE> init_y;
        std::shared_ptr<DualVar> dual_var;

        PnpProblemSolver(const SparseRowMatrix &A_rows, const SparseColMatrix &A_cols,
                         const SparseColMatrix &c_vec,
                         RowMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_rows,
                         ColMatrix<NUM_CONSTRAINTS, Y_SIZE> zero_sub_A_cols,
                         ColMatrix<NUM_CONSTRAINTS, NUM_CONSTRAINTS> zero_mat_15_15,
                         ColVector<Y_SIZE> init_y, std::shared_ptr<DualVar> dual_var);

        static std::shared_ptr<PnpProblemSolver> init();

        PnpResult
        solve_pnp(std::shared_ptr<PnpObjective> pnp_objective,
                  std::shared_ptr<BarrierMethodSettings> barrier_method_settings);

        void
        line_search(const ColVector<Y_SIZE> &delta_y,
                    const std::function<double(const ColVector<Y_SIZE> &)> &obj_func,
                    std::shared_ptr<BarrierMethodSettings> barrier_settings);

        void centering_step(
            const std::function<double(const ColVector<Y_SIZE> &)> &obj_func,
            const std::function<const Eigen::VectorXd &(DualVar &)> &gen_grad,
            const std::function<const Eigen::MatrixXd &(DualVar &)> &gen_hess,
            int outer_iteration, bool &is_last_iter, PnpResult &pnp_result,
            std::shared_ptr<PnpObjective> pnp_objective,
            std::shared_ptr<BarrierMethodSettings> barrier_method_settings);

        Eigen::Vector4d
        perform_local_search(std::shared_ptr<PnpObjective> pnp_objective,
                             Eigen::Vector4d quats);
    };
} // namespace NPnP

#endif // PNP_USING_EIGEN_LIBRARY_PNPPROBLEMSOLVER_H
