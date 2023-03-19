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
#ifndef PNP_USING_EIGEN_LIBRARY_DUALVAR_H
#define PNP_USING_EIGEN_LIBRARY_DUALVAR_H

#include <memory>

#include <Eigen/Core>

#include "BarrierMethodSettings.h"
#include "../Utils_Npnp/Definitions.h"
#include "NPnpObjective.h"

namespace NPnP
{
    class PnpProblemSolver;

    class DualVar
    {
    public:
        ColVector<Y_SIZE> y_vec;
        ColVector<NUM_CONSTRAINTS> zero_vars;
        ColVector<A_ROWS> slack;
        ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> matrix_15_15;
        ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> matrix_15_15_inv;
        ColVector<M_MATRIX_DIM * M_MATRIX_DIM> matrix_15_15_inv_vec;
        Eigen::Array<double, M_MATRIX_DIM, 1> eigenvals;

        Eigen::VectorXd gradient_helper_vec;
        Eigen::VectorXd gradient_vec;
        Eigen::MatrixXd hessian_helper_mat;
        Eigen::MatrixXd hessian_mat;
        Eigen::MatrixXd equation_mat;
        Eigen::VectorXd equation_result;

        Eigen::VectorXd res_vec_84;
        Eigen::VectorXd temp_vec_69;

        ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> temp_col_1;
        ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> temp_col_2;
        RowMatrix<M_MATRIX_DIM, M_MATRIX_DIM> temp_row_1;
        RowMatrix<M_MATRIX_DIM, M_MATRIX_DIM> temp_row_2;
        ColVector<M_MATRIX_DIM> temp_vec;
        ColVector<A_ROWS> B_vec;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            hessian_multiplication_mat;

        DualVar(ColVector<Y_SIZE> y_vec, ColVector<NUM_CONSTRAINTS> zero_vars,
                ColVector<A_ROWS> slack,
                ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> matrix_15_15,
                ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> matrix_15_15_inv,
                ColVector<M_MATRIX_DIM * M_MATRIX_DIM> matrix_15_15_inv_vec,
                Eigen::Array<double, M_MATRIX_DIM, 1> eigenvals,
                const RowMatrix<NUM_CONSTRAINTS, Y_SIZE> &A_sub_rows);

        static std::shared_ptr<DualVar>
        init(ColVector<Y_SIZE> y_vec, const SparseRowMatrix &A_mat_rows,
             const RowMatrix<NUM_CONSTRAINTS, Y_SIZE> &A_sub_rows,
             const SparseColMatrix &c_vec);

        void set_y_vec(ColVector<Y_SIZE> y_vec, const SparseRowMatrix &A_mat_rows,
                       const SparseColMatrix &c_vec);

        Eigen::VectorXd *cone_barrier_gradient(const PnpProblemSolver &pnp);

        Eigen::MatrixXd *cone_barrier_hessian(const PnpProblemSolver &pnp);

        Eigen::Vector4d extract_quaternions();
    };
} // namespace PnP

#endif // PNP_USING_EIGEN_LIBRARY_DUALVAR_H
