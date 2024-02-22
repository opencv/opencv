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
#include "DualVar.h"
#include "../Utils_Npnp/Definitions.h"
#include "NPnpProblemSolver.h"
#include <Eigen/Eigenvalues>
#include <memory>

namespace NPnP
{

    Eigen::Vector4d DualVar::extract_quaternions()
    {
        const double eps = 1E-3;
        auto q11 = y_vec[4];
        auto q12 = y_vec[5];
        auto q13 = y_vec[6];
        auto q14 = y_vec[7];
        auto q22 = y_vec[8];
        auto q23 = y_vec[9];
        auto q24 = y_vec[10];
        auto q33 = y_vec[11];
        auto q34 = y_vec[12];
        auto q44 = y_vec[13];
        double q1;
        double q2;
        double q3;
        double q4;

        if (q11 > eps)
        {
            q1 = std::sqrt(q11);
            q2 = q12 / q1;
            q3 = q13 / q1;
            q4 = q14 / q1;
        }
        else if (q22 > eps)
        {
            q1 = 0.0;
            q2 = std::sqrt(q22);
            q3 = q23 / q2;
            q4 = q24 / q2;
        }
        else if (q33 > eps)
        {
            q1 = 0.0;
            q2 = 0.0;
            q3 = std::sqrt(q33);
            q4 = q34 / q3;
        }
        else if (q44 > eps)
        {
            q1 = 0.0;
            q2 = 0.0;
            q3 = 0.0;
            q4 = 1.0;
        }
        Eigen::Vector4d quats(q1, q2, q3, q4);
        quats.normalize();
        return std::move(quats);
    }

    DualVar::DualVar(ColVector<Y_SIZE> a_y_vec, ColVector<NUM_CONSTRAINTS> a_zero_vars,
                     ColVector<A_ROWS> a_slack,
                     ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> a_matrix_15_15,
                     ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM> a_matrix_15_15_inv,
                     ColVector<M_MATRIX_DIM * M_MATRIX_DIM> a_matrix_15_15_inv_vec,
                     Eigen::Array<double, M_MATRIX_DIM, 1> a_eigenvals,
                     const RowMatrix<NUM_CONSTRAINTS, Y_SIZE> &A_sub_rows)
        : y_vec(std::move(a_y_vec)), zero_vars(std::move(a_zero_vars)),
          slack(std::move(a_slack)), matrix_15_15(std::move(a_matrix_15_15)),
          matrix_15_15_inv(std::move(a_matrix_15_15_inv)),
          matrix_15_15_inv_vec(std::move(a_matrix_15_15_inv_vec)),
          eigenvals(std::move(a_eigenvals))
    {
        this->gradient_helper_vec = Eigen::VectorXd(A_ROWS);
        this->gradient_vec = Eigen::VectorXd(Y_SIZE);
        this->hessian_helper_mat = Eigen::MatrixXd::Zero(A_ROWS, A_ROWS);
        this->hessian_mat = Eigen::MatrixXd::Zero(Y_SIZE, Y_SIZE);
        this->hessian_multiplication_mat =
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(
                Y_SIZE, A_ROWS);
        this->equation_result = Eigen::VectorXd(Y_SIZE + NUM_CONSTRAINTS);
        this->res_vec_84 = Eigen::VectorXd::Zero(Y_SIZE + NUM_CONSTRAINTS);
        this->temp_vec_69 = Eigen::VectorXd::Zero(Y_SIZE);

        this->equation_mat =
            Eigen::MatrixXd::Zero(Y_SIZE + NUM_CONSTRAINTS, Y_SIZE + NUM_CONSTRAINTS);
        this->equation_mat.block(0, Y_SIZE, Y_SIZE, NUM_CONSTRAINTS) =
            A_sub_rows.transpose();
        this->equation_mat.block(Y_SIZE, 0, NUM_CONSTRAINTS, Y_SIZE) = A_sub_rows;
    }

    Eigen::VectorXd *DualVar::cone_barrier_gradient(const PnpProblemSolver &pnp)
    {
        gradient_helper_vec.block<NUM_CONSTRAINTS, 1>(0, 0).setZero();
        gradient_helper_vec.block<M_MATRIX_DIM * M_MATRIX_DIM, 1>(
            NUM_CONSTRAINTS, 0) = this->matrix_15_15_inv_vec;
        gradient_vec.noalias() = pnp.A_cols.transpose() * gradient_helper_vec;
        return &gradient_vec;
    }

    Eigen::MatrixXd *DualVar::cone_barrier_hessian(const PnpProblemSolver &pnp)
    {

        hessian_helper_mat
            .block(NUM_CONSTRAINTS, NUM_CONSTRAINTS, A_ROWS - NUM_CONSTRAINTS,
                   A_ROWS - NUM_CONSTRAINTS)
            .noalias() = matrix_15_15_inv_vec * matrix_15_15_inv_vec.transpose();

        auto cop = hessian_helper_mat.eval();

        for (int i = 0; i < M_MATRIX_DIM; i++)
            for (int j = 0; j < M_MATRIX_DIM; j++)
                if (i != j)
                    hessian_helper_mat
                        .block(M_MATRIX_DIM + i * M_MATRIX_DIM,
                               M_MATRIX_DIM + j * M_MATRIX_DIM, M_MATRIX_DIM, M_MATRIX_DIM)
                        .transposeInPlace();

        hessian_multiplication_mat.noalias() =
            pnp.A_cols.transpose() * hessian_helper_mat;
        hessian_mat.noalias() = hessian_multiplication_mat * pnp.A_cols;

        return &hessian_mat;
    }

    std::shared_ptr<DualVar>
    DualVar::init(ColVector<Y_SIZE> y_vec, const SparseRowMatrix &A_mat_rows,
                  const RowMatrix<NUM_CONSTRAINTS, Y_SIZE> &A_sub_rows,
                  const SparseColMatrix &c_vec)
    {
        ColVector<A_ROWS> slack = (c_vec - A_mat_rows * y_vec).toDense();
        Eigen::Map<ColVector<NUM_CONSTRAINTS>> zero_part(slack.data(),
                                                         NUM_CONSTRAINTS);
        Eigen::Map<ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM>> matrix_15_15(
            slack.data() + NUM_CONSTRAINTS, M_MATRIX_DIM, M_MATRIX_DIM);
        for (int i = 0; i < M_MATRIX_DIM; i++)
            matrix_15_15(i, i) += 0.00000001;
        auto matrix_15_15_inv = matrix_15_15.inverse().eval();
        Eigen::Map<ColVector<M_MATRIX_DIM * M_MATRIX_DIM>> matrix_15_15_inv_vec(
            matrix_15_15_inv.data(), M_MATRIX_DIM * M_MATRIX_DIM);
        auto eigenvals = matrix_15_15.selfadjointView<Eigen::Upper>().eigenvalues();

        return std::make_shared<DualVar>(std::move(y_vec), zero_part, slack,
                                         matrix_15_15, matrix_15_15_inv,
                                         matrix_15_15_inv_vec, eigenvals, A_sub_rows);
    }

    void DualVar::set_y_vec(ColVector<Y_SIZE> new_y_vec,
                            const SparseRowMatrix &A_mat_rows,
                            const SparseColMatrix &c_vec)
    {
        this->y_vec = std::move(new_y_vec);
        this->slack = (c_vec - A_mat_rows * y_vec).toDense();

        Eigen::Map<ColVector<NUM_CONSTRAINTS>> new_zero_vars(slack.data(),
                                                             NUM_CONSTRAINTS);
        Eigen::Map<ColMatrix<M_MATRIX_DIM, M_MATRIX_DIM>> new_matrix_15_15(
            slack.data() + NUM_CONSTRAINTS, M_MATRIX_DIM, M_MATRIX_DIM);
        for (int i = 0; i < M_MATRIX_DIM; i++)
        {
            new_matrix_15_15(i, i) += 0.00000001;
        }
        this->matrix_15_15 = new_matrix_15_15;
        this->zero_vars = new_zero_vars;
        this->matrix_15_15_inv = new_matrix_15_15.inverse();

        Eigen::Map<ColVector<M_MATRIX_DIM * M_MATRIX_DIM>> new_matrix_15_15_inv_vec(
            matrix_15_15_inv.data(), M_MATRIX_DIM * M_MATRIX_DIM);
        this->matrix_15_15_inv_vec = new_matrix_15_15_inv_vec;
        this->eigenvals = matrix_15_15.selfadjointView<Eigen::Upper>().eigenvalues();
    }
} // namespace NPnP
