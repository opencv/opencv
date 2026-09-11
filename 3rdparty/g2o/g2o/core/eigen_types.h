// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_EIGEN_TYPES_H
#define G2O_EIGEN_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "g2o/config.h"

namespace g2o {

  typedef Eigen::Matrix<int,2,1,Eigen::ColMajor>                                  Vector2I;
  typedef Eigen::Matrix<int,3,1,Eigen::ColMajor>                                  Vector3I;
  typedef Eigen::Matrix<int,4,1,Eigen::ColMajor>                                  Vector4I;
  typedef Eigen::Matrix<int,Eigen::Dynamic,1,Eigen::ColMajor>                     VectorXI; 

  typedef Eigen::Matrix<float,2,1,Eigen::ColMajor>                                Vector2F; 
  typedef Eigen::Matrix<float,3,1,Eigen::ColMajor>                                Vector3F; 
  typedef Eigen::Matrix<float,4,1,Eigen::ColMajor>                                Vector4F; 
  typedef Eigen::Matrix<float,Eigen::Dynamic,1,Eigen::ColMajor>                   VectorXF;

  template<int N>
  using VectorN = Eigen::Matrix<number_t, N, 1, Eigen::ColMajor>;
  using Vector2 = VectorN<2>;
  using Vector3 = VectorN<3>;
  using Vector4 = VectorN<4>;
  using Vector6 = VectorN<6>;
  using Vector7 = VectorN<7>;
  using VectorX = VectorN<Eigen::Dynamic>;

  typedef Eigen::Matrix<int,2,2,Eigen::ColMajor>                                  Matrix2I;
  typedef Eigen::Matrix<int,3,3,Eigen::ColMajor>                                  Matrix3I;
  typedef Eigen::Matrix<int,4,4,Eigen::ColMajor>                                  Matrix4I;
  typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>        MatrixXI;

  typedef Eigen::Matrix<float,2,2,Eigen::ColMajor>                                Matrix2F;
  typedef Eigen::Matrix<float,3,3,Eigen::ColMajor>                                Matrix3F;
  typedef Eigen::Matrix<float,4,4,Eigen::ColMajor>                                Matrix4F;
  typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>      MatrixXF;

  template<int N>
  using MatrixN = Eigen::Matrix<number_t, N, N, Eigen::ColMajor>;
  using Matrix2 = MatrixN<2>;
  using Matrix3 = MatrixN<3>;
  using Matrix4 = MatrixN<4>;
  using MatrixX = MatrixN<Eigen::Dynamic>;

  typedef Eigen::Transform<number_t,2,Eigen::Isometry,Eigen::ColMajor>            Isometry2;
  typedef Eigen::Transform<number_t,3,Eigen::Isometry,Eigen::ColMajor>            Isometry3;

  typedef Eigen::Transform<number_t,2,Eigen::Affine,Eigen::ColMajor>              Affine2;
  typedef Eigen::Transform<number_t,3,Eigen::Affine,Eigen::ColMajor>              Affine3;

  typedef Eigen::Rotation2D<number_t>                                             Rotation2D;

  typedef Eigen::Quaternion<number_t>                                             Quaternion;

  typedef Eigen::AngleAxis<number_t>                                              AngleAxis;

} // end namespace g2o

#endif
