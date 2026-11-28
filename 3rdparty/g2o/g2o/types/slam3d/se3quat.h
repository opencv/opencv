// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
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

#ifndef G2O_SE3QUAT_H_
#define G2O_SE3QUAT_H_

#include "se3_ops.h"
#include "g2o/stuff/misc.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {

  class G2O_TYPES_SLAM3D_API SE3Quat {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    protected:

      Quaternion _r;
      Vector3 _t;

    public:
      SE3Quat(){
        _r.setIdentity();
        _t.setZero();
      }

      SE3Quat(const Matrix3& R, const Vector3& t):_r(Quaternion(R)),_t(t){ 
        normalizeRotation();
      }

      SE3Quat(const Quaternion& q, const Vector3& t):_r(q),_t(t){
        normalizeRotation();
      }

      /**
       * templaized constructor which allows v to be an arbitrary Eigen Vector type, e.g., Vector6 or Map<Vector6>
       */
      template <typename Derived>
        explicit SE3Quat(const Eigen::MatrixBase<Derived>& v)
        {
          assert((v.size() == 6 || v.size() == 7) && "Vector dimension does not match");
          if (v.size() == 6) {
            for (int i=0; i<3; i++){
              _t[i]=v[i];
              _r.coeffs()(i)=v[i+3];
            }
            _r.w() = 0.; // recover the positive w
            if (_r.norm()>1.){
              _r.normalize();
            } else {
              number_t w2= cst(1.)-_r.squaredNorm();
              _r.w()= (w2<cst(0.)) ? cst(0.) : std::sqrt(w2);
            }
          }
          else if (v.size() == 7) {
            int idx = 0;
            for (int i=0; i<3; ++i, ++idx)
              _t(i) = v(idx);
            for (int i=0; i<4; ++i, ++idx)
              _r.coeffs()(i) = v(idx);
            normalizeRotation();
          }
        }

      inline const Vector3& translation() const {return _t;}

      inline void setTranslation(const Vector3& t_) {_t = t_;}

      inline const Quaternion& rotation() const {return _r;}

      void setRotation(const Quaternion& r_) {_r=r_;}

      inline SE3Quat operator* (const SE3Quat& tr2) const{
        SE3Quat result(*this);
        result._t += _r*tr2._t;
        result._r*=tr2._r;
        result.normalizeRotation();
        return result;
      }

      inline SE3Quat& operator*= (const SE3Quat& tr2){
        _t+=_r*tr2._t;
        _r*=tr2._r;
        normalizeRotation();
        return *this;
      }

      inline Vector3 operator* (const Vector3& v) const {
        return _t+_r*v;
      }

      inline SE3Quat inverse() const{
        SE3Quat ret;
        ret._r=_r.conjugate();
        ret._t=ret._r*(_t*-cst(1.));
        return ret;
      }

      inline number_t operator [](int i) const {
        assert(i<7);
        if (i<3)
          return _t[i];
        return _r.coeffs()[i-3];
      }


      inline Vector7 toVector() const{
        Vector7 v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=_r.x();
        v[4]=_r.y();
        v[5]=_r.z();
        v[6]=_r.w();
        return v;
      }

      inline void fromVector(const Vector7& v){
        _r=Quaternion(v[6], v[3], v[4], v[5]);
        _t=Vector3(v[0], v[1], v[2]);
      }

      inline Vector6 toMinimalVector() const{
        Vector6 v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=_r.x();
        v[4]=_r.y();
        v[5]=_r.z();
        return v;
      }

      inline void fromMinimalVector(const Vector6& v){
        number_t w = cst(1.)-v[3]*v[3]-v[4]*v[4]-v[5]*v[5];
        if (w>0){
          _r=Quaternion(std::sqrt(w), v[3], v[4], v[5]);
        } else {
          _r=Quaternion(0, -v[3], -v[4], -v[5]);
        }
        _t=Vector3(v[0], v[1], v[2]);
      }



      Vector6 log() const {
        Vector6 res;
        Matrix3 _R = _r.toRotationMatrix();
        number_t d = cst(0.5)*(_R(0,0)+_R(1,1)+_R(2,2)-1);
        Vector3 omega;
        Vector3 upsilon;


        Vector3 dR = deltaR(_R);
        Matrix3 V_inv;

        if (std::abs(d)>cst(0.99999))
        {

          omega=0.5*dR;
          Matrix3 Omega = skew(omega);
          V_inv = Matrix3::Identity()- cst(0.5)*Omega + (cst(1.)/ cst(12.))*(Omega*Omega);
        }
        else
        {
          number_t theta = std::acos(d);
          omega = theta/(2*std::sqrt(1-d*d))*dR;
          Matrix3 Omega = skew(omega);
          V_inv = ( Matrix3::Identity() - cst(0.5)*Omega
              + ( 1-theta/(2*std::tan(theta/2)))/(theta*theta)*(Omega*Omega) );
        }

        upsilon = V_inv*_t;
        for (int i=0; i<3;i++){
          res[i]=omega[i];
        }
        for (int i=0; i<3;i++){
          res[i+3]=upsilon[i];
        }

        return res;

      }

      Vector3 map(const Vector3 & xyz) const
      {
        return _r*xyz + _t;
      }


      static SE3Quat exp(const Vector6 & update)
      {
        Vector3 omega;
        for (int i=0; i<3; i++)
          omega[i]=update[i];
        Vector3 upsilon;
        for (int i=0; i<3; i++)
          upsilon[i]=update[i+3];

        number_t theta = omega.norm();
        Matrix3 Omega = skew(omega);

        Matrix3 R;
        Matrix3 V;
        if (theta<cst(0.00001))
        {
          Matrix3 Omega2 = Omega*Omega;

          R = (Matrix3::Identity()
              + Omega
              + cst(0.5) * Omega2);

          V = (Matrix3::Identity()
              + cst(0.5) * Omega
              + cst(1.) / cst(6.) * Omega2);
        }
        else
        {
          Matrix3 Omega2 = Omega*Omega;

          R = (Matrix3::Identity()
              + std::sin(theta)/theta *Omega
              + (1-std::cos(theta))/(theta*theta)*Omega2);

          V = (Matrix3::Identity()
              + (1-std::cos(theta))/(theta*theta)*Omega
              + (theta-std::sin(theta))/(std::pow(theta,3))*Omega2);
        }
        return SE3Quat(Quaternion(R),V*upsilon);
      }

      Eigen::Matrix<number_t, 6, 6, Eigen::ColMajor> adj() const
      {
        Matrix3 R = _r.toRotationMatrix();
        Eigen::Matrix<number_t, 6, 6, Eigen::ColMajor> res;
        res.block(0,0,3,3) = R;
        res.block(3,3,3,3) = R;
        res.block(3,0,3,3) = skew(_t)*R;
        res.block(0,3,3,3) = Matrix3::Zero(3,3);
        return res;
      }

      Eigen::Matrix<number_t,4,4,Eigen::ColMajor> to_homogeneous_matrix() const
      {
        Eigen::Matrix<number_t,4,4,Eigen::ColMajor> homogeneous_matrix;
        homogeneous_matrix.setIdentity();
        homogeneous_matrix.block(0,0,3,3) = _r.toRotationMatrix();
        homogeneous_matrix.col(3).head(3) = translation();

        return homogeneous_matrix;
      }

      void normalizeRotation(){
        if (_r.w()<0){
          _r.coeffs() *= -1;
        }
        _r.normalize();
      }

      /**
       * cast SE3Quat into an Isometry3
       */
      operator Isometry3() const
      {
        Isometry3 result = (Isometry3) rotation();
        result.translation() = translation();
        return result;
      }
  };

  inline std::ostream& operator <<(std::ostream& out_str, const SE3Quat& se3)
  {
    out_str << se3.to_homogeneous_matrix()  << std::endl;
    return out_str;
  }

  //G2O_TYPES_SLAM3D_API Quaternion euler_to_quat(number_t yaw, number_t pitch, number_t roll);
  //G2O_TYPES_SLAM3D_API void quat_to_euler(const Quaternion& q, number_t& yaw, number_t& pitch, number_t& roll);
  //G2O_TYPES_SLAM3D_API void jac_quat3_euler3(Eigen::Matrix<number_t, 6, 6, Eigen::ColMajor>& J, const SE3Quat& t);

} // end namespace

#endif
