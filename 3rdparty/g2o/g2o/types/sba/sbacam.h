// g2o - General Graph Optimization
// Copyright (C) 2011 Kurt Konolige
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

///
/// Basic types definitions for SBA translation to HChol
///
/// Camera nodes use camera pose in real world
///   v3 position
///   normalized quaternion rotation
///
/// Point nodes:
///   v3 position
///

#ifndef G2O_SBACam_
#define G2O_SBACam_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "g2o/stuff/misc.h"
#include "g2o/stuff/macros.h"
#include "g2o/types/slam3d/se3quat.h"

#include "g2o_types_sba_api.h"

// this seems to have to go outside of the AISNav namespace...
//USING_PART_OF_NAMESPACE_EIGEN;

namespace g2o {

  class G2O_TYPES_SBA_API SBACam: public SE3Quat
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
      // camera matrix and stereo baseline
      Matrix3 Kcam; 
      number_t baseline;

      // transformations
      Eigen::Matrix<number_t,3,4> w2n; // transform from world to node coordinates
      Eigen::Matrix<number_t,3,4> w2i; // transform from world to image coordinates

      // Derivatives of the rotation matrix transpose wrt quaternion xyz, used for
      // calculating Jacobian wrt pose of a projection.
      Matrix3 dRdx, dRdy, dRdz;

      // initialize an object
      SBACam()
      {
        setKcam(1,1, cst(0.5), cst(0.5),0);  // unit image projection
      }


      // set the object pose
      SBACam(const Quaternion&  r_, const Vector3& t_) : SE3Quat(r_, t_)
    {
      setTransform();
      setProjection();
      setDr();
    }

      SBACam(const SE3Quat& p) : SE3Quat(p) {
        setTransform();
        setProjection();
        setDr();
      }

      // update from the linear solution
      //defined in se3quat
      void update(const Vector6& update)
      {
        //      std::cout << "UPDATE " << update.transpose() << std::endl;
        // position update
        _t += update.head(3);
        // small quaternion update
        Quaternion qr;
        qr.vec() = update.segment<3>(3); 
        qr.w() = sqrt(cst(1.0) - qr.vec().squaredNorm()); // should always be positive
        _r *= qr;                 // post-multiply
        _r.normalize();    
        setTransform();
        setProjection();
        setDr();
        //      std::cout << t.transpose() << std::endl;
        //      std::cout << r.coeffs().transpose() << std::endl;
      }

      // transforms
      static void transformW2F(Eigen::Matrix<number_t,3,4> &m, 
          const Vector3 &trans, 
          const Quaternion &qrot)
      {
        m.block<3,3>(0,0) = qrot.toRotationMatrix().transpose();
        m.col(3).setZero();         // make sure there's no translation
        Vector4 tt;
        tt.head(3) = trans;
        tt[3] = 1.0;
        m.col(3) = -m*tt;
      }

      static void transformF2W(Eigen::Matrix<number_t,3,4> &m, 
          const Vector3 &trans, 
          const Quaternion &qrot)
      {
        m.block<3,3>(0,0) = qrot.toRotationMatrix();
        m.col(3) = trans;
      }

      // set up camera matrix
      void setKcam(number_t fx, number_t fy, number_t cx, number_t cy, number_t tx)
      { 
        Kcam.setZero();
        Kcam(0,0) = fx;
        Kcam(1,1) = fy;
        Kcam(0,2) = cx;
        Kcam(1,2) = cy;
        Kcam(2,2) = cst(1.0);
        baseline = tx;
        setProjection();
        setDr();
      }

      // set transform from world to cam coords
      void setTransform()  { transformW2F(w2n,_t,_r); }

      // Set up world-to-image projection matrix (w2i), assumes camera parameters
      // are filled.
      void setProjection() { w2i = Kcam * w2n; }

      // sets angle derivatives
      void setDr()
      {
        // inefficient, just for testing
        // use simple multiplications and additions for production code in calculating dRdx,y,z
        Matrix3 dRidx, dRidy, dRidz;
        dRidx << cst(0.0), cst(0.0), cst(0.0),
            cst(0.0), cst(0.0), cst(2.0),
            cst(0.0), cst(-2.0), cst(0.0);
        dRidy  << cst(0.0), cst(0.0), cst(-2.0),
            cst(0.0), cst(0.0), cst(0.0),
            cst(2.0), cst(0.0), cst(0.0);
        dRidz  << cst(0.0), cst(2.0), cst(0.0),
            cst(-2.0), cst(0.0), cst(0.0),
            cst(0.0), cst(0.0), cst(0.0);

        // for dS'*R', with dS the incremental change
        dRdx = dRidx * w2n.block<3,3>(0,0);
        dRdy = dRidy * w2n.block<3,3>(0,0);
        dRdz = dRidz * w2n.block<3,3>(0,0);
      }

  };


  // human-readable SBACam object
  inline std::ostream& operator <<(std::ostream& out_str,
                                   const SBACam& cam)
  {
    out_str << cam.translation().transpose() << std::endl;
    out_str << cam.rotation().coeffs().transpose() << std::endl << std::endl;
    out_str << cam.Kcam << std::endl << std::endl;
    out_str << cam.w2n << std::endl << std::endl;
    out_str << cam.w2i << std::endl << std::endl;

    return out_str;
  };


  //
  // class for edges from vps to points with normals
  //

  class G2O_TYPES_SBA_API EdgeNormal
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
      // point position
      Vector3 pos;

      // unit normal
      Vector3 normal;

      // rotation matrix for normal
      Matrix3 R; 

      // initialize an object
      EdgeNormal()
      {
        pos.setZero();
        normal << 0, 0, 1;
        makeRot();
      }

      // set up rotation matrix
      void makeRot()
      {
        Vector3 y;
        y << 0, 1, 0;
        R.row(2) = normal;
        y = y - normal(1)*normal;
        y.normalize();            // need to check if y is close to 0
        R.row(1) = y;
        R.row(0) = normal.cross(R.row(1));
        //      cout << normal.transpose() << endl;
        //      cout << R << endl << endl;
        //      cout << R*R.transpose() << endl << endl;
      }

  };

} // end namespace


#endif  // SBACam
