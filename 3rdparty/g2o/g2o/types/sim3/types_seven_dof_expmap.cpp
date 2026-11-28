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

#include "types_seven_dof_expmap.h"

#include "g2o/core/factory.h"
#include "g2o/stuff/macros.h"

namespace g2o {

  G2O_USE_TYPE_GROUP(sba);
  
  G2O_REGISTER_TYPE_GROUP(sim3);

  G2O_REGISTER_TYPE(VERTEX_SIM3:EXPMAP, VertexSim3Expmap);
  G2O_REGISTER_TYPE(EDGE_SIM3:EXPMAP, EdgeSim3);
  G2O_REGISTER_TYPE(EDGE_PROJECT_SIM3_XYZ:EXPMAP, EdgeSim3ProjectXYZ);
  G2O_REGISTER_TYPE(EDGE_PROJECT_INVERSE_SIM3_XYZ:EXPMAP, EdgeInverseSim3ProjectXYZ);
  
    VertexSim3Expmap::VertexSim3Expmap() : BaseVertex<7, Sim3>()
    {
        _marginalized=false;
        _fix_scale = false;
        
        _principle_point1[0] = 0;
        _principle_point1[1] = 0;
        _focal_length1[0] = 1;
        _focal_length1[1] = 1;
        
        _principle_point2[0] = 0;
        _principle_point2[1] = 0;
        _focal_length2[0] = 1;
        _focal_length2[1] = 1;
    }
    
    
    EdgeSim3::EdgeSim3() :
    BaseBinaryEdge<7, Sim3, VertexSim3Expmap, VertexSim3Expmap>()
    {
    }
    
    
    bool VertexSim3Expmap::read(std::istream &is) {
        Vector7 cam2world;
        for (int i = 0; i < 6; i++) {
            is >> cam2world[i];
        }
        is >> cam2world[6];
        //    if (! is) {
        //      // if the scale is not specified we set it to 1;
        //      std::cerr << "!s";
        //      cam2world[6]=0.;
        //    }
        
        for (int i = 0; i < 2; i++) {
            is >> _focal_length1[i];
        }
        for (int i = 0; i < 2; i++) {
            is >> _principle_point1[i];
        }
        
        setEstimate(Sim3(cam2world).inverse());
        return true;
    }
    
    bool VertexSim3Expmap::write(std::ostream &os) const {
        Sim3 cam2world(estimate().inverse());
        Vector7 lv = cam2world.log();
        for (int i = 0; i < 7; i++) {
            os << lv[i] << " ";
        }
        for (int i = 0; i < 2; i++) {
            os << _focal_length1[i] << " ";
        }
        for (int i = 0; i < 2; i++) {
            os << _principle_point1[i] << " ";
        }
        return os.good();
    }
    
bool EdgeSim3::read(std::istream& is)
  {
    Vector7 v7;
    for (int i=0; i<7; i++){
      is >> v7[i];
    }

    Sim3 cam2world(v7);
    setMeasurement(cam2world.inverse());

    for (int i=0; i<7; i++)
      for (int j=i; j<7; j++)
      {
        is >> information()(i,j);
        if (i!=j)
          information()(j,i)=information()(i,j);
      }
    return true;
  }

  bool EdgeSim3::write(std::ostream& os) const
  {
    Sim3 cam2world(measurement().inverse());
    Vector7 v7 = cam2world.log();
    for (int i=0; i<7; i++)
    {
      os  << v7[i] << " ";
    }
    for (int i=0; i<7; i++)
      for (int j=i; j<7; j++){
        os << " " <<  information()(i,j);
    }
    return os.good();
  }

#if G2O_SIM3_JACOBIAN
void EdgeSim3::linearizeOplus() {
    VertexSim3Expmap *v1 = static_cast<VertexSim3Expmap *>(_vertices[0]);
    VertexSim3Expmap *v2 = static_cast<VertexSim3Expmap *>(_vertices[1]);
    const Sim3 Si(v1->estimate());//Siw
    const Sim3 Sj(v2->estimate());

    const Sim3& Sji = _measurement;

    // error in Lie Algebra
    const Eigen::Matrix<double, 7, 1> error = (Sji * Si * Sj.inverse()).log();
    const Eigen::Vector3d phi = error.block<3, 1>(0, 0); // rotation
    const Eigen::Vector3d tau = error.block<3, 1>(3, 0); // translation
    const double s = error(6);                           // scale

    const Eigen::Matrix<double, 7, 7> I7 = Eigen::Matrix<double, 7, 7>::Identity();
    const Eigen::Matrix<double, 3, 3> I3 = Eigen::Matrix<double, 3, 3>::Identity();

    // Jacobi Matrix of Si 
    // note: because the order of rotation and translation is different, 
    //       so it is slightly different from the formula.
    Eigen::Matrix<double, 7, 7> jacobi_i = Eigen::Matrix<double, 7, 7>::Zero();
    jacobi_i.block<3, 3>(0, 0) = -skew(phi);
    jacobi_i.block<3, 3>(3, 3) = -(skew(phi) + s * I3);
    jacobi_i.block<3, 3>(3, 0) = -skew(tau);
    jacobi_i.block<3, 1>(3, 6) = tau;

    // Adjoint matrix of Sji
    Eigen::Matrix<double, 7, 7> adj_Sji = I7;
    adj_Sji.block<3, 3>(0, 0) = Sji.rotation().toRotationMatrix();
    adj_Sji.block<3, 3>(3, 3) = Sji.scale() * Sji.rotation().toRotationMatrix();
    adj_Sji.block<3, 3>(3, 0) =
        skew(Sji.translation()) * Sji.rotation().toRotationMatrix();
    adj_Sji.block<3, 1>(3, 6) = -Sji.translation();

    _jacobianOplusXi = (I7 + 0.5 * jacobi_i) * adj_Sji;

    // Jacobi Matrix of Sj 
    Eigen::Matrix<double, 7, 7> jacobi_j = Eigen::Matrix<double, 7, 7>::Zero();
    jacobi_j.block<3, 3>(0, 0) = skew(phi);
    jacobi_j.block<3, 3>(3, 3) = skew(phi) + s * I3;
    jacobi_j.block<3, 3>(3, 0) = skew(tau);
    jacobi_j.block<3, 1>(3, 6) = -tau;

    _jacobianOplusXj = -(I7 + 0.5 * jacobi_j);
}
#endif
    
  /**Sim3ProjectXYZ*/

  EdgeSim3ProjectXYZ::EdgeSim3ProjectXYZ() :
  BaseBinaryEdge<2, Vector2, VertexSBAPointXYZ, VertexSim3Expmap>()
  {
  }

  bool EdgeSim3ProjectXYZ::read(std::istream& is)
  {
    for (int i=0; i<2; i++)
    {
      is >> _measurement[i];
    }

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++) {
  is >> information()(i,j);
      if (i!=j)
        information()(j,i)=information()(i,j);
    }
    return true;
  }

  bool EdgeSim3ProjectXYZ::write(std::ostream& os) const
  {
    for (int i=0; i<2; i++){
      os  << _measurement[i] << " ";
    }

    for (int i=0; i<2; i++)
      for (int j=i; j<2; j++){
  os << " " <<  information()(i,j);
    }
    return os.good();
  }

EdgeInverseSim3ProjectXYZ::EdgeInverseSim3ProjectXYZ() :
    BaseBinaryEdge<2, Vector2, VertexSBAPointXYZ, VertexSim3Expmap>() {
}

bool EdgeInverseSim3ProjectXYZ::read(std::istream &is) {
  for (int i = 0; i < 2; i++) {
    is >> _measurement[i];
  }

  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      is >> information()(i, j);
      if (i != j)
        information()(j, i) = information()(i, j);
    }
  return true;
}

bool EdgeInverseSim3ProjectXYZ::write(std::ostream &os) const {
  for (int i = 0; i < 2; i++) {
    os << _measurement[i] << " ";
  }

  for (int i = 0; i < 2; i++)
    for (int j = i; j < 2; j++) {
      os << " " << information()(i, j);
    }
  return os.good();
}

//  void EdgeSim3ProjectXYZ::linearizeOplus()
//  {
//    VertexSim3Expmap * vj = static_cast<VertexSim3Expmap *>(_vertices[1]);
//    Sim3 T = vj->estimate();

//    VertexPointXYZ* vi = static_cast<VertexPointXYZ*>(_vertices[0]);
//    Vector3 xyz = vi->estimate();
//    Vector3 xyz_trans = T.map(xyz);

//    number_t x = xyz_trans[0];
//    number_t y = xyz_trans[1];
//    number_t z = xyz_trans[2];
//    number_t z_2 = z*z;

//    Matrix<number_t,2,3,Eigen::ColMajor> tmp;
//    tmp(0,0) = _focal_length(0);
//    tmp(0,1) = 0;
//    tmp(0,2) = -x/z*_focal_length(0);

//    tmp(1,0) = 0;
//    tmp(1,1) = _focal_length(1);
//    tmp(1,2) = -y/z*_focal_length(1);

//    _jacobianOplusXi =  -1./z * tmp * T.rotation().toRotationMatrix();

//    _jacobianOplusXj(0,0) =  x*y/z_2 * _focal_length(0);
//    _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *_focal_length(0);
//    _jacobianOplusXj(0,2) = y/z *_focal_length(0);
//    _jacobianOplusXj(0,3) = -1./z *_focal_length(0);
//    _jacobianOplusXj(0,4) = 0;
//    _jacobianOplusXj(0,5) = x/z_2 *_focal_length(0);
//    _jacobianOplusXj(0,6) = 0; // scale is ignored


//    _jacobianOplusXj(1,0) = (1+y*y/z_2) *_focal_length(1);
//    _jacobianOplusXj(1,1) = -x*y/z_2 *_focal_length(1);
//    _jacobianOplusXj(1,2) = -x/z *_focal_length(1);
//    _jacobianOplusXj(1,3) = 0;
//    _jacobianOplusXj(1,4) = -1./z *_focal_length(1);
//    _jacobianOplusXj(1,5) = y/z_2 *_focal_length(1);
//    _jacobianOplusXj(1,6) = 0; // scale is ignored
//  }

} // end namespace
