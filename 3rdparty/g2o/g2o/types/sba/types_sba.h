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

#ifndef G2O_SBA_TYPES
#define G2O_SBA_TYPES

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "sbacam.h"
#include <Eigen/Geometry>
#include <iostream>

#include "g2o_types_sba_api.h"

namespace g2o {

/**
 * \brief Vertex encoding the intrinsics of the camera fx, fy, cx, xy, baseline;
 */

class G2O_TYPES_SBA_API VertexIntrinsics : public BaseVertex<4, Eigen::Matrix<number_t, 5, 1, Eigen::ColMajor> >
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexIntrinsics();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
      
    virtual void setToOriginImpl() {
      _estimate << cst(1.), cst(1.), cst(0.5), cst(0.5), cst(0.1);
    }
      
    virtual void oplusImpl(const number_t* update)
    {
      _estimate.head<4>() += Vector4(update);
    }
 };

/**
 * \brief SBACam Vertex, (x,y,z,qw,qx,qy,qz)
 * the parameterization for the increments constructed is a 6d vector
 * (x,y,z,qx,qy,qz) (note that we leave out the w part of the quaternion.
 * qw is assumed to be positive, otherwise there is an ambiguity in qx,qy,qz as a rotation
 */


  class G2O_TYPES_SBA_API VertexCam : public BaseVertex<6, SBACam>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCam();

    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
      _estimate = SBACam();
    }
    
    virtual void setEstimate(const SBACam& cam){
      BaseVertex<6, SBACam>::setEstimate(cam);
      _estimate.setTransform();
      _estimate.setProjection();
      _estimate.setDr();
    }
    
    virtual void oplusImpl(const number_t* update)
    {
      Eigen::Map<const Vector6> v(update);
      _estimate.update(v);
      _estimate.setTransform();
      _estimate.setProjection();
      _estimate.setDr();
    }
    

    virtual bool setEstimateDataImpl(const number_t* est){
      Eigen::Map <const Vector7> v(est);
      _estimate.fromVector(v);
      return true;
    }

    virtual bool getEstimateData(number_t* est) const{
      Eigen::Map <Vector7> v(est);
      v = estimate().toVector();
      return true;
    }

    virtual int estimateDimension() const {
      return 7;
    }

    virtual bool setMinimalEstimateDataImpl(const number_t* est){
      Eigen::Map<const Vector6> v(est);
      _estimate.fromMinimalVector(v);
      return true;
    }

    virtual bool getMinimalEstimateData(number_t* est) const{
      Eigen::Map<Vector6> v(est);
      v = _estimate.toMinimalVector();
      return true;
    }

    virtual int minimalEstimateDimension() const {
      return 6;
    }
 };

/**
 * \brief Point vertex, XYZ
 */
 class G2O_TYPES_SBA_API VertexSBAPointXYZ : public BaseVertex<3, Vector3>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW    
    VertexSBAPointXYZ();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
      _estimate.fill(0);
    }

    virtual void oplusImpl(const number_t* update)
    {
      Eigen::Map<const Vector3> v(update);
      _estimate += v;
    }
};


// monocular projection
// first two args are the measurement type, second two the connection classes
 class G2O_TYPES_SBA_API EdgeProjectP2MC : public  BaseBinaryEdge<2, Vector2, VertexSBAPointXYZ, VertexCam> 
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectP2MC();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    // return the error estimate as a 2-vector
    void computeError()
    {
      // from <Point> to <Cam>
      const VertexSBAPointXYZ *point = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      const VertexCam *cam = static_cast<const VertexCam*>(_vertices[1]);

      // calculate the projection
      const Vector3 &pt = point->estimate();
      Vector4 ppt(pt(0),pt(1),pt(2),1);
      Vector3 p = cam->estimate().w2i * ppt;
      Vector2 perr;
      perr = p.head<2>()/p(2);
      //      std::cout << std::endl << "CAM   " << cam->estimate() << std::endl;
      //      std::cout << "POINT " << pt.transpose() << std::endl;
      //      std::cout << "PROJ  " << p.transpose() << std::endl;
      //      std::cout << "CPROJ " << perr.transpose() << std::endl;
      //      std::cout << "MEAS  " << _measurement.transpose() << std::endl;

      // error, which is backwards from the normal observed - calculated
      // _measurement is the measured projection
      _error = perr - _measurement;
      // std::cerr << _error.x() << " " << _error.y() <<  " " << chi2() << std::endl;
    }

    // jacobian
    virtual void linearizeOplus();
};

// stereo projection
// first two args are the measurement type, second two the connection classes
 class G2O_TYPES_SBA_API EdgeProjectP2SC : public  BaseBinaryEdge<3, Vector3, VertexSBAPointXYZ, VertexCam>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectP2SC();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    // return the error estimate as a 2-vector
    void computeError()
    {
      // from <Point> to <Cam>
      const VertexSBAPointXYZ *point = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      VertexCam *cam = static_cast<VertexCam*>(_vertices[1]);

      // calculate the projection
      Vector3 kp;
      Vector4 pt;
      pt.head<3>() = point->estimate();
      pt(3) = 1;
      const SBACam& nd = cam->estimate();
      // these should be already ok
      /* nd.setTransform(); */
      /* nd.setProjection(); */
      /* nd.setDr(); */

      Vector3 p1 = nd.w2i * pt; 
      Vector3 p2 = nd.w2n * pt; 
      Vector3 pb(nd.baseline,0,0);

      number_t invp1 = cst(1.0)/p1(2);
      kp.head<2>() = p1.head<2>()*invp1;

      // right camera px
      p2 = nd.Kcam*(p2-pb);
      kp(2) = p2(0)/p2(2);

      // std::cout << std::endl << "CAM   " << cam->estimate() << std::endl; 
      // std::cout << "POINT " << pt.transpose() << std::endl; 
      // std::cout << "PROJ  " << p1.transpose() << std::endl; 
      // std::cout << "PROJ  " << p2.transpose() << std::endl; 
      // std::cout << "CPROJ " << kp.transpose() << std::endl; 
      // std::cout << "MEAS  " << _measurement.transpose() << std::endl; 

      // error, which is backwards from the normal observed - calculated
      // _measurement is the measured projection
      _error = kp - _measurement;
    }

    // jacobian
    virtual void linearizeOplus();

};

// monocular projection with parameter calibration
// first two args are the measurement type, second two the connection classes
 class G2O_TYPES_SBA_API EdgeProjectP2MC_Intrinsics : public  BaseMultiEdge<2, Vector2> 
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeProjectP2MC_Intrinsics();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    // return the error estimate as a 2-vector
    void computeError()
    {
      // from <Point> to <Cam>, the intrinsics in KCam should be already set!
      const VertexSBAPointXYZ *point = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      VertexCam *cam = static_cast<VertexCam*>(_vertices[1]);
      // calculate the projection
      const Vector3 &pt = point->estimate();
      Vector4 ppt(pt(0),pt(1),pt(2),cst(1.0));
      Vector3 p = cam->estimate().w2i * ppt;
      Vector2 perr = p.head<2>()/p(2);
      _error = perr - _measurement;
    }

    // jacobian
    virtual void linearizeOplus();

};


/**
 * \brief 3D edge between two SBAcam
 */
 class G2O_TYPES_SBA_API EdgeSBACam : public BaseBinaryEdge<6, SE3Quat, VertexCam, VertexCam>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSBACam();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    void computeError()
    {
      const VertexCam* v1 = dynamic_cast<const VertexCam*>(_vertices[0]);
      const VertexCam* v2 = dynamic_cast<const VertexCam*>(_vertices[1]);
      SE3Quat delta = _inverseMeasurement * (v1->estimate().inverse()*v2->estimate());
      _error[0]=delta.translation().x();
      _error[1]=delta.translation().y();
      _error[2]=delta.translation().z();
      _error[3]=delta.rotation().x();
      _error[4]=delta.rotation().y();
      _error[5]=delta.rotation().z();
    }
    
    virtual void setMeasurement(const SE3Quat& meas){
      _measurement=meas;
      _inverseMeasurement=meas.inverse();
    }

    virtual number_t initialEstimatePossible(const OptimizableGraph::VertexSet& , OptimizableGraph::Vertex* ) { return cst(1.);}
    virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to);

    virtual bool setMeasurementData(const number_t* d){
      Eigen::Map<const Vector7> v(d);
      _measurement.fromVector(v);
      _inverseMeasurement = _measurement.inverse();
      return true;
    }

    virtual bool getMeasurementData(number_t* d) const{
      Eigen::Map<Vector7> v(d);
      v = _measurement.toVector();
      return true;
    }

    virtual int measurementDimension() const {return 7;}

    virtual bool setMeasurementFromState();
    
  protected:
    SE3Quat _inverseMeasurement;
};


/**
 * \brief edge between two SBAcam that specifies the distance between them
 */
 class G2O_TYPES_SBA_API EdgeSBAScale : public BaseBinaryEdge<1, number_t, VertexCam, VertexCam>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSBAScale();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;
    void computeError()
    {
      const VertexCam* v1 = dynamic_cast<const VertexCam*>(_vertices[0]);
      const VertexCam* v2 = dynamic_cast<const VertexCam*>(_vertices[1]);
      Vector3 dt=v2->estimate().translation()-v1->estimate().translation();
      _error[0] = _measurement - dt.norm();
    }
    virtual void setMeasurement(const number_t& m){
      _measurement = m;
    }
    virtual number_t initialEstimatePossible(const OptimizableGraph::VertexSet& , OptimizableGraph::Vertex* ) { return cst(1.);}
    virtual void initialEstimate(const OptimizableGraph::VertexSet& from_, OptimizableGraph::Vertex* to_);
};



} // end namespace

#endif // SBA_TYPES
