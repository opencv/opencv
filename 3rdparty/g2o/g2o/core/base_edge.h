// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, H. Strasdat, W. Burgard
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

#ifndef G2O_BASE_EDGE_H
#define G2O_BASE_EDGE_H

#include <iostream>
#include <limits>

#include <Eigen/Core>

#include "optimizable_graph.h"

namespace g2o {

  namespace internal {

  #ifdef G2O_OPENMP
    struct QuadraticFormLock {
      explicit QuadraticFormLock(OptimizableGraph::Vertex& vertex)
        :_vertex(vertex) {
          _vertex.lockQuadraticForm();
        }
      ~QuadraticFormLock() {
        _vertex.unlockQuadraticForm();
      }
      private:
      OptimizableGraph::Vertex& _vertex;
    };
  #else
    struct QuadraticFormLock {
      explicit QuadraticFormLock(OptimizableGraph::Vertex& ) { }
      ~QuadraticFormLock() { }
    };
  #endif

  } // internal namespace

  template <int D, typename E>
  class BaseEdge : public OptimizableGraph::Edge
  {
    public:

      static const int Dimension = D;
      typedef E Measurement;
      typedef Eigen::Matrix<number_t, D, 1, Eigen::ColMajor> ErrorVector;
      typedef Eigen::Matrix<number_t, D, D, Eigen::ColMajor> InformationType;

      BaseEdge() : OptimizableGraph::Edge()
      {
        _dimension = D;
      }

      virtual ~BaseEdge() {}

      virtual number_t chi2() const
      {
        return _error.dot(information()*_error);
      }

      virtual const number_t* errorData() const { return _error.data();}
      virtual number_t* errorData() { return _error.data();}
      const ErrorVector& error() const { return _error;}
      ErrorVector& error() { return _error;}

      //! information matrix of the constraint
      EIGEN_STRONG_INLINE const InformationType& information() const { return _information;}
      EIGEN_STRONG_INLINE InformationType& information() { return _information;}
      EIGEN_STRONG_INLINE void setInformation(const InformationType& information) { _information = information;}

      virtual const number_t* informationData() const { return _information.data();}
      virtual number_t* informationData() { return _information.data();}

      //! accessor functions for the measurement represented by the edge
      EIGEN_STRONG_INLINE const Measurement& measurement() const { return _measurement;}
      virtual void setMeasurement(const Measurement& m) { _measurement = m;}

      virtual int rank() const {return _dimension;}

      virtual void initialEstimate(const OptimizableGraph::VertexSet&, OptimizableGraph::Vertex*)
      {
        std::cerr << "inititialEstimate() is not implemented, please give implementation in your derived class" << std::endl;
      }

    protected:

      Measurement _measurement;
      InformationType _information;
      ErrorVector _error;

      /**
       * calculate the robust information matrix by updating the information matrix of the error
       */
      InformationType robustInformation(const Vector3& rho)
      {
        InformationType result = rho[1] * _information;
        //ErrorVector weightedErrror = _information * _error;
        //result.noalias() += 2 * rho[2] * (weightedErrror * weightedErrror.transpose());
        return result;
      }

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };


  template<typename E>
  class BaseEdge<-1,E> : public OptimizableGraph::Edge
  {
    public:

      static const int Dimension = -1;
      typedef E Measurement;
      typedef Eigen::Matrix<number_t, Eigen::Dynamic, 1, Eigen::ColMajor> ErrorVector;
      typedef Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> InformationType;

      BaseEdge() : OptimizableGraph::Edge(){

      }

      virtual ~BaseEdge() {}

      virtual number_t chi2() const
      {
        return _error.dot(information()*_error);
      }

      virtual const number_t* errorData() const { return _error.data();}
      virtual number_t* errorData() { return _error.data();}
      const ErrorVector& error() const { return _error;}
      ErrorVector& error() { return _error;}

      //! information matrix of the constraint
      const InformationType& information() const { return _information;}
      InformationType& information() { return _information;}
      void setInformation(const InformationType& information) { _information = information;}

      virtual const number_t* informationData() const { return _information.data();}
      virtual number_t* informationData() { return _information.data();}

      //! accessor functions for the measurement represented by the edge
      const Measurement& measurement() const { return _measurement;}
      virtual void setMeasurement(const Measurement& m) { _measurement = m;}

      virtual int rank() const {return _dimension;}

      virtual void initialEstimate(const OptimizableGraph::VertexSet&, OptimizableGraph::Vertex*)
      {
        std::cerr << "inititialEstimate() is not implemented, please give implementation in your derived class" << std::endl;
      }

    protected:

      Measurement _measurement;
      InformationType _information;
      ErrorVector _error;

      /**
       * calculate the robust information matrix by updating the information matrix of the error
       */
      InformationType robustInformation(const Vector3& rho)
      {
        InformationType result = rho[1] * _information;
        //ErrorVector weightedErrror = _information * _error;
        //result.noalias() += 2 * rho[2] * (weightedErrror * weightedErrror.transpose());
        return result;
      }

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };


} // end namespace g2o

#endif
