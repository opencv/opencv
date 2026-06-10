// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_3D_DETAIL_OPTIMIZER_HPP
#define OPENCV_3D_DETAIL_OPTIMIZER_HPP

#include "opencv2/core/affine.hpp"
#include "opencv2/core/quaternion.hpp"
#include "opencv2/geometry/3d.hpp"

namespace cv
{
namespace detail
{

/*
This class provides functions required for Levenberg-Marquadt algorithm implementation.
See LevMarqBase::optimize() source code for details.
*/
class CV_EXPORTS LevMarqBackend
{
public:
    virtual ~LevMarqBackend() { }

    // enables geodesic acceleration support in a backend, returns true on success
    virtual bool enableGeo() = 0;

    // calculates an energy and/or jacobian at probe param vector
    virtual bool calcFunc(double& energy, bool calcEnergy = true, bool calcJacobian = false) = 0;

    // adds x to current variables and writes the sum to probe var
    // or to geodesic acceleration var if geo flag is set
    virtual void currentOplusX(const Mat_<double>& x, bool geo = false) = 0;

    // allocates jtj, jtb and other resources for objective function calculation, sets probeX to current X
    virtual void prepareVars() = 0;
    // returns a J^T*b vector (aka gradient)
    virtual const Mat_<double> getJtb() = 0;
    // returns a J^T*J diagonal vector
    virtual const Mat_<double> getDiag() = 0;
    // sets a J^T*J diagonal
    virtual void setDiag(const Mat_<double>& d) = 0;
    // performs jacobi scaling if the option is turned on
    virtual void doJacobiScaling(const Mat_<double>& di) = 0;

    // decomposes LevMarq matrix before solution
    virtual bool decompose() = 0;
    // solves LevMarq equation (J^T*J + lmdiag) * x = -right for current iteration using existing decomposition
    // right can be equal to J^T*b for LevMarq equation or J^T*rvv for geodesic acceleration equation
    virtual bool solveDecomposed(const Mat_<double>& right, Mat_<double>& x) = 0;

    // calculates J^T*f(geo) where geo is geodesic acceleration variable
    // this is used for J^T*rvv calculation for geodesic acceleration
    // calculates J^T*rvv where rvv is second directional derivative of the function in direction v
    // rvv = (f(x0 + v*h) - f(x0))/h - J*v)/h
    // where v is a LevMarq equation solution
    virtual bool calcJtbv(Mat_<double>& jtbv) = 0;

    // sets current params vector to probe params
    virtual void acceptProbe() = 0;
};

/** @brief Base class for Levenberg-Marquadt solvers.

This class can be used for general local optimization using sparse linear solvers, exponential param update or fixed variables
implemented in child classes.
This base class does not depend on a type, layout or a group structure of a param vector or an objective function jacobian.
A child class should provide a storage for that data and implement all virtual member functions that process it.
This class does not support fixed/masked variables, this should also be implemented in child classes.
*/
class CV_EXPORTS LevMarqBase
{
public:
    virtual ~LevMarqBase() { }

    // runs optimization using given termination conditions
    virtual LevMarq::Report optimize();

    LevMarqBase(const Ptr<LevMarqBackend>& backend_, const LevMarq::Settings& settings_):
        backend(backend_), settings(settings_)
    { }

    Ptr<LevMarqBackend> backend;
    LevMarq::Settings settings;
};

}  // namespace detail
}  // namespace cv

#endif // include guard
