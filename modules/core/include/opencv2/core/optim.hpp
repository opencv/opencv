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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_OPTIM_HPP
#define OPENCV_OPTIM_HPP

#include "opencv2/core.hpp"

namespace cv
{

/** @addtogroup core_optim
The algorithms in this section minimize or maximize function value within specified constraints or
without any constraints.
@{
*/

/** @brief Basic interface for all solvers
 */
class CV_EXPORTS MinProblemSolver : public Algorithm
{
public:
    /** @brief Represents function being optimized
     */
    class CV_EXPORTS Function
    {
    public:
        virtual ~Function() {}
        virtual int getDims() const = 0;
        virtual double getGradientEps() const;
        virtual double calc(const double* x) const = 0;
        virtual void getGradient(const double* x,double* grad);
    };

    /** @brief Getter for the optimized function.

    The optimized function is represented by Function interface, which requires derivatives to
    implement the calc(double*) and getDim() methods to evaluate the function.

    @return Smart-pointer to an object that implements Function interface - it represents the
    function that is being optimized. It can be empty, if no function was given so far.
     */
    virtual Ptr<Function> getFunction() const = 0;

    /** @brief Setter for the optimized function.

    *It should be called at least once before the call to* minimize(), as default value is not usable.

    @param f The new function to optimize.
     */
    virtual void setFunction(const Ptr<Function>& f) = 0;

    /** @brief Getter for the previously set terminal criteria for this algorithm.

    @return Deep copy of the terminal criteria used at the moment.
     */
    virtual TermCriteria getTermCriteria() const = 0;

    /** @brief Set terminal criteria for solver.

    This method *is not necessary* to be called before the first call to minimize(), as the default
    value is sensible.

    Algorithm stops when the number of function evaluations done exceeds termcrit.maxCount, when
    the function values at the vertices of simplex are within termcrit.epsilon range or simplex
    becomes so small that it can enclosed in a box with termcrit.epsilon sides, whatever comes
    first.
    @param termcrit Terminal criteria to be used, represented as cv::TermCriteria structure.
     */
    virtual void setTermCriteria(const TermCriteria& termcrit) = 0;

    /** @brief actually runs the algorithm and performs the minimization.

    The sole input parameter determines the centroid of the starting simplex (roughly, it tells
    where to start), all the others (terminal criteria, initial step, function to be minimized) are
    supposed to be set via the setters before the call to this method or the default values (not
    always sensible) will be used.

    @param x The initial point, that will become a centroid of an initial simplex. After the algorithm
    will terminate, it will be set to the point where the algorithm stops, the point of possible
    minimum.
    @return The value of a function at the point found.
     */
    virtual double minimize(InputOutputArray x) = 0;
};

/** @brief This class is used to perform the non-linear non-constrained minimization of a function,

defined on an `n`-dimensional Euclidean space, using the **Nelder-Mead method**, also known as
**downhill simplex method**. The basic idea about the method can be obtained from
<http://en.wikipedia.org/wiki/Nelder-Mead_method>.

It should be noted, that this method, although deterministic, is rather a heuristic and therefore
may converge to a local minima, not necessary a global one. It is iterative optimization technique,
which at each step uses an information about the values of a function evaluated only at `n+1`
points, arranged as a *simplex* in `n`-dimensional space (hence the second name of the method). At
each step new point is chosen to evaluate function at, obtained value is compared with previous
ones and based on this information simplex changes it's shape , slowly moving to the local minimum.
Thus this method is using *only* function values to make decision, on contrary to, say, Nonlinear
Conjugate Gradient method (which is also implemented in optim).

Algorithm stops when the number of function evaluations done exceeds termcrit.maxCount, when the
function values at the vertices of simplex are within termcrit.epsilon range or simplex becomes so
small that it can enclosed in a box with termcrit.epsilon sides, whatever comes first, for some
defined by user positive integer termcrit.maxCount and positive non-integer termcrit.epsilon.

@note DownhillSolver is a derivative of the abstract interface
cv::MinProblemSolver, which in turn is derived from the Algorithm interface and is used to
encapsulate the functionality, common to all non-linear optimization algorithms in the optim
module.

@note term criteria should meet following condition:
@code
    termcrit.type == (TermCriteria::MAX_ITER + TermCriteria::EPS) && termcrit.epsilon > 0 && termcrit.maxCount > 0
@endcode
 */
class CV_EXPORTS DownhillSolver : public MinProblemSolver
{
public:
    /** @brief Returns the initial step that will be used in downhill simplex algorithm.

    @param step Initial step that will be used in algorithm. Note, that although corresponding setter
    accepts column-vectors as well as row-vectors, this method will return a row-vector.
    @see DownhillSolver::setInitStep
     */
    virtual void getInitStep(OutputArray step) const=0;

    /** @brief Sets the initial step that will be used in downhill simplex algorithm.

    Step, together with initial point (given in DownhillSolver::minimize) are two `n`-dimensional
    vectors that are used to determine the shape of initial simplex. Roughly said, initial point
    determines the position of a simplex (it will become simplex's centroid), while step determines the
    spread (size in each dimension) of a simplex. To be more precise, if \f$s,x_0\in\mathbb{R}^n\f$ are
    the initial step and initial point respectively, the vertices of a simplex will be:
    \f$v_0:=x_0-\frac{1}{2} s\f$ and \f$v_i:=x_0+s_i\f$ for \f$i=1,2,\dots,n\f$ where \f$s_i\f$ denotes
    projections of the initial step of *n*-th coordinate (the result of projection is treated to be
    vector given by \f$s_i:=e_i\cdot\left<e_i\cdot s\right>\f$, where \f$e_i\f$ form canonical basis)

    @param step Initial step that will be used in algorithm. Roughly said, it determines the spread
    (size in each dimension) of an initial simplex.
     */
    virtual void setInitStep(InputArray step)=0;

    /** @brief This function returns the reference to the ready-to-use DownhillSolver object.

    All the parameters are optional, so this procedure can be called even without parameters at
    all. In this case, the default values will be used. As default value for terminal criteria are
    the only sensible ones, MinProblemSolver::setFunction() and DownhillSolver::setInitStep()
    should be called upon the obtained object, if the respective parameters were not given to
    create(). Otherwise, the two ways (give parameters to createDownhillSolver() or miss them out
    and call the MinProblemSolver::setFunction() and DownhillSolver::setInitStep()) are absolutely
    equivalent (and will drop the same errors in the same way, should invalid input be detected).
    @param f Pointer to the function that will be minimized, similarly to the one you submit via
    MinProblemSolver::setFunction.
    @param initStep Initial step, that will be used to construct the initial simplex, similarly to the one
    you submit via MinProblemSolver::setInitStep.
    @param termcrit Terminal criteria to the algorithm, similarly to the one you submit via
    MinProblemSolver::setTermCriteria.
     */
    static Ptr<DownhillSolver> create(const Ptr<MinProblemSolver::Function>& f=Ptr<MinProblemSolver::Function>(),
                                      InputArray initStep=Mat_<double>(1,1,0.0),
                                      TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5000,0.000001));
};

/** @brief This class is used to perform the non-linear non-constrained minimization of a function
with known gradient,

defined on an *n*-dimensional Euclidean space, using the **Nonlinear Conjugate Gradient method**.
The implementation was done based on the beautifully clear explanatory article [An Introduction to
the Conjugate Gradient Method Without the Agonizing
Pain](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf) by Jonathan Richard
Shewchuk. The method can be seen as an adaptation of a standard Conjugate Gradient method (see, for
example <http://en.wikipedia.org/wiki/Conjugate_gradient_method>) for numerically solving the
systems of linear equations.

It should be noted, that this method, although deterministic, is rather a heuristic method and
therefore may converge to a local minima, not necessary a global one. What is even more disastrous,
most of its behaviour is ruled by gradient, therefore it essentially cannot distinguish between
local minima and maxima. Therefore, if it starts sufficiently near to the local maximum, it may
converge to it. Another obvious restriction is that it should be possible to compute the gradient of
a function at any point, thus it is preferable to have analytic expression for gradient and
computational burden should be born by the user.

The latter responsibility is accomplished via the getGradient method of a
MinProblemSolver::Function interface (which represents function being optimized). This method takes
point a point in *n*-dimensional space (first argument represents the array of coordinates of that
point) and compute its gradient (it should be stored in the second argument as an array).

@note class ConjGradSolver thus does not add any new methods to the basic MinProblemSolver interface.

@note term criteria should meet following condition:
@code
    termcrit.type == (TermCriteria::MAX_ITER + TermCriteria::EPS) && termcrit.epsilon > 0 && termcrit.maxCount > 0
    // or
    termcrit.type == TermCriteria::MAX_ITER) && termcrit.maxCount > 0
@endcode
 */
class CV_EXPORTS ConjGradSolver : public MinProblemSolver
{
public:
    /** @brief This function returns the reference to the ready-to-use ConjGradSolver object.

    All the parameters are optional, so this procedure can be called even without parameters at
    all. In this case, the default values will be used. As default value for terminal criteria are
    the only sensible ones, MinProblemSolver::setFunction() should be called upon the obtained
    object, if the function was not given to create(). Otherwise, the two ways (submit it to
    create() or miss it out and call the MinProblemSolver::setFunction()) are absolutely equivalent
    (and will drop the same errors in the same way, should invalid input be detected).
    @param f Pointer to the function that will be minimized, similarly to the one you submit via
    MinProblemSolver::setFunction.
    @param termcrit Terminal criteria to the algorithm, similarly to the one you submit via
    MinProblemSolver::setTermCriteria.
    */
    static Ptr<ConjGradSolver> create(const Ptr<MinProblemSolver::Function>& f=Ptr<ConjGradSolver::Function>(),
                                      TermCriteria termcrit=TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,5000,0.000001));
};

//! return codes for cv::solveLP() function
enum SolveLPResult
{
    SOLVELP_UNBOUNDED    = -2, //!< problem is unbounded (target function can achieve arbitrary high values)
    SOLVELP_UNFEASIBLE    = -1, //!< problem is unfeasible (there are no points that satisfy all the constraints imposed)
    SOLVELP_SINGLE    = 0, //!< there is only one maximum for target function
    SOLVELP_MULTI    = 1 //!< there are multiple maxima for target function - the arbitrary one is returned
};

/** @brief Solve given (non-integer) linear programming problem using the Simplex Algorithm (Simplex Method).

What we mean here by "linear programming problem" (or LP problem, for short) can be formulated as:

\f[\mbox{Maximize } c\cdot x\\
 \mbox{Subject to:}\\
 Ax\leq b\\
 x\geq 0\f]

Where \f$c\f$ is fixed `1`-by-`n` row-vector, \f$A\f$ is fixed `m`-by-`n` matrix, \f$b\f$ is fixed `m`-by-`1`
column vector and \f$x\f$ is an arbitrary `n`-by-`1` column vector, which satisfies the constraints.

Simplex algorithm is one of many algorithms that are designed to handle this sort of problems
efficiently. Although it is not optimal in theoretical sense (there exist algorithms that can solve
any problem written as above in polynomial time, while simplex method degenerates to exponential
time for some special cases), it is well-studied, easy to implement and is shown to work well for
real-life purposes.

The particular implementation is taken almost verbatim from **Introduction to Algorithms, third
edition** by T. H. Cormen, C. E. Leiserson, R. L. Rivest and Clifford Stein. In particular, the
Bland's rule <http://en.wikipedia.org/wiki/Bland%27s_rule> is used to prevent cycling.

@param Func This row-vector corresponds to \f$c\f$ in the LP problem formulation (see above). It should
contain 32- or 64-bit floating point numbers. As a convenience, column-vector may be also submitted,
in the latter case it is understood to correspond to \f$c^T\f$.
@param Constr `m`-by-`n+1` matrix, whose rightmost column corresponds to \f$b\f$ in formulation above
and the remaining to \f$A\f$. It should contain 32- or 64-bit floating point numbers.
@param z The solution will be returned here as a column-vector - it corresponds to \f$c\f$ in the
formulation above. It will contain 64-bit floating point numbers.
@return One of cv::SolveLPResult
 */
CV_EXPORTS_W int solveLP(InputArray Func, InputArray Constr, OutputArray z);

//! @}

}// cv

#endif
