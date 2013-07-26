Downhill Simplex Method
=======================

.. highlight:: cpp

optim::DownhillSolver
---------------------------------

.. ocv:class:: optim::DownhillSolver

This class is used to perform the non-linear non-constrained *minimization* of a function, given on an *n*-dimensional Euclidean space,
using the **Nelder-Mead method**, also known as **downhill simplex method**. The basic idea about the method can be obtained from
(`http://en.wikipedia.org/wiki/Nelder-Mead\_method <http://en.wikipedia.org/wiki/Nelder-Mead_method>`_). It should be noted, that
this method, although deterministic, is rather a heuristic and therefore may converge to a local minima, not necessary a global one.
It is iterative optimization technique, which at each step uses an information about the values of a function evaluated only at
*n+1* points, arranged as a *simplex* in *n*-dimensional space (hence the second name of the method). At each step new point is
chosen to evaluate function at, obtained value is compared with previous ones and based on this information simplex changes it's shape
, slowly moving to the local minimum.

Algorithm stops when the number of function evaluations done exceeds ``termcrit.maxCount``, when the function values at the
vertices of simplex are within ``termcrit.epsilon`` range or simplex becomes so small that it
can enclosed in a box with ``termcrit.epsilon`` sides, whatever comes first, for some defined by user
positive integer ``termcrit.maxCount`` and positive non-integer ``termcrit.epsilon``.

::

    class CV_EXPORTS Solver : public Algorithm
    {
    public:
        class CV_EXPORTS Function
        {
        public:
           virtual ~Function() {}
           //! ndim - dimensionality
           virtual double calc(const double* x) const = 0;     
        };

        virtual Ptr<Function> getFunction() const = 0;
        virtual void setFunction(const Ptr<Function>& f) = 0;

        virtual TermCriteria getTermCriteria() const = 0;
        virtual void setTermCriteria(const TermCriteria& termcrit) = 0;

        // x contain the initial point before the call and the minima position (if algorithm converged) after. x is assumed to be (something that
        // after getMat() will return) row-vector or column-vector. *It's size  and should
        // be consisted with previous dimensionality data given, if any (otherwise, it determines dimensionality)*
        virtual double minimize(InputOutputArray x) = 0;
    };

    class CV_EXPORTS DownhillSolver : public Solver
    {
    public:
        //! returns row-vector, even if the column-vector was given
        virtual void getInitStep(OutputArray step) const=0;
        //!This should be called at least once before the first call to minimize() and step is assumed to be (something that
        //! after getMat() will return) row-vector or column-vector. *It's dimensionality determines the dimensionality of a problem.*
        virtual void setInitStep(InputArray step)=0;
    };

It should be noted, that ``optim::DownhillSolver`` is a derivative of the abstract interface ``optim::Solver``, which in
turn is derived from the ``Algorithm`` interface and is used to encapsulate the functionality, common to all non-linear optimization
algorithms in the ``optim`` module.

optim::DownhillSolver::getFunction
--------------------------------------------

Getter for the optimized function. The optimized function is represented by ``Solver::Function`` interface, which requires 
derivatives to implement the sole method ``calc(double*)`` to evaluate the function.

.. ocv:function:: Ptr<Solver::Function> optim::DownhillSolver::getFunction()

    :return: Smart-pointer to an object that implements ``Solver::Function`` interface - it represents the function that is being
    optimized. It can be empty, if no function was given so far.

optim::DownhillSolver::setFunction
-----------------------------------------------

Setter for the optimized function. It should be called at least once before the call to ``DownhillSolver::minimize()``, as
there is no usable default value.

.. ocv:function:: void optim::DownhillSolver::setFunction(const Ptr<Solver::Function>& f)

    :param f: The new function to optimize.

optim::DownhillSolver::getTermCriteria
----------------------------------------------------

Getter for the previously set terminal criteria for this algorithm.

.. ocv:function:: TermCriteria optim::DownhillSolver::getTermCriteria()

    :return: Deep copy of the terminal criteria used at the moment.

optim::DownhillSolver::setTermCriteria
------------------------------------------

Set terminal criteria for downhill simplex method. Two things should be noted. First, this method *is not necessary* to be call before
the first call to ``DownhillSolver::minimize()``, as the default value is sensible. Second, the method will raise an error
if ``termcrit.type!=(TermCriteria::MAX_ITER+TermCriteria::EPS)``, ``termcrit.epsilon<=0`` or ``termcrit.maxCount<=0``. That is,
both ``epsilon`` and ``maxCount`` should be set to positive values (non-integer and integer respectively) and they represent
tolerance and maximal number of function evaluations that is allowed.

Algorithm stops when the number of function evaluations done exceeds ``termcrit.maxCount``, when the function values at the
vertices of simplex are within ``termcrit.epsilon`` range or simplex becomes so small that it
can enclosed in a box with ``termcrit.epsilon`` sides, whatever comes first.

.. ocv:function:: void optim::DownhillSolver::setTermCriteria(const TermCriteria& termcrit)

    :param termcrit: Terminal criteria to be used, represented as ``TermCriteria`` structure (defined elsewhere in openCV). Mind you,
    that it should meet
    ``(termcrit.type==(TermCriteria::MAX_ITER+TermCriteria::EPS) && termcrit.epsilon>0 && termcrit.maxCount>0)``, otherwise the
    error will be raised.

optim::DownhillSolver::getInitStep
-----------------------------------

Explain function.

.. ocv:function:: void optim::getInitStep(OutputArray step)

Explain parameters.

optim::DownhillSolver::setInitStep
----------------------------------

.. ocv:function:: void optim::setInitStep(InputArray step)

Explain parameters.

optim::DownhillSolver::minimize
-----------------------------------

Explain function.

.. ocv:function:: double optim::DownhillSolver::minimize(InputOutputArray x)

Explain parameters.

optim::createDownhillSolver
------------------------------------

Explain function.

.. ocv:function:: Ptr<optim::DownhillSolver> optim::createDownhillSolver(const Ptr<Solver::Function>& f,InputArray initStep, TermCriteria termcrit)

Explain parameters.
