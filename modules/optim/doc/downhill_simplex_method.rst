Downhill Simplex Method
=======================

.. highlight:: cpp

optim::DownhillSolver
---------------------------------

.. ocv:class:: optim::DownhillSolver

This class is used to perform the non-linear non-constrained *minimization* of a function, defined on an *n*-dimensional Euclidean space,
using the **Nelder-Mead method**, also known as **downhill simplex method**. The basic idea about the method can be obtained from
(`http://en.wikipedia.org/wiki/Nelder-Mead\_method <http://en.wikipedia.org/wiki/Nelder-Mead_method>`_). It should be noted, that
this method, although deterministic, is rather a heuristic and therefore may converge to a local minima, not necessary a global one.
It is iterative optimization technique, which at each step uses an information about the values of a function evaluated only at
*n+1* points, arranged as a *simplex* in *n*-dimensional space (hence the second name of the method). At each step new point is
chosen to evaluate function at, obtained value is compared with previous ones and based on this information simplex changes it's shape
, slowly moving to the local minimum. Thus this method is using *only* function values to make decision, on contrary to, say, Nonlinear
Conjugate Gradient method (which is also implemented in ``optim``).

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
            virtual double calc(const double* x) const = 0;
            virtual void getGradient(const double* /*x*/,double* /*grad*/) {}
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

    :return: Smart-pointer to an object that implements ``Solver::Function`` interface - it represents the function that is being optimized. It can be empty, if no function was given so far.

optim::DownhillSolver::setFunction
-----------------------------------------------

Setter for the optimized function. *It should be called at least once before the call to* ``DownhillSolver::minimize()``, as
default value is not usable.

.. ocv:function:: void optim::DownhillSolver::setFunction(const Ptr<Solver::Function>& f)

    :param f: The new function to optimize.

optim::DownhillSolver::getTermCriteria
----------------------------------------------------

Getter for the previously set terminal criteria for this algorithm.

.. ocv:function:: TermCriteria optim::DownhillSolver::getTermCriteria()

    :return: Deep copy of the terminal criteria used at the moment.

optim::DownhillSolver::setTermCriteria
------------------------------------------

Set terminal criteria for downhill simplex method. Two things should be noted. First, this method *is not necessary* to be called
before the first call to ``DownhillSolver::minimize()``, as the default value is sensible. Second, the method will raise an error
if ``termcrit.type!=(TermCriteria::MAX_ITER+TermCriteria::EPS)``, ``termcrit.epsilon<=0`` or ``termcrit.maxCount<=0``. That is,
both ``epsilon`` and ``maxCount`` should be set to positive values (non-integer and integer respectively) and they represent
tolerance and maximal number of function evaluations that is allowed.

Algorithm stops when the number of function evaluations done exceeds ``termcrit.maxCount``, when the function values at the
vertices of simplex are within ``termcrit.epsilon`` range or simplex becomes so small that it
can enclosed in a box with ``termcrit.epsilon`` sides, whatever comes first.

.. ocv:function:: void optim::DownhillSolver::setTermCriteria(const TermCriteria& termcrit)

    :param termcrit: Terminal criteria to be used, represented as ``TermCriteria`` structure (defined elsewhere in openCV). Mind you, that it should meet ``(termcrit.type==(TermCriteria::MAX_ITER+TermCriteria::EPS) && termcrit.epsilon>0 && termcrit.maxCount>0)``, otherwise the error will be raised.

optim::DownhillSolver::getInitStep
-----------------------------------

Returns the initial step that will be used in downhill simplex algorithm. See the description
of corresponding setter (follows next) for the meaning of this parameter.

.. ocv:function:: void optim::getInitStep(OutputArray step)

    :param step: Initial step that will be used in algorithm. Note, that although corresponding setter accepts column-vectors as well as row-vectors, this method will return a row-vector.

optim::DownhillSolver::setInitStep
----------------------------------

Sets the initial step that will be used in downhill simplex algorithm. Step, together with initial point (givin in ``DownhillSolver::minimize``)
are two *n*-dimensional vectors that are used to determine the shape of initial simplex. Roughly said, initial point determines the position
of a simplex (it will become simplex's centroid), while step determines the spread (size in each dimension) of a simplex. To be more precise,
if :math:`s,x_0\in\mathbb{R}^n` are the initial step and initial point respectively, the vertices of a simplex will be: :math:`v_0:=x_0-\frac{1}{2}
s` and :math:`v_i:=x_0+s_i` for :math:`i=1,2,\dots,n` where :math:`s_i` denotes projections of the initial step of *n*-th coordinate (the result
of projection is treated to be vector given by :math:`s_i:=e_i\cdot\left<e_i\cdot s\right>`, where :math:`e_i` form canonical basis)

.. ocv:function:: void optim::setInitStep(InputArray step)

    :param step: Initial step that will be used in algorithm. Roughly said, it determines the spread (size in each dimension) of an initial simplex.

optim::DownhillSolver::minimize
-----------------------------------

The main method of the ``DownhillSolver``. It actually runs the algorithm and performs the minimization. The sole input parameter determines the
centroid of the starting simplex (roughly, it tells where to start), all the others (terminal criteria, initial step, function to be minimized)
are supposed to be set via the setters before the call to this method or the default values (not always sensible) will be used.

.. ocv:function:: double optim::DownhillSolver::minimize(InputOutputArray x)

    :param x: The initial point, that will become a centroid of an initial simplex. After the algorithm will terminate, it will be setted to the point where the algorithm stops, the point of possible minimum.

    :return: The value of a function at the point found.

optim::createDownhillSolver
------------------------------------

This function returns the reference to the ready-to-use ``DownhillSolver`` object. All the parameters are optional, so this procedure can be called
even without parameters at all. In this case, the default values will be used. As default value for terminal criteria are the only sensible ones,
``DownhillSolver::setFunction()`` and ``DownhillSolver::setInitStep()`` should be called upon the obtained object, if the respective parameters
were not given to ``createDownhillSolver()``. Otherwise, the two ways (give parameters to ``createDownhillSolver()`` or miss them out and call the
``DownhillSolver::setFunction()`` and ``DownhillSolver::setInitStep()``) are absolutely equivalent (and will drop the same errors in the same way,
should invalid input be detected).

.. ocv:function:: Ptr<optim::DownhillSolver> optim::createDownhillSolver(const Ptr<Solver::Function>& f,InputArray initStep, TermCriteria termcrit)

    :param f: Pointer to the function that will be minimized, similarly to the one you submit via ``DownhillSolver::setFunction``.
    :param step: Initial step, that will be used to construct the initial simplex, similarly to the one you submit via ``DownhillSolver::setInitStep``.
    :param termcrit: Terminal criteria to the algorithm, similarly to the one you submit via ``DownhillSolver::setTermCriteria``.
