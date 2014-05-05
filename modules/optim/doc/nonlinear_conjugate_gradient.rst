Nonlinear Conjugate Gradient
===============================

.. highlight:: cpp

optim::ConjGradSolver
---------------------------------

.. ocv:class:: optim::ConjGradSolver

This class is used to perform the non-linear non-constrained *minimization* of a function with *known gradient*
, defined on an *n*-dimensional Euclidean space,
using the **Nonlinear Conjugate Gradient method**. The implementation was done based on the beautifully clear explanatory article `An Introduction to the Conjugate Gradient Method Without the Agonizing Pain <http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf>`_
by Jonathan Richard Shewchuk. The method can be seen as an adaptation of a standard Conjugate Gradient method (see, for example
`http://en.wikipedia.org/wiki/Conjugate_gradient_method <http://en.wikipedia.org/wiki/Conjugate_gradient_method>`_) for numerically solving the
systems of linear equations.

It should be noted, that
this method, although deterministic, is rather a heuristic method and therefore may converge to a local minima, not necessary a global one. What
is even more disastrous, most of its behaviour is ruled by gradient, therefore it essentially cannot distinguish between local minima and maxima.
Therefore, if it starts sufficiently near to the local maximum, it may converge to it. Another obvious restriction is that it should be possible
to compute the gradient of a function at any point, thus it is preferable to have analytic expression for gradient and computational burden
should be born by the user.

The latter responsibility is accompilished via the ``getGradient(const double* x,double* grad)`` method of a
``Solver::Function`` interface (which represents function that is being optimized). This method takes point a point in *n*-dimensional space
(first argument represents the array of coordinates of that point) and comput its gradient (it should be stored in the second argument as an array).

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

    class CV_EXPORTS ConjGradSolver : public Solver{
    };

Note, that class ``ConjGradSolver`` thus does not add any new methods to the basic ``Solver`` interface.

optim::ConjGradSolver::getFunction
--------------------------------------------

Getter for the optimized function. The optimized function is represented by ``Solver::Function`` interface, which requires
derivatives to implement the method ``calc(double*)`` to evaluate the function. It should be emphasized once more, that since Nonlinear
Conjugate Gradient method requires gradient to be computable in addition to the function values,
``getGradient(const double* x,double* grad)`` method of a ``Solver::Function`` interface should be also implemented meaningfully.

.. ocv:function:: Ptr<Solver::Function> optim::ConjGradSolver::getFunction()

    :return: Smart-pointer to an object that implements ``Solver::Function`` interface - it represents the function that is being optimized. It can be empty, if no function was given so far.

optim::ConjGradSolver::setFunction
-----------------------------------------------

Setter for the optimized function. *It should be called at least once before the call to* ``ConjGradSolver::minimize()``, as
default value is not usable.

.. ocv:function:: void optim::ConjGradSolver::setFunction(const Ptr<Solver::Function>& f)

    :param f: The new function to optimize.

optim::ConjGradSolver::getTermCriteria
----------------------------------------------------

Getter for the previously set terminal criteria for this algorithm.

.. ocv:function:: TermCriteria optim::ConjGradSolver::getTermCriteria()

    :return: Deep copy of the terminal criteria used at the moment.

optim::ConjGradSolver::setTermCriteria
------------------------------------------

Set terminal criteria for downhill simplex method. Two things should be noted. First, this method *is not necessary* to be called
before the first call to ``ConjGradSolver::minimize()``, as the default value is sensible. Second, the method will raise an error
if ``termcrit.type!=(TermCriteria::MAX_ITER+TermCriteria::EPS)`` and ``termcrit.type!=TermCriteria::MAX_ITER``. This means that termination criteria
has to restrict maximum number of iterations to be done and may optionally allow algorithm to stop earlier if certain tolerance
is achieved (what we mean by "tolerance is achieved" will be clarified below). If ``termcrit`` restricts both tolerance and maximum iteration
number, both ``termcrit.epsilon`` and ``termcrit.maxCount`` should be positive. In case, if ``termcrit.type==TermCriteria::MAX_ITER``,
only member ``termcrit.maxCount`` is required to be positive and in this case algorithm will just work for required number of iterations.

In current implementation, "tolerance is achieved" means that we have arrived at the point where the :math:`L_2`-norm of the gradient is less
than the tolerance value.

.. ocv:function:: void optim::ConjGradSolver::setTermCriteria(const TermCriteria& termcrit)

    :param termcrit: Terminal criteria to be used, represented as ``TermCriteria`` structure (defined elsewhere in openCV). Mind you, that it should meet ``termcrit.type==(TermCriteria::MAX_ITER+TermCriteria::EPS) && termcrit.epsilon>0 && termcrit.maxCount>0`` or ``termcrit.type==TermCriteria::MAX_ITER) && termcrit.maxCount>0``, otherwise the error will be raised.

optim::ConjGradSolver::minimize
-----------------------------------

The main method of the ``ConjGradSolver``. It actually runs the algorithm and performs the minimization. The sole input parameter determines the
centroid of the starting simplex (roughly, it tells where to start), all the others (terminal criteria and function to be minimized)
are supposed to be set via the setters before the call to this method or the default values (not always sensible) will be used. Sometimes it may
throw an error, if these default values cannot be used (say, you forgot to set the function to minimize and default value, that is, empty function,
cannot be used).

.. ocv:function:: double optim::ConjGradSolver::minimize(InputOutputArray x)

    :param x: The initial point. It is hard to overemphasize how important the choise of initial point is when you are using the heuristic algorithm like this one. Badly chosen initial point can make algorithm converge to (local) maximum instead of minimum, do not converge at all, converge to local minimum instead of global one.

    :return: The value of a function at the point found.

optim::createConjGradSolver
------------------------------------

This function returns the reference to the ready-to-use ``ConjGradSolver`` object. All the parameters are optional, so this procedure can be called
even without parameters at all. In this case, the default values will be used. As default value for terminal criteria are the only sensible ones,
``ConjGradSolver::setFunction()`` should be called upon the obtained object, if the function
was not given to ``createConjGradSolver()``. Otherwise, the two ways (submit it to ``createConjGradSolver()`` or miss it out and call the
``ConjGradSolver::setFunction()``) are absolutely equivalent (and will drop the same errors in the same way,
should invalid input be detected).

.. ocv:function:: Ptr<optim::ConjGradSolver> optim::createConjGradSolver(const Ptr<Solver::Function>& f, TermCriteria termcrit)

    :param f: Pointer to the function that will be minimized, similarly to the one you submit via ``ConjGradSolver::setFunction``.
    :param termcrit: Terminal criteria to the algorithm, similarly to the one you submit via ``ConjGradSolver::setTermCriteria``.
