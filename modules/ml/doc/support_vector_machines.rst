Support Vector Machines
=======================

.. highlight:: cpp

Originally, support vector machines (SVM) was a technique for building an optimal binary (2-class) classifier. Later the technique was extended to regression and clustering problems. SVM is a partial case of kernel-based methods. It maps feature vectors into a higher-dimensional space using a kernel function and builds an optimal linear discriminating function in this space or an optimal hyper-plane that fits into the training data. In case of SVM, the kernel is not defined explicitly. Instead, a distance between any 2 points in the hyper-space needs to be defined.

The solution is optimal, which means that the margin between the separating hyper-plane and the nearest feature vectors from both classes (in case of 2-class classifier) is maximal. The feature vectors that are the closest to the hyper-plane are called *support vectors*, which means that the position of other vectors does not affect the hyper-plane (the decision function).

There are a lot of good references on SVM. You may consider starting with the following:

*
    [Burges98] C. Burges. *A tutorial on support vector machines for pattern recognition*, Knowledge Discovery and Data Mining 2(2), 1998.
    (available online at
    http://citeseer.ist.psu.edu/burges98tutorial.html
    ).

*
    Chih-Chung Chang and Chih-Jen Lin. *LIBSVM - A Library for Support Vector Machines* 
    (
    http://www.csie.ntu.edu.tw/~cjlin/libsvm/
    )

For details of implementation and various SVM formulations see:

.. _LIBSVM:

*
    [LibSVM] C.-C. Chang and C.-J. Lin. *LIBSVM: a library for support vector machines*, ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. 
    (
    http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf
    )

CvSVMParams
-----------
.. ocv:class:: CvSVMParams

SVM training parameters.

The structure must be initialized and passed to the training method of :ocv:class:`CvSVM`.

CvSVMParams::CvSVMParams
------------------------
The constructors.

.. ocv:function:: CvSVMParams::CvSVMParams()

.. ocv:function:: CvSVMParams::CvSVMParams( int svm_type, int kernel_type, double degree, double gamma, double coef0, double Cvalue, double nu, double p, CvMat* class_weights, CvTermCriteria term_crit );

    :param svm_type: Type of a SVM formulation. Possible values are:

        * **CvSVM::C_SVC** C-Support Vector Classification.
        * **CvSVM::NU_SVC** :math:`\nu`-Support Vector Classification.
        * **CvSVM::ONE_CLASS** Distribution Estimation (One-class SVM)
        * **CvSVM::EPS_SVR** :math:`\epsilon`-Support Vector Regression
        * **CvSVM::NU_SVR** :math:`\nu`-Support Vector Regression

        See :ref:`[LibSVM] <LibSVM>` for details.

    :param kernel_type: Type of a SVM kernel. Possible values are:

        * **CvSVM::LINEAR** Linear kernel: :math:`K(x_i, x_j) = x_i^T x_j`.
        * **CvSVM::POLY** Polynomial kernel: :math:`K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0`.
        * **CvSVM::RBF** Radial basis function (RBF): :math:`K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0`.
        * **CvSVM::SIGMOID** Sigmoid kernel: :math:`K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0)`.
 
    :param degree: Parameter ``degree`` of a kernel function (POLY).

    :param gamma: Parameter :math:`\gamma` of a kernel function (POLY / RBF / SIGMOID).

    :param coef0: Parameter ``coef0`` of a kernel function (POLY / SIGMOID).

    :param Cvalue: Parameter ``C`` of a SVM formulation (C_SVC / EPS_SVR / NU_SVR).

    :param nu: Parameter :math:`\nu` of a SVM formulation (NU_SVC / ONE_CLASS / NU_SVR).

    :param p: Parameter :math:`\epsilon` of a SVM formulation (EPS_SVR)

    :param class_weights: Sets the parameter ``C`` of class ``#i`` to :math:`class\_weights_i * C` (C_SVC).

    :param term_crit: Termination criteria of SVM training optimization loop: you can specify tolerance and/or the maximum number of iterations.

The default constructor initialize the structure with following values:

::

    CvSVMParams::CvSVMParams() :
        svm_type(CvSVM::C_SVC), kernel_type(CvSVM::RBF), degree(0),
        gamma(1), coef0(0), C(1), nu(0), p(0), class_weights(0)
    {
        term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
    }



CvSVM
-----
.. ocv:class:: CvSVM

Support Vector Machines. ::

    class CvSVM : public CvStatModel
    {
    public:
        // SVM type
        enum { C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104 };

        // SVM kernel type
        enum { LINEAR=0, POLY=1, RBF=2, SIGMOID=3 };

        // SVM params type
        enum { C=0, GAMMA=1, P=2, NU=3, COEF=4, DEGREE=5 };

        CvSVM();
        virtual ~CvSVM();

        CvSVM( const Mat& _train_data, const Mat& _responses,
               const Mat& _var_idx=Mat(), const Mat& _sample_idx=Mat(),
               CvSVMParams _params=CvSVMParams() );

        virtual bool train( const Mat& _train_data, const Mat& _responses,
                            const Mat& _var_idx=Mat(), const Mat& _sample_idx=Mat(),
                            CvSVMParams _params=CvSVMParams() );

        virtual bool train_auto( const Mat& _train_data, const Mat& _responses,
            const Mat& _var_idx, const Mat& _sample_idx, CvSVMParams _params,
            int k_fold = 10,
            CvParamGrid C_grid      = get_default_grid(CvSVM::C),
            CvParamGrid gamma_grid  = get_default_grid(CvSVM::GAMMA),
            CvParamGrid p_grid      = get_default_grid(CvSVM::P),
            CvParamGrid nu_grid     = get_default_grid(CvSVM::NU),
            CvParamGrid coef_grid   = get_default_grid(CvSVM::COEF),
            CvParamGrid degree_grid = get_default_grid(CvSVM::DEGREE) );

        virtual float predict( const Mat& _sample ) const;
        virtual int get_support_vector_count() const;
        virtual const float* get_support_vector(int i) const;
        virtual CvSVMParams get_params() const { return params; };
        virtual void clear();

        static CvParamGrid get_default_grid( int param_id );

        virtual void save( const char* filename, const char* name=0 );
        virtual void load( const char* filename, const char* name=0 );

        virtual void write( CvFileStorage* storage, const char* name );
        virtual void read( CvFileStorage* storage, CvFileNode* node );
        int get_var_count() const { return var_idx ? var_idx->cols : var_all; }

    protected:
        ...
    };


CvSVM::train
------------
Trains an SVM.

.. ocv:function:: bool CvSVM::train(  const Mat& _train_data,  const Mat& _responses,                     const Mat& _var_idx=Mat(),  const Mat& _sample_idx=Mat(),                     CvSVMParams _params=CvSVMParams() )

.. ocv:pyfunction:: cv2.CvSVM.train(trainData, responses[, varIdx[, sampleIdx[, params]]]) -> retval

The method trains the SVM model. It follows the conventions of the generic ``train`` approach with the following limitations: 

* Only the ``CV_ROW_SAMPLE`` data layout is supported.

* Input variables are all ordered.

* Output variables can be either categorical ( ``_params.svm_type=CvSVM::C_SVC`` or ``_params.svm_type=CvSVM::NU_SVC`` ), or ordered ( ``_params.svm_type=CvSVM::EPS_SVR`` or ``_params.svm_type=CvSVM::NU_SVR`` ), or not required at all ( ``_params.svm_type=CvSVM::ONE_CLASS`` ).

* Missing measurements are not supported.

All the other parameters are gathered in the
:ocv:class:`CvSVMParams` structure.


CvSVM::train_auto
-----------------
Trains an SVM with optimal parameters.

.. ocv:function:: train_auto(  const Mat& _train_data,  const Mat& _responses,          const Mat& _var_idx,  const Mat& _sample_idx,          CvSVMParams params,  int k_fold = 10,          CvParamGrid C_grid      = get_default_grid(CvSVM::C),          CvParamGrid gamma_grid  = get_default_grid(CvSVM::GAMMA),          CvParamGrid p_grid      = get_default_grid(CvSVM::P),          CvParamGrid nu_grid     = get_default_grid(CvSVM::NU),          CvParamGrid coef_grid   = get_default_grid(CvSVM::COEF),          CvParamGrid degree_grid = get_default_grid(CvSVM::DEGREE) )

    :param k_fold: Cross-validation parameter. The training set is divided into  ``k_fold``  subsets. One subset is used to train the model, the others form the test set. So, the SVM algorithm is executed  ``k_fold``  times.

The method trains the SVM model automatically by choosing the optimal
parameters ``C`` , ``gamma`` , ``p`` , ``nu`` , ``coef0`` , ``degree`` from
:ocv:class:`CvSVMParams`. Parameters are considered optimal
when the cross-validation estimate of the test set error
is minimal. The parameters are iterated by a logarithmic grid, for
example, the parameter ``gamma`` takes values in the set
(
:math:`min`,
:math:`min*step`,
:math:`min*{step}^2` , ...
:math:`min*{step}^n` )
where
:math:`min` is ``gamma_grid.min_val`` ,
:math:`step` is ``gamma_grid.step`` , and
:math:`n` is the maximal index where

.. math::

    \texttt{gamma\_grid.min\_val} * \texttt{gamma\_grid.step} ^n <  \texttt{gamma\_grid.max\_val}

So ``step`` must always be greater than 1.

If there is no need to optimize a parameter, the corresponding grid step should be set to any value less than or equal to 1. For example, to avoid optimization in ``gamma`` , set ``gamma_grid.step = 0`` , ``gamma_grid.min_val`` , ``gamma_grid.max_val`` as arbitrary numbers. In this case, the value ``params.gamma`` is taken for ``gamma`` .

And, finally, if the optimization in a parameter is required but
the corresponding grid is unknown, you may call the function ``CvSVM::get_default_grid`` . To generate a grid, for example, for ``gamma`` , call ``CvSVM::get_default_grid(CvSVM::GAMMA)`` .

This function works for the classification
( ``params.svm_type=CvSVM::C_SVC`` or ``params.svm_type=CvSVM::NU_SVC`` )
as well as for the regression
( ``params.svm_type=CvSVM::EPS_SVR`` or ``params.svm_type=CvSVM::NU_SVR`` ). If ``params.svm_type=CvSVM::ONE_CLASS`` , no optimization is made and the usual SVM with parameters specified in ``params``  is executed.

CvSVM::get_default_grid
-----------------------
Generates a grid for SVM parameters.

.. ocv:function:: CvParamGrid CvSVM::get_default_grid( int param_id )

    :param param_id: SVN parameters IDs that must be one of the following:

            * **CvSVM::C**

            * **CvSVM::GAMMA**

            * **CvSVM::P**

            * **CvSVM::NU**

            * **CvSVM::COEF**

            * **CvSVM::DEGREE**

        The grid is generated for the parameter with this ID.

The function generates a grid for the specified parameter of the SVM algorithm. The grid may be passed to the function ``CvSVM::train_auto`` .

CvSVM::get_params
-----------------
Returns the current SVM parameters.

.. ocv:function:: CvSVMParams CvSVM::get_params() const

This function may be used to get the optimal parameters obtained while automatically training ``CvSVM::train_auto`` .

CvSVM::get_support_vector
--------------------------
Retrieves a number of support vectors and the particular vector.

.. ocv:function:: int CvSVM::get_support_vector_count() const

.. ocv:function:: const float* CvSVM::get_support_vector(int i) const

The methods can be used to retrieve a set of support vectors.

