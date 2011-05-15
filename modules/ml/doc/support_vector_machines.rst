Support Vector Machines
=======================

.. highlight:: cpp

Originally, support vector machines (SVM) was a technique for building an optimal binary (2-class) classifier. Later the technique has been extended to regression and clustering problems. SVM is a partial case of kernel-based methods. It maps feature vectors into a higher-dimensional space using a kernel function and builds an optimal linear discriminating function in this space or an optimal hyper-plane that fits into the training data. In case of SVM, the kernel is not defined explicitly. Instead, a distance between any 2 points in the hyper-space needs to be defined.

The solution is optimal, which means that the margin between the separating hyper-plane and the nearest feature vectors from both classes (in case of 2-class classifier) is maximal. The feature vectors that are the closest to the hyper-plane are called "support vectors", which means that the position of other vectors does not affect the hyper-plane (the decision function).

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

.. index:: CvSVM

.. _CvSVM:

CvSVM
-----
.. c:type:: CvSVM

Support Vector Machines ::

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

        CvSVM( const CvMat* _train_data, const CvMat* _responses,
               const CvMat* _var_idx=0, const CvMat* _sample_idx=0,
               CvSVMParams _params=CvSVMParams() );

        virtual bool train( const CvMat* _train_data, const CvMat* _responses,
                            const CvMat* _var_idx=0, const CvMat* _sample_idx=0,
                            CvSVMParams _params=CvSVMParams() );

        virtual bool train_auto( const CvMat* _train_data, const CvMat* _responses,
            const CvMat* _var_idx, const CvMat* _sample_idx, CvSVMParams _params,
            int k_fold = 10,
            CvParamGrid C_grid      = get_default_grid(CvSVM::C),
            CvParamGrid gamma_grid  = get_default_grid(CvSVM::GAMMA),
            CvParamGrid p_grid      = get_default_grid(CvSVM::P),
            CvParamGrid nu_grid     = get_default_grid(CvSVM::NU),
            CvParamGrid coef_grid   = get_default_grid(CvSVM::COEF),
            CvParamGrid degree_grid = get_default_grid(CvSVM::DEGREE) );

        virtual float predict( const CvMat* _sample ) const;
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


.. index:: CvSVMParams

.. _CvSVMParams:

CvSVMParams
-----------
.. c:type:: CvSVMParams

SVM training parameters ::

    struct CvSVMParams
    {
        CvSVMParams();
        CvSVMParams( int _svm_type, int _kernel_type,
                     double _degree, double _gamma, double _coef0,
                     double _C, double _nu, double _p,
                     CvMat* _class_weights, CvTermCriteria _term_crit );

        int         svm_type;
        int         kernel_type;
        double      degree; // for poly
        double      gamma;  // for poly/rbf/sigmoid
        double      coef0;  // for poly/sigmoid

        double      C;  // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        double      nu; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
        double      p; // for CV_SVM_EPS_SVR
        CvMat*      class_weights; // for CV_SVM_C_SVC
        CvTermCriteria term_crit; // termination criteria
    };


The structure must be initialized and passed to the training method of
:ref:`CvSVM` .

.. index:: CvSVM::train

.. _CvSVM::train:

CvSVM::train
------------
.. c:function:: bool CvSVM::train(  const CvMat* _train_data,  const CvMat* _responses,                     const CvMat* _var_idx=0,  const CvMat* _sample_idx=0,                     CvSVMParams _params=CvSVMParams() )

    Trains SVM.

The method trains the SVM model. It follows the conventions of the generic ``train`` "method" with the following limitations: 

* Only the ``CV_ROW_SAMPLE`` data layout is supported.
* Input variables are all ordered.
* Output variables can be either categorical ( ``_params.svm_type=CvSVM::C_SVC`` or ``_params.svm_type=CvSVM::NU_SVC`` ), or ordered ( ``_params.svm_type=CvSVM::EPS_SVR`` or ``_params.svm_type=CvSVM::NU_SVR`` ), or not required at all ( ``_params.svm_type=CvSVM::ONE_CLASS`` ).
* Missing measurements are not supported.

All the other parameters are gathered in the
:ref:`CvSVMParams` structure.

.. index:: CvSVM::train_auto

.. _CvSVM::train_auto:

CvSVM::train_auto
-----------------
.. c:function:: train_auto(  const CvMat* _train_data,  const CvMat* _responses,          const CvMat* _var_idx,  const CvMat* _sample_idx,          CvSVMParams params,  int k_fold = 10,          CvParamGrid C_grid      = get_default_grid(CvSVM::C),          CvParamGrid gamma_grid  = get_default_grid(CvSVM::GAMMA),          CvParamGrid p_grid      = get_default_grid(CvSVM::P),          CvParamGrid nu_grid     = get_default_grid(CvSVM::NU),          CvParamGrid coef_grid   = get_default_grid(CvSVM::COEF),          CvParamGrid degree_grid = get_default_grid(CvSVM::DEGREE) )

    Trains SVM with optimal parameters.

    :param k_fold: Cross-validation parameter. The training set is divided into  ``k_fold``  subsets. One subset is used to train the model, the others form the test set. So, the SVM algorithm is executed  ``k_fold``  times.

The method trains the SVM model automatically by choosing the optimal
parameters ``C`` , ``gamma`` , ``p`` , ``nu`` , ``coef0`` , ``degree`` from
:ref:`CvSVMParams`. Parameters are considered optimal
when the cross-validation estimate of the test set error
is minimal. The parameters are iterated by a logarithmic grid, for
example, the parameter ``gamma`` takes the values in the set
(
:math:`min`,
:math:`min*step`,
:math:`min*{step}^2` , ...
:math:`min*{step}^n` )
where
:math:`min` is ``gamma_grid.min_val`` ,
:math:`step` is ``gamma_grid.step`` , and
:math:`n` is the maximal index such that

.. math::

    \texttt{gamma\_grid.min\_val} * \texttt{gamma\_grid.step} ^n <  \texttt{gamma\_grid.max\_val}

So ``step`` must always be greater than 1.

If there is no need to optimize a parameter, the corresponding grid step should be set to any value less or equal to 1. For example, to avoid optimization in ``gamma`` , set ``gamma_grid.step = 0`` , ``gamma_grid.min_val`` , ``gamma_grid.max_val`` as arbitrary numbers. In this case, the value ``params.gamma`` is taken for ``gamma`` .

And, finally, if the optimization in a parameter is required but
the corresponding grid is unknown, you may call the function ``CvSVM::get_default_grid`` . To generate a grid, for example, for ``gamma`` , call ``CvSVM::get_default_grid(CvSVM::GAMMA)`` .

This function works for the case of classification
( ``params.svm_type=CvSVM::C_SVC`` or ``params.svm_type=CvSVM::NU_SVC`` )
as well as for the regression
( ``params.svm_type=CvSVM::EPS_SVR`` or ``params.svm_type=CvSVM::NU_SVR`` ). If ``params.svm_type=CvSVM::ONE_CLASS`` , no optimization is made and the usual SVM with parameters specified in ``params``  is executed.

.. index:: CvSVM::get_default_grid

.. _CvSVM::get_default_grid:

CvSVM::get_default_grid
-----------------------
.. c:function:: CvParamGrid CvSVM::get_default_grid( int param_id )

    Generates a grid for SVM parameters.

    :param param_id: SVN parameters IDs that must be one of the following:

            * **CvSVM::C**

            * **CvSVM::GAMMA**

            * **CvSVM::P**

            * **CvSVM::NU**

            * **CvSVM::COEF**

            * **CvSVM::DEGREE**

        The grid will be generated for the parameter with this ID.

The function generates a grid for the specified parameter of the SVM algorithm. The grid may be passed to the function ``CvSVM::train_auto`` .

.. index:: CvSVM::get_params

.. _CvSVM::get_params:

CvSVM::get_params
-----------------
.. c:function:: CvSVMParams CvSVM::get_params() const

    Returns the current SVM parameters.

This function may be used to get the optimal parameters obtained while automatically training ``CvSVM::train_auto`` .

.. index:: CvSVM::get_support_vector*

.. _CvSVM::get_support_vector*:

CvSVM::get_support_vector*
--------------------------
.. c:function:: int CvSVM::get_support_vector_count() const

.. c:function:: const float* CvSVM::get_support_vector(int i) const

    Retrieves a number of support vectors and the particular vector.

The methods can be used to retrieve a set of support vectors.

