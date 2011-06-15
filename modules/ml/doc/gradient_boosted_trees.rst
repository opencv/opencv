.. _Gradient Boosted Trees:

Gradient Boosted Trees
======================

Gradient Boosted Trees (GBT) is a generalized boosting algorithm, introduced by
Jerome Friedman: http://www.salfordsystems.com/doc/GreedyFuncApproxSS.pdf .
In contrast to AdaBoost.M1 algorithm GBT can deal with both multiclass
classification and regression problems. More than that it can use any
differential loss function, some popular ones are implemented.
Decision trees (:ref:`CvDTree`) usage as base learners allows to process ordered
and categorical variables.


.. _Training the GBT model:

Training the GBT model
----------------------

Gradient Boosted Trees model represents an ensemble of single regression trees,
that are built in a greedy fashion. Training procedure is an iterative proccess
similar to the numerical optimazation via gradient descent method. Summary loss
on the training set depends only from the current model predictions on the
thaining samples,  in other words
:math:`\sum^N_{i=1}L(y_i, F(x_i)) \equiv \mathcal{L}(F(x_1), F(x_2), ... , F(x_N))
\equiv \mathcal{L}(F)`. And the :math:`\mathcal{L}(F)`
gradient can be computed as follows:

.. math::
    grad(\mathcal{L}(F)) = \left( \dfrac{\partial{L(y_1, F(x_1))}}{\partial{F(x_1)}},
    \dfrac{\partial{L(y_2, F(x_2))}}{\partial{F(x_2)}}, ... ,
    \dfrac{\partial{L(y_N, F(x_N))}}{\partial{F(x_N)}} \right) .
On every training step a single regression tree is built to predict an
antigradient vector components. Step length is computed corresponding to the
loss function and separately for every region determined by the tree leaf, and
can be eliminated by changing leaves' values directly.

The main scheme of the training proccess is shown below.

#.
    Find the best constant model.
#.
    For :math:`i` in :math:`[1,M]`:

    #.
        Compute the antigradient.
    #.
        Grow a regression tree to predict antigradient components.
    #.
        Change values in the tree leaves.
    #.
        Add the tree to the model.


The following loss functions are implemented:

*for regression problems:*

#.
    Squared loss (``CvGBTrees::SQUARED_LOSS``):
    :math:`L(y,f(x))=\dfrac{1}{2}(y-f(x))^2`
#.
    Absolute loss (``CvGBTrees::ABSOLUTE_LOSS``):
    :math:`L(y,f(x))=|y-f(x)|`
#.
    Huber loss (``CvGBTrees::HUBER_LOSS``):
    :math:`L(y,f(x)) = \left\{ \begin{array}{lr}
    \delta\cdot\left(|y-f(x)|-\dfrac{\delta}{2}\right) & : |y-f(x)|>\delta\\
    \dfrac{1}{2}\cdot(y-f(x))^2 & : |y-f(x)|\leq\delta \end{array} \right.`,
    where :math:`\delta` is the :math:`\alpha`-quantile estimation of the
    :math:`|y-f(x)|`. In the current implementation :math:`\alpha=0.2`.

*for classification problems:*

4.
    Deviance or cross-entropy loss (``CvGBTrees::DEVIANCE_LOSS``):
    :math:`K` functions are built, one function for each output class, and
    :math:`L(y,f_1(x),...,f_K(x)) = -\sum^K_{k=0}1(y=k)\ln{p_k(x)}`,
    where :math:`p_k(x)=\dfrac{\exp{f_k(x)}}{\sum^K_{i=1}\exp{f_i(x)}}`
    is the estimation of the probability that :math:`y=k`.

In the end we get the model in the following form:

.. math:: f(x) = f_0 + \nu\cdot\sum^M_{i=1}T_i(x) ,
where :math:`f_0` is the initial guess (the best constant model) and :math:`\nu`
is a regularization parameter from the interval :math:`(0,1]`, futher called
*shrinkage*.


.. _Predicting with GBT model:

Predicting with GBT model
-------------------------

To get the GBT model prediciton it is needed to compute the sum of responses of
all the trees in the ensemble. For regression problems it is the answer, and
for classification problems the result is :math:`\arg\max_{i=1..K}(f_i(x))`.


.. highlight:: cpp


.. index:: CvGBTreesParams
.. _CvGBTreesParams:

CvGBTreesParams
---------------
.. c:type:: CvGBTreesParams

GBT training parameters ::

    struct CvGBTreesParams : public CvDTreeParams
    {
        int weak_count;
        int loss_function_type;
        float subsample_portion;
        float shrinkage;

        CvGBTreesParams();
        CvGBTreesParams( int loss_function_type, int weak_count, float shrinkage,
            float subsample_portion, int max_depth, bool use_surrogates );
    };

The structure contains parameters for each sigle decision tree in the ensemble,
as well as the whole model characteristics. The structure is derived from
:ref:`CvDTreeParams` but not all of the decision tree parameters are supported:
cross-validation, pruning and class priorities are not used. The whole
parameters list is shown below:

``weak_count``

    The count of boosting algorithm iterations. ``weak_count*K`` -- is the total
    count of trees in the GBT model, where ``K`` is the output classes count
    (equal to one in the case of regression).
    
``loss_function_type``

    The type of the loss function used for training
    (see :ref:`Training the GBT model`). It must be one of the
    following: ``CvGBTrees::SQUARED_LOSS``, ``CvGBTrees::ABSOLUTE_LOSS``,
    ``CvGBTrees::HUBER_LOSS``, ``CvGBTrees::DEVIANCE_LOSS``. The first three
    ones are used for the case of regression problems, and the last one for
    classification.
    
``shrinkage``

    Regularization parameter (see :ref:`Training the GBT model`).
    
``subsample_portion``

    The portion of the whole training set used on each algorithm iteration.
    Subset is generated randomly
    (For more information see
    http://www.salfordsystems.com/doc/StochasticBoostingSS.pdf).

``max_depth``

    The maximal depth of each decision tree in the ensemble (see :ref:`CvDTree`).

``use_surrogates``

    If ``true`` surrogate splits are built (see :ref:`CvDTree`).
    
By default the following constructor is used:

.. code-block:: cpp

    CvGBTreesParams(CvGBTrees::SQUARED_LOSS, 200, 0.8f, 0.01f, 3, false)
        : CvDTreeParams( 3, 10, 0, false, 10, 0, false, false, 0 )



.. index:: CvGBTrees
.. _CvGBTrees:

CvGBTrees
---------
.. c:type:: CvGBTrees

GBT model ::

	class CvGBTrees : public CvStatModel
	{
	public:

		enum {SQUARED_LOSS=0, ABSOLUTE_LOSS, HUBER_LOSS=3, DEVIANCE_LOSS};

		CvGBTrees();
		CvGBTrees( const cv::Mat& trainData, int tflag,
                        const Mat& responses, const Mat& varIdx=Mat(),
                        const Mat& sampleIdx=Mat(), const cv::Mat& varType=Mat(),
                        const Mat& missingDataMask=Mat(),
                        CvGBTreesParams params=CvGBTreesParams() );

		virtual ~CvGBTrees();
		virtual bool train( const Mat& trainData, int tflag,
                        const Mat& responses, const Mat& varIdx=Mat(),
                        const Mat& sampleIdx=Mat(), const Mat& varType=Mat(),
                        const Mat& missingDataMask=Mat(),
                        CvGBTreesParams params=CvGBTreesParams(),
                        bool update=false );
		
		virtual bool train( CvMLData* data,
                        CvGBTreesParams params=CvGBTreesParams(),
                        bool update=false );

		virtual float predict( const Mat& sample, const Mat& missing=Mat(),
                        const Range& slice = Range::all(),
                        int k=-1 ) const;

		virtual void clear();

		virtual float calc_error( CvMLData* _data, int type,
                        std::vector<float> *resp = 0 );

		virtual void write( CvFileStorage* fs, const char* name ) const;

		virtual void read( CvFileStorage* fs, CvFileNode* node );

	protected:
		
		CvDTreeTrainData* data;
		CvGBTreesParams params;
		CvSeq** weak;
		Mat& orig_response;
		Mat& sum_response;
		Mat& sum_response_tmp;
		Mat& weak_eval;
		Mat& sample_idx;
		Mat& subsample_train;
		Mat& subsample_test;
		Mat& missing;
		Mat& class_labels;
		RNG* rng;
		int class_count;
		float delta;
		float base_value;
		
		...

	};


	
.. index:: CvGBTrees::train

.. _CvGBTrees::train:

CvGBTrees::train
----------------
.. c:function:: bool train(const Mat & trainData, int tflag, const Mat & responses, const Mat & varIdx=Mat(), const Mat & sampleIdx=Mat(), const Mat & varType=Mat(), const Mat & missingDataMask=Mat(), CvGBTreesParams params=CvGBTreesParams(), bool update=false)

.. c:function:: bool train(CvMLData* data, CvGBTreesParams params=CvGBTreesParams(), bool update=false)
    
	Trains a Gradient boosted tree model.
	
The first train method follows the common template (see :ref:`CvStatModel::train`).
Both ``tflag`` values (``CV_ROW_SAMPLE``, ``CV_COL_SAMPLE``) are supported.
``trainData`` must be of ``CV_32F`` type. ``responses`` must be a matrix of type
``CV_32S`` or ``CV_32F``, in both cases it is converted into the ``CV_32F``
matrix inside the training procedure. ``varIdx`` and ``sampleIdx`` must be a
list of indices (``CV_32S``), or a mask (``CV_8U`` or ``CV_8S``). ``update`` is
a dummy parameter.

The second form of :ref:`CvGBTrees::train` function uses :ref:`CvMLData` as a
data set container. ``update`` is still a dummy parameter. 

All parameters specific to the GBT model are passed into the training function
as a :ref:`CvGBTreesParams` structure.


.. index:: CvGBTrees::predict

.. _CvGBTrees::predict:

CvGBTrees::predict
------------------
.. c:function:: float predict(const Mat & sample, const Mat & missing=Mat(), const Range & slice = Range::all(), int k=-1) const

    Predicts a response for an input sample.
 
The method predicts the response, corresponding to the given sample
(see :ref:`Predicting with GBT model`).
The result is either the class label or the estimated function value.
:c:func:`predict` method allows to use the parallel version of the GBT model
prediction if the OpenCV is built with the TBB library. In this case predicitons
of single trees are computed in a parallel fashion.

``sample``

    An input feature vector, that has the same format as every training set
    element. Hence, if not all the variables were actualy used while training,
    ``sample`` have to contain fictive values on the appropriate places.
    
``missing``

    The missing values mask. The one dimentional matrix of the same size as
    ``sample`` having a ``CV_8U`` type. ``1`` corresponds to the missing value
    in the same position in the ``sample`` vector. If there are no missing values
    in the feature vector empty matrix can be passed instead of the missing mask.
    
``weak_responses``

    In addition to the prediciton of the whole model all the trees' predcitions
    can be obtained by passing a ``weak_responses`` matrix with :math:`K` rows,
    where :math:`K` is the output classes count (1 for the case of regression)
    and having as many columns as the ``slice`` length.
    
``slice``
    
    Defines the part of the ensemble used for prediction.
    All trees are used when ``slice = Range::all()``. This parameter is useful to
    get predictions of the GBT models with different ensemble sizes learning
    only the one model actually.
    
``k``
    
    In the case of the classification problem not the one, but :math:`K` tree
    ensembles are built (see :ref:`Training the GBT model`). By passing this
    parameter the ouput can be changed to sum of the trees' predictions in the
    ``k``'th ensemble only. To get the total GBT model prediction ``k`` value
    must be -1. For regression problems ``k`` have to be equal to -1 also.
    

    
.. index:: CvGBTrees::clear

.. _CvGBTrees::clear:

CvGBTrees::clear
----------------
.. c:function:: void clear()

    Clears the model.
    
Deletes the data set information, all the weak models and sets all internal
variables to the initial state. Is called in :ref:`CvGBTrees::train` and in the
destructor.


.. index:: CvGBTrees::calc_error

.. _CvGBTrees::calc_error:

CvGBTrees::calc_error
---------------------
.. c:function:: float calc_error( CvMLData* _data, int type, std::vector<float> *resp = 0 )

    Calculates training or testing error.
    
If the :ref:`CvMLData` data is used to store the data set :c:func:`calc_error` can be
used to get the training or testing error easily and (optionally) all predictions
on the training/testing set. If TBB library is used, the error is computed in a
parallel way: predictions for different samples are computed at the same time.
In the case of regression problem mean squared error is returned. For
classifications the result is the misclassification error in percent.

``_data``

    Data set.
    
``type``
    
    Defines what error should be computed: train (``CV_TRAIN_ERROR``) or test
    (``CV_TEST_ERROR``).

``resp``
    
    If not ``0`` a vector of predictions on the corresponding data set is
    returned.

