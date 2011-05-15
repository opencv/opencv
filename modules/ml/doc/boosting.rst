.. _Boosting:

Boosting
========

A common machine learning task is supervised learning. In supervised learning, the goal is to learn the functional relationship
:math:`F: y = F(x)` between the input
:math:`x` and the output
:math:`y` . Predicting the qualitative output is called classification, while predicting the quantitative output is called regression.

Boosting is a powerful learning concept that provides a solution to the supervised classification learning task. It combines the performance of many "weak" classifiers to produce a powerful 'committee'
:ref:`[HTF01] <HTF01>` . A weak classifier is only required to be better than chance, and thus can be very simple and computationally inexpensive. However, many of them smartly combine results to a strong classifier that often outperforms most "monolithic" strong classifiers such as SVMs and Neural Networks.??

Decision trees are the most popular weak classifiers used in boosting schemes. Often the simplest decision trees with only a single split node per tree (called ``stumps`` ) are sufficient.

The boosted model is based on
:math:`N` training examples
:math:`{(x_i,y_i)}1N` with
:math:`x_i \in{R^K}` and
:math:`y_i \in{-1, +1}` .
:math:`x_i` is a
:math:`K` -component vector. Each component encodes a feature relevant for the learning task at hand. The desired two-class output is encoded as -1 and +1.

Different variants of boosting are known as Discrete Adaboost, Real AdaBoost, LogitBoost, and Gentle AdaBoost
:ref:`[FHT98] <FHT98>` . All of them are very similar in their overall structure. Therefore, this chapter focuses only on the standard two-class Discrete AdaBoost algorithm as shown in the box below??. Each sample is initially assigned the same weight (step 2). Then, a weak classifier
:math:`f_{m(x)}` is trained on the weighted training data (step 3a). Its weighted training error and scaling factor
:math:`c_m` is computed (step 3b). The weights are increased for training samples that have been misclassified (step 3c). All weights are then normalized, and the process of finding the next weak classifier continues for another
:math:`M` -1 times. The final classifier
:math:`F(x)` is the sign of the weighted sum over the individual weak classifiers (step 4).

#.
    Set
    :math:`N`     examples
    :math:`{(x_i,y_i)}1N`     with
    :math:`x_i \in{R^K}, y_i \in{-1, +1}`     .

#.
    Assign weights as
    :math:`w_i = 1/N, i = 1,...,N`     .

#.
    Repeat for
    :math:`m`     =
    :math:`1,2,...,M`     :

    ##.
        Fit the classifier
        :math:`f_m(x) \in{-1,1}`         , using weights
        :math:`w_i`         on the training data.

    ##.
        Compute
        :math:`err_m = E_w [1_{(y =\neq f_m(x))}], c_m = log((1 - err_m)/err_m)`         .

    ##.
        Set
        :math:`w_i \Leftarrow w_i exp[c_m 1_{(y_i \neq f_m(x_i))}], i = 1,2,...,N,`         and renormalize so that
        :math:`\Sigma i w_i = 1`         .

    ##.
        Output the classifier sign
        :math:`[\Sigma m = 1M c_m f_m(x)]`         .

Two-class Discrete AdaBoost Algorithm: Training (steps 1 to 3) and Evaluation (step 4)??you need to revise this section. what is this? a title for the image that is missing?

**NOTE:**

Similar to the classical boosting methods, the current implementation supports two-class classifiers only. For M
:math:`>` two classes, there is the
**AdaBoost.MH**
algorithm (described in
:ref:`[FHT98] <FHT98>` ) that reduces the problem to the two-class problem, yet with a much larger training set.

To reduce computation time for boosted models without substantially losing accuracy, the influence trimming technique may be employed. As the training algorithm proceeds and the number of trees in the ensemble is increased, a larger number of the training samples are classified correctly and with increasing confidence, thereby those samples receive smaller weights on the subsequent iterations. Examples with a very low relative weight have a small impact on the weak classifier training. Thus, such examples may be excluded during the weak classifier training without having much effect on the induced classifier. This process is controlled with the ``weight_trim_rate`` parameter. Only examples with the summary fraction ``weight_trim_rate`` of the total weight mass are used in the weak classifier training. Note that the weights for
**all**
training examples are recomputed at each training iteration. Examples deleted at a particular iteration may be used again for learning some of the weak classifiers further
:ref:`[FHT98] <FHT98>` .

.. _HTF01:??what is this meant to be? it doesn't work

[HTF01] Hastie, T., Tibshirani, R., Friedman, J. H. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer Series in Statistics*. 2001.

.. _FHT98:??the same comment

[FHT98] Friedman, J. H., Hastie, T. and Tibshirani, R. Additive Logistic Regression: a Statistical View of Boosting. Technical Report, Dept. of Statistics*, Stanford University, 1998.

.. index:: CvBoostParams

.. _CvBoostParams:

CvBoostParams
-------------
.. c:type:: CvBoostParams

Boosting training parameters ::

    struct CvBoostParams : public CvDTreeParams
    {
        int boost_type;
        int weak_count;
        int split_criteria;
        double weight_trim_rate;

        CvBoostParams();
        CvBoostParams( int boost_type, int weak_count, double weight_trim_rate,
                       int max_depth, bool use_surrogates, const float* priors );
    };


The structure is derived from
:ref:`CvDTreeParams`  but not all of the decision tree parameters are supported. In particular, cross-validation is not supported.

.. index:: CvBoostTree

.. _CvBoostTree:

CvBoostTree
-----------
.. c:type:: CvBoostTree

Weak tree classifier ::

    class CvBoostTree: public CvDTree
    {
    public:
        CvBoostTree();
        virtual ~CvBoostTree();

        virtual bool train( CvDTreeTrainData* _train_data,
                            const CvMat* subsample_idx, CvBoost* ensemble );
        virtual void scale( double s );
        virtual void read( CvFileStorage* fs, CvFileNode* node,
                           CvBoost* ensemble, CvDTreeTrainData* _data );
        virtual void clear();

    protected:
        ...
        CvBoost* ensemble;
    };


The weak classifier, a component of the boosted tree classifier
:ref:`CvBoost` , is a derivative of
:ref:`CvDTree` . Normally, there is no need to use the weak classifiers directly. However, they can be accessed as elements of the sequence ``CvBoost::weak`` , retrieved by ``CvBoost::get_weak_predictors`` .

**Note:**

In case of LogitBoost and Gentle AdaBoost, each weak predictor is a regression tree, rather than a classification tree. Even in case of Discrete AdaBoost and Real AdaBoost, the ``CvBoostTree::predict`` return value ( ``CvDTreeNode::value`` ) is not an output class label. A negative value "votes" for class
#
0, a positive - for class
#
1. The votes are weighted. The weight of each individual tree may be increased or decreased using the method ``CvBoostTree::scale`` .

.. index:: CvBoost

.. _CvBoost:

CvBoost
-------
.. c:type:: CvBoost

Boosted tree classifier ::

    class CvBoost : public CvStatModel
    {
    public:
        // Boosting type
        enum { DISCRETE=0, REAL=1, LOGIT=2, GENTLE=3 };

        // Splitting criteria
        enum { DEFAULT=0, GINI=1, MISCLASS=3, SQERR=4 };

        CvBoost();
        virtual ~CvBoost();

        CvBoost( const CvMat* _train_data, int _tflag,
                 const CvMat* _responses, const CvMat* _var_idx=0,
                 const CvMat* _sample_idx=0, const CvMat* _var_type=0,
                 const CvMat* _missing_mask=0,
                 CvBoostParams params=CvBoostParams() );

        virtual bool train( const CvMat* _train_data, int _tflag,
                 const CvMat* _responses, const CvMat* _var_idx=0,
                 const CvMat* _sample_idx=0, const CvMat* _var_type=0,
                 const CvMat* _missing_mask=0,
                 CvBoostParams params=CvBoostParams(),
                 bool update=false );

        virtual float predict( const CvMat* _sample, const CvMat* _missing=0,
                               CvMat* weak_responses=0, CvSlice slice=CV_WHOLE_SEQ,
                               bool raw_mode=false ) const;

        virtual void prune( CvSlice slice );

        virtual void clear();

        virtual void write( CvFileStorage* storage, const char* name );
        virtual void read( CvFileStorage* storage, CvFileNode* node );

        CvSeq* get_weak_predictors();
        const CvBoostParams& get_params() const;
        ...

    protected:
        virtual bool set_params( const CvBoostParams& _params );
        virtual void update_weights( CvBoostTree* tree );
        virtual void trim_weights();
        virtual void write_params( CvFileStorage* fs );
        virtual void read_params( CvFileStorage* fs, CvFileNode* node );

        CvDTreeTrainData* data;
        CvBoostParams params;
        CvSeq* weak;
        ...
    };


.. index:: CvBoost::train

.. _CvBoost::train:

CvBoost::train
--------------
.. c:function:: bool CvBoost::train(  const CvMat* _train_data,  int _tflag,               const CvMat* _responses,  const CvMat* _var_idx=0,               const CvMat* _sample_idx=0,  const CvMat* _var_type=0,               const CvMat* _missing_mask=0,               CvBoostParams params=CvBoostParams(),               bool update=false )

    Trains a boosted tree classifier.

The train method follows the common template. The last parameter ``update`` specifies whether the classifier needs to be updated (the new weak tree classifiers added to the existing ensemble) or the classifier needs to be rebuilt from scratch. The responses must be categorical, which means that boosted trees cannot be built for regression, and there should be two classes.

.. index:: CvBoost::predict

.. _CvBoost::predict:

CvBoost::predict
----------------
.. c:function:: float CvBoost::predict(  const CvMat* sample,  const CvMat* missing=0,                          CvMat* weak_responses=0,  CvSlice slice=CV_WHOLE_SEQ,                          bool raw_mode=false ) const

    Predicts a response for an input sample.

The method ``CvBoost::predict`` runs the sample through the trees in the ensemble and returns the output class label based on the weighted voting.

.. index:: CvBoost::prune

.. _CvBoost::prune:

CvBoost::prune
--------------
.. c:function:: void CvBoost::prune( CvSlice slice )

    Removes the specified weak classifiers.

The method removes the specified weak classifiers from the sequence. 

**Note:**

Do not confuse this method with the pruning of individual decision trees, which is currently not supported.

.. index:: CvBoost::get_weak_predictors

.. _CvBoost::get_weak_predictors:

CvBoost::get_weak_predictors
----------------------------
.. c:function:: CvSeq* CvBoost::get_weak_predictors()

    Returns the sequence of weak tree classifiers.

The method returns the sequence of weak classifiers. Each element of the sequence is a pointer to the ``CvBoostTree`` class or, probably, to some of its derivatives.

