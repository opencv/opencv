.. _Random Trees:

Random Trees
============

Random trees have been introduced by Leo Breiman and Adele Cutler:
http://www.stat.berkeley.edu/users/breiman/RandomForests/
. The algorithm can deal with both classification and regression problems. Random trees is a collection (ensemble) of tree predictors that is called
*forest*
further in this section (the term has been also introduced by L. Breiman). The classification works as follows: the random trees classifier takes the input feature vector, classifies it with every tree in the forest, and outputs the class label that recieved the majority of "votes". In case of regression, the classifier response is the average of the responses over all the trees in the forest.

All the trees are trained with the same parameters but on different training sets that are generated from the original training set using the bootstrap procedure: for each training set, you randomly select the same number of vectors as in the original set ( ``=N`` ). The vectors are chosen with replacement. That is, some vectors will occur more than once and some will be absent. At each node of each trained tree,  not all the variables are used to find the best split, rather than a random subset of them. With each node a new subset is generated. However, its size is fixed for all the nodes and all the trees. It is a training parameter set to
:math:`\sqrt{number\_of\_variables}` by default. None of the built trees are pruned.

In random trees there is no need for any accuracy estimation procedures, such as cross-validation or bootstrap, or a separate test set to get an estimate of the training error. The error is estimated internally during the training. When the training set for the current tree is drawn by sampling with replacement, some vectors are left out (so-called
*oob (out-of-bag) data*
). The size of oob data is about ``N/3`` . The classification error is estimated by using this oob-data as follows:

#.
    Get a prediction for each vector, which is oob relative to the i-th tree, using the very i-th tree.

#.
    After all the trees have been trained, for each vector that has ever been oob, find the class-"winner" for it (the class that has got the majority of votes in the trees where the vector was oob) and compare it to the ground-truth response.

#.
    Compute the classification error estimate as ratio of the number of misclassified oob vectors to all the vectors in the original data. In case of regression, the oob-error is computed as the squared error for oob vectors difference divided by the total number of vectors.

**References:**

*
    Machine Learning, Wald I, July 2002.

    http://stat-www.berkeley.edu/users/breiman/wald2002-1.pdf

*
    Looking Inside the Black Box, Wald II, July 2002.

    http://stat-www.berkeley.edu/users/breiman/wald2002-2.pdf

*
    Software for the Masses, Wald III, July 2002.

    http://stat-www.berkeley.edu/users/breiman/wald2002-3.pdf

*
    And other articles from the web site
    http://www.stat.berkeley.edu/users/breiman/RandomForests/cc_home.htm
    .

.. index:: CvRTParams

.. _CvRTParams:

CvRTParams
----------
.. c:type:: CvRTParams

Training parameters of random trees ::

    struct CvRTParams : public CvDTreeParams
    {
        bool calc_var_importance;
        int nactive_vars;
        CvTermCriteria term_crit;

        CvRTParams() : CvDTreeParams( 5, 10, 0, false, 10, 0, false, false, 0 ),
            calc_var_importance(false), nactive_vars(0)
        {
            term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 50, 0.1 );
        }

        CvRTParams( int _max_depth, int _min_sample_count,
                    float _regression_accuracy, bool _use_surrogates,
                    int _max_categories, const float* _priors,
                    bool _calc_var_importance,
                    int _nactive_vars, int max_tree_count,
                    float forest_accuracy, int termcrit_type );
    };


The set of training parameters for the forest is a superset of the training parameters for a single tree. However, random trees do not need all the functionality/features of decision trees. Most noticeably, the trees are not pruned, so the cross-validation parameters are not used.

.. index:: CvRTrees

.. _CvRTrees:

CvRTrees
--------
.. c:type:: CvRTrees

Random trees ::

    class CvRTrees : public CvStatModel
    {
    public:
        CvRTrees();
        virtual ~CvRTrees();
        virtual bool train( const Mat& _train_data, int _tflag,
                            const Mat& _responses, const Mat& _var_idx=Mat(),
                            const Mat& _sample_idx=Mat(), const Mat& _var_type=Mat(),
                            const Mat& _missing_mask=Mat(),
                            CvRTParams params=CvRTParams() );
        virtual float predict( const Mat& sample, const Mat& missing = 0 )
                                                                    const;
        virtual void clear();

        virtual const Mat& get_var_importance();
        virtual float get_proximity( const Mat& sample_1, const Mat& sample_2 )
                                                                            const;

        virtual void read( CvFileStorage* fs, CvFileNode* node );
        virtual void write( CvFileStorage* fs, const char* name );

        Mat& get_active_var_mask();
        CvRNG* get_rng();

        int get_tree_count() const;
        CvForestTree* get_tree(int i) const;

    protected:

        bool grow_forest( const CvTermCriteria term_crit );

        // array of the trees of the forest
        CvForestTree** trees;
        CvDTreeTrainData* data;
        int ntrees;
        int nclasses;
        ...
    };


.. index:: CvRTrees::train

.. _CvRTrees::train:

CvRTrees::train
---------------
.. ocv:function:: bool CvRTrees::train(  const Mat& train_data,  int tflag,                      const Mat& responses,  const Mat& comp_idx=Mat(),                      const Mat& sample_idx=Mat(),  const Mat& var_type=Mat(),                      const Mat& missing_mask=Mat(),                      CvRTParams params=CvRTParams() )

    Trains the Random Tree model.

The method ``CvRTrees::train`` is very similar to the first form of ``CvDTree::train`` () and follows the generic method ``CvStatModel::train`` conventions. All the parameters specific to the algorithm training are passed as a
:ref:`CvRTParams` instance. The estimate of the training error ( ``oob-error`` ) is stored in the protected class member ``oob_error`` .

.. index:: CvRTrees::predict

.. _CvRTrees::predict:

CvRTrees::predict
-----------------
.. ocv:function:: double CvRTrees::predict(  const Mat& sample,  const Mat& missing=Mat() ) const

    Predicts the output for an input sample.

The input parameters of the prediction method are the same as in ``CvDTree::predict``  but the return value type is different. This method returns the cumulative result from all the trees in the forest (the class that receives the majority of voices, or the mean of the regression function estimates).

.. index:: CvRTrees::get_var_importance

.. _CvRTrees::get_var_importance:

CvRTrees::get_var_importance
----------------------------
.. ocv:function:: const Mat& CvRTrees::get_var_importance() const

    Retrieves the variable importance array.

The method returns the variable importance vector, computed at the training stage when ``:ref:`CvRTParams`::calc_var_importance`` is set. If the training flag is not set, the ``NULL`` pointer is returned. This differs from the decision trees where variable importance can be computed anytime after the training.

.. index:: CvRTrees::get_proximity

.. _CvRTrees::get_proximity:

CvRTrees::get_proximity
-----------------------
.. ocv:function:: float CvRTrees::get_proximity(  const Mat& sample_1,  const Mat& sample_2 ) const

    Retrieves the proximity measure between two training samples.

The method returns proximity measure between any two samples, which is the ratio of those trees in the ensemble, in which the samples fall into the same leaf node, to the total number of the trees.

For the random trees usage example, please, see letter_recog.cpp sample in OpenCV distribution.
