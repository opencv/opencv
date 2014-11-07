.. _Random Trees:

Random Trees
============

.. highlight:: cpp

Random trees have been introduced by Leo Breiman and Adele Cutler:
http://www.stat.berkeley.edu/users/breiman/RandomForests/
. The algorithm can deal with both classification and regression problems. Random trees is a collection (ensemble) of tree predictors that is called
*forest*
further in this section (the term has been also introduced by L. Breiman). The classification works as follows: the random trees classifier takes the input feature vector, classifies it with every tree in the forest, and outputs the class label that received the majority of "votes". In case of a regression, the classifier response is the average of the responses over all the trees in the forest.

All the trees are trained with the same parameters but on different training sets. These sets are generated from the original training set using the bootstrap procedure: for each training set, you randomly select the same number of vectors as in the original set ( ``=N`` ). The vectors are chosen with replacement. That is, some vectors will occur more than once and some will be absent. At each node of each trained tree,  not all the variables are used to find the best split, but a random subset of them. With each node a new subset is generated. However, its size is fixed for all the nodes and all the trees. It is a training parameter set to
:math:`\sqrt{number\_of\_variables}` by default. None of the built trees are pruned.

In random trees there is no need for any accuracy estimation procedures, such as cross-validation or bootstrap, or a separate test set to get an estimate of the training error. The error is estimated internally during the training. When the training set for the current tree is drawn by sampling with replacement, some vectors are left out (so-called
*oob (out-of-bag) data*
). The size of oob data is about ``N/3`` . The classification error is estimated by using this oob-data as follows:

#.
    Get a prediction for each vector, which is oob relative to the i-th tree, using the very i-th tree.

#.
    After all the trees have been trained, for each vector that has ever been oob, find the class-*winner* for it (the class that has got the majority of votes in the trees where the vector was oob) and compare it to the ground-truth response.

#.
    Compute the classification error estimate as a ratio of the number of misclassified oob vectors to all the vectors in the original data. In case of regression, the oob-error is computed as the squared error for oob vectors difference divided by the total number of vectors.


For the random trees usage example, please, see letter_recog.cpp sample in OpenCV distribution.

**References:**

  * *Machine Learning*, Wald I, July 2002. http://stat-www.berkeley.edu/users/breiman/wald2002-1.pdf

  * *Looking Inside the Black Box*, Wald II, July 2002. http://stat-www.berkeley.edu/users/breiman/wald2002-2.pdf

  * *Software for the Masses*, Wald III, July 2002. http://stat-www.berkeley.edu/users/breiman/wald2002-3.pdf

  * And other articles from the web site http://www.stat.berkeley.edu/users/breiman/RandomForests/cc_home.htm

RTrees::Params
--------------
.. ocv:struct:: RTrees::Params : public DTrees::Params

    Training parameters of random trees.

The set of training parameters for the forest is a superset of the training parameters for a single tree. However, random trees do not need all the functionality/features of decision trees. Most noticeably, the trees are not pruned, so the cross-validation parameters are not used.


RTrees::Params::Params
-----------------------
The constructors

.. ocv:function:: RTrees::Params::Params()

.. ocv:function:: RTrees::Params::Params( int maxDepth, int minSampleCount, double regressionAccuracy, bool useSurrogates, int maxCategories, const Mat& priors, bool calcVarImportance, int nactiveVars, TermCriteria termCrit )

    :param maxDepth: the depth of the tree. A low value will likely underfit and conversely a high value will likely overfit. The optimal value can be obtained using cross validation or other suitable methods.

    :param minSampleCount: minimum samples required at a leaf node for it to be split. A reasonable value is a small percentage of the total data e.g. 1%.

    :param maxCategories: Cluster possible values of a categorical variable into ``K <= maxCategories`` clusters to find a suboptimal split. If a discrete variable, on which the training procedure tries to make a split, takes more than ``max_categories`` values, the precise best subset estimation may take a very long time because the algorithm is exponential. Instead, many decision trees engines (including ML) try to find sub-optimal split in this case by clustering all the samples into ``maxCategories`` clusters that is some categories are merged together. The clustering is applied only in ``n``>2-class classification problems for categorical variables with ``N > max_categories`` possible values. In case of regression and 2-class classification the optimal split can be found efficiently without employing clustering, thus the parameter is not used in these cases.

    :param calcVarImportance: If true then variable importance will be calculated and then it can be retrieved by ``RTrees::getVarImportance``.

    :param nactiveVars: The size of the randomly selected subset of features at each tree node and that are used to find the best split(s). If you set it to 0 then the size will be set to the square root of the total number of features.

    :param termCrit: The termination criteria that specifies when the training algorithm stops - either when the specified number of trees is trained and added to the ensemble or when sufficient accuracy (measured as OOB error) is achieved. Typically the more trees you have the better the accuracy. However, the improvement in accuracy generally diminishes and asymptotes pass a certain number of trees. Also to keep in mind, the number of tree increases the prediction time linearly.

The default constructor sets all parameters to default values which are different from default values of ``DTrees::Params``:

::

    RTrees::Params::Params() : DTrees::Params( 5, 10, 0, false, 10, 0, false, false, Mat() ),
        calcVarImportance(false), nactiveVars(0)
    {
        termCrit = cvTermCriteria( TermCriteria::MAX_ITERS + TermCriteria::EPS, 50, 0.1 );
    }


RTrees
--------
.. ocv:class:: RTrees : public DTrees

    The class implements the random forest predictor as described in the beginning of this section.

RTrees::create
---------------
Creates the empty model

.. ocv:function:: bool RTrees::create(const RTrees::Params& params=Params())

Use ``StatModel::train`` to train the model, ``StatModel::train<RTrees>(traindata, params)`` to create and train the model, ``StatModel::load<RTrees>(filename)`` to load the pre-trained model.

RTrees::getVarImportance
----------------------------
Returns the variable importance array.

.. ocv:function:: Mat RTrees::getVarImportance() const

The method returns the variable importance vector, computed at the training stage when ``RTParams::calcVarImportance`` is set to true. If this flag was set to false, the empty matrix is returned.
