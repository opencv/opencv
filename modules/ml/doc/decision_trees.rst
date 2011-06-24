Decision Trees
==============

The ML classes discussed in this section implement Classification and Regression Tree algorithms described in `[Breiman84] <#paper_Breiman84>`_
.

The class
:ocv:class:`CvDTree` represents a single decision tree that may be used alone or as a base class in tree ensembles (see
:ref:`Boosting` and
:ref:`Random Trees` ).

A decision tree is a binary tree (tree where each non-leaf node has two child nodes). It can be used either for classification or for regression. For classification, each tree leaf is marked with a class label; multiple leaves may have the same label. For regression, a constant is also assigned to each tree leaf, so the approximation function is piecewise constant.

Predicting with Decision Trees
------------------------------

To reach a leaf node and to obtain a response for the input feature
vector, the prediction procedure starts with the root node. From each
non-leaf node the procedure goes to the left (selects the left
child node as the next observed node) or to the right based on the
value of a certain variable whose index is stored in the observed
node. The following variables are possible:

* 
  **Ordered variables.** The variable value is compared with a threshold that is also stored in the node. If the value is less than the threshold, the procedure goes to the left. Otherwise, it goes to the right. For example, if the weight is less than 1 kilogram, the procedure goes to the left, else to the right.
* 
  **Categorical variables.**  A discrete variable value is tested to see whether it belongs to a certain subset of values (also stored in the node) from a limited set of values the variable could take. If it does, the procedure goes to the left. Otherwise, it goes to the right. For example, if the color is green or red, go to the left, else to the right.

So, in each node, a pair of entities (``variable_index`` , ``decision_rule
(threshold/subset)`` ) is used. This pair is called a *split* (split on
the variable ``variable_index`` ). Once a leaf node is reached, the value
assigned to this node is used as the output of the prediction procedure.

Sometimes, certain features of the input vector are missed (for example, in the darkness it is difficult to determine the object color), and the prediction procedure may get stuck in the certain node (in the mentioned example, if the node is split by color). To avoid such situations, decision trees use so-called *surrogate splits*. That is, in addition to the best "primary" split, every tree node may also be split to one or more other variables with nearly the same results.

Training Decision Trees
-----------------------

The tree is built recursively, starting from the root node. All training data (feature vectors and responses) is used to split the root node. In each node the optimum decision rule (the best "primary" split) is found based on some criteria. In machine learning, ``gini`` "purity" criteria are used for classification, and sum of squared errors is used for regression. Then, if necessary, the surrogate splits are found. They resemble the results of the primary split on the training data. All the data is divided using the primary and the surrogate splits (like it is done in the prediction procedure) between the left and the right child node. Then, the procedure recursively splits both left and right nodes. At each node the recursive procedure may stop (that is, stop splitting the node further) in one of the following cases:

* Depth of the constructed tree branch has reached the specified maximum value.

* Number of training samples in the node is less than the specified threshold when it is not statistically representative to split the node further.

* All the samples in the node belong to the same class or, in case of regression, the variation is too small.

* The best found split does not give any noticeable improvement compared to a random choice.

When the tree is built, it may be pruned using a cross-validation procedure, if necessary. That is, some branches of the tree that may lead to the model overfitting are cut off. Normally, this procedure is only applied to standalone decision trees. Usually tree ensembles build trees that are small enough and use their own protection schemes against overfitting.

Variable Importance
-------------------

Besides the prediction that is an obvious use of decision trees, the tree can be also used for various data analyses. One of the key properties of the constructed decision tree algorithms is an ability to compute the importance (relative decisive power) of each variable. For example, in a spam filter that uses a set of words occurred in the message as a feature vector, the variable importance rating can be used to determine the most "spam-indicating" words and thus help keep the dictionary size reasonable.

Importance of each variable is computed over all the splits on this variable in the tree, primary and surrogate ones. Thus, to compute variable importance correctly, the surrogate splits must be enabled in the training parameters, even if there is no missing data.

[Breiman84] Breiman, L., Friedman, J. Olshen, R. and Stone, C. (1984), *Classification and Regression Trees*, Wadsworth.

CvDTreeSplit
------------
.. ocv:class:: CvDTreeSplit


The structure represents a possible decision tree node split. It has public members:

.. ocv:member:: int var_idx

    Index of variable on which the split is created.

.. ocv:member:: int inversed

    If it is not null then inverse split rule is used that is a left branch and a right branch are switched.

.. ocv:member:: float quality

    Quality of the split.

.. ocv:member:: CvDTreeSplit* next

    Pointer to the next split in the node list of splits.

.. ocv:member:: int subset[2]

    Parameters of the split on a categorical variable.

.. ocv:member:: struct {float c; int split_point;} ord

    Parameters of the split on ordered variable.


CvDTreeNode
-----------
.. ocv:class:: CvDTreeNode


The structure represents a node in a decision tree. It has public members:    
 
.. ocv:member:: int Tn

    Tree index in a sequence of pruned trees. Nodes with :math:`Tn \leq CvDTree::pruned\_tree\_idx` are not used at prediction stage (they are pruned).

.. ocv:member:: double value

    Value at the node: a class label in case of classification or estimated function value in case of regression.

.. ocv:member:: CvDTreeNode* parent

    Pointer to the parent node.

.. ocv:mebmer:: CvDTreeNode* left

    Pointer to the left child node.

.. ocv:member:: CvDTreeNode* right

    Pointer to the right child node.

.. ocv:member:: CvDTreeSplit* split

    Pointer to the first (primary) split in the node list of splits.

.. ocv:mebmer:: int sample_count

    Number of samples in the node.

.. ocv:member:: int depth

    Depth of the node.

Other numerous fields of ``CvDTreeNode`` are used internally at the training stage.

CvDTreeTrainData
----------------
.. ocv:class:: CvDTreeTrainData

Decision tree training data and shared data for tree ensembles. ::

CvDTreeParams
-------------
.. c:type:: CvDTreeParams

    Decision tree training parameters.

The structure contains all the decision tree training parameters. You can initialize it by default constructor and then override any parameters directly before training, or the structure may be fully initialized using the advanced variant of the constructor.

CvDTreeParams::CvDTreeParams
----------------------------
.. ocv:function:: CvDTreeParams::CvDTreeParams()  

.. ocv:function:: CvDTreeParams::CvDTreeParams( int max_depth, int min_sample_count, float regression_accuracy, bool use_surrogates, int max_categories, int cv_folds, bool use_1se_rule, bool truncate_pruned_tree, const float* priors )

    :param max_depth: The maximum number of levels in a tree. The depth of a constructed tree may be smaller due to other termination criterias or pruning of the tree.

    :param min_sample_count: If the number of samples in a node is less than this parameter then the node will not be splitted.

    :param regression_accuracy: Termination criteria for regression trees. If all absolute differences between an estimated value in a node and values of train samples in this node are less than this parameter then the node will not be splitted.
 
    :param use_surrogates: If true then surrogate splits will be built. These splits allow to work with missing data and compute variable importance correctly.

    :param max_categories: Cluster possible values of a categorical variable into ``K`` :math:`\leq` ``max_categories`` clusters to find a suboptimal split. The clustering is applied only in n>2-class classification problems for categorical variables with ``N > max_categories`` possible values. See the Learning OpenCV book (page 489) for more detailed explanation.

    :param cv_folds: If ``cv_folds > 1`` then prune a tree with ``K``-fold cross-validation where ``K`` is equal to ``cv_folds``.

    :param use_1se_rule: If true then a pruning will be harsher. This will make a tree more compact but a bit less accurate.

    :param truncate_pruned_tree: If true then pruned branches are removed completely from the tree. Otherwise they are retained and it is possible to get the unpruned tree or prune the tree differently by changing ``CvDTree::pruned_tree_idx`` parameter.

    :param priors: Weights of prediction categories which determine relative weights that you give to misclassification. That is, if the weight of the first category is 1 and the weight of the second category is 10, then each mistake in predicting the second category is equivalent to making 10 mistakes in predicting the first category.

The default constructor initializes all the parameters with the default values tuned for the standalone classification tree:

::

    CvDTreeParams() : max_categories(10), max_depth(INT_MAX), min_sample_count(10),
        cv_folds(10), use_surrogates(true), use_1se_rule(true),
        truncate_pruned_tree(true), regression_accuracy(0.01f), priors(0)
    {}

 
CvDTreeTrainData
----------------
.. ocv:class:: CvDTreeTrainData

    Decision tree training data and shared data for tree ensembles.

The structure is mostly used internally for storing both standalone trees and tree ensembles efficiently. Basically, it contains the following types of information:

#. Training parameters, an instance of :ocv:class:`CvDTreeParams`.

#. Training data preprocessed to find the best splits more efficiently. For tree ensembles, this preprocessed data is reused by all trees. Additionally, the training data characteristics shared by all trees in the ensemble are stored here: variable types, the number of classes, a class label compression map, and so on.

#. Buffers, memory storages for tree nodes, splits, and other elements of the constructed trees.

There are two ways of using this structure. In simple cases (for example, a standalone tree or the ready-to-use "black box" tree ensemble from machine learning, like
:ref:`Random Trees` or
:ref:`Boosting` ), there is no need to care or even to know about the structure. You just construct the needed statistical model, train it, and use it. The ``CvDTreeTrainData`` structure is constructed and used internally. However, for custom tree algorithms or another sophisticated cases, the structure may be constructed and used explicitly. The scheme is the following:

#.
    The structure is initialized using the default constructor, followed by ``set_data``, or it is built using the full form of constructor. The parameter ``_shared`` must be set to ``true``.

#.
    One or more trees are trained using this data (see the special form of the method :ocv:func:`CvDTree::train`).

#.
    The structure is released as soon as all the trees using it are released.

CvDTree
-------
.. ocv:class:: CvDTree

    Decision tree.

The class implements a decision tree predictor as described in the beginning of this section.


CvDTree::train
--------------
.. ocv:function:: bool CvDTree::train( const Mat& train_data,  int tflag, const Mat& responses,  const Mat& var_idx=Mat(), const Mat& sample_idx=Mat(), const Mat& var_type=Mat(), const Mat& missing_mask=Mat(), CvDTreeParams params=CvDTreeParams() )

.. ocv:function:: bool CvDTree::train( const CvMat* trainData, int tflag, const CvMat* responses, const CvMat* varIdx=0, const CvMat* sampleIdx=0, const CvMat* varType=0, const CvMat* missingDataMask=0, CvDTreeParams params=CvDTreeParams() )

.. ocv:function:: bool CvDTree::train( CvMLData* trainData, CvDTreeParams params=CvDTreeParams() )

.. ocv:function:: bool CvDTree::train( CvDTreeTrainData* trainData, const CvMat* subsampleIdx )

    Trains a decision tree.

There are four ``train`` methods in :ocv:class:`CvDTree`:

* The **first two** methods follow the generic ``CvStatModel::train`` conventions. It is the most complete form. Both data layouts (``tflag=CV_ROW_SAMPLE`` and ``tflag=CV_COL_SAMPLE``) are supported, as well as sample and variable subsets, missing measurements, arbitrary combinations of input and output variable types, and so on. The last parameter contains all of the necessary training parameters (see the :ref:`CvDTreeParams` description).

* The **third** method uses :ocv:class:`CvMLData` to pass training data to a decision tree.

* The **last** method ``train`` is mostly used for building tree ensembles. It takes the pre-constructed :ref:`CvDTreeTrainData` instance and an optional subset of the training set. The indices in ``subsampleIdx`` are counted relatively to the ``_sample_idx`` , passed to the ``CvDTreeTrainData`` constructor. For example, if ``_sample_idx=[1, 5, 7, 100]`` , then ``subsampleIdx=[0,3]`` means that the samples ``[1, 100]`` of the original training set are used.



CvDTree::predict
----------------
.. ocv:function:: CvDTreeNode* CvDTree::predict( const Mat& sample, const Mat& missing_data_mask=Mat(), bool raw_mode=false ) const

.. ocv:function:: CvDTreeNode* CvDTree::predict( const CvMat* sample, const CvMat* missingDataMask=0, bool preprocessedInput=false ) const

    Returns the leaf node of a decision tree corresponding to the input vector.

    :param sample: Sample for prediction.

    :param missing_data: Optional input missing measurement mask.

    :param preprocessedInput: This parameter is normally set to ``false``, implying a regular input. If it is ``true``, the method assumes that all the values of the discrete input variables have been already normalized to :math:`0` to :math:`num\_of\_categories_i-1` ranges since the decision tree uses such normalized representation internally. It is useful for faster prediction with tree ensembles. For ordered input variables, the flag is not used.
       
The method traverses the decision tree and returns the reached leaf node as output. The prediction result, either the class label or the estimated function value, may be retrieved as the ``value`` field of the :ref:`CvDTreeNode` structure, for example: ``dtree->predict(sample,mask)->value``.



CvDTree::calc_error
-------------------
.. ocv:function:: float CvDTree::calc_error( CvMLData* trainData, int type, std::vector<float> *resp = 0 )

    Returns error of the decision tree.

    :param data: Data for the decision tree.
    
    :param type: Type of error. Possible values are:

        * **CV_TRAIN_ERROR** Error on train samples.

        * **CV_TEST_ERROR** Erron on test samples.

    :param resp: If it is not null then size of this vector will be set to the number of samples and each element will be set to result of prediction on the corresponding sample.

The method calculates error of the decision tree. In case of classification it is the percentage of incorrectly classified samples and in case of regression it is the mean of squared errors on samples.


CvDTree::getVarImportance
-------------------------
.. ocv:function:: Mat CvDTree::getVarImportance()

.. ocv:function:: const CvMat* CvDTree::get_var_importance()

    Returns the variable importance array.


CvDTree::get_root
-----------------
.. ocv:function:: const CvDTreeNode* CvDTree::get_root() const

    Returns the root of the decision tree.


CvDTree::get_pruned_tree_idx
----------------------------
.. ocv:function:: int CvDTree::get_pruned_tree_idx() const

    Returns the ``CvDTree::pruned_tree_idx`` parameter.

The parameter ``DTree::pruned_tree_idx`` is used to prune a decision tree. See the ``CvDTreeNode::Tn`` parameter.

CvDTree::get_data
-----------------
.. ocv:function:: const CvDTreeTrainData* CvDTree::get_data() const

    Returns used train data of the decision tree.

Example: building a tree for classifying mushrooms.  See the ``mushroom.cpp`` sample that demonstrates how to build and use the
decision tree.

