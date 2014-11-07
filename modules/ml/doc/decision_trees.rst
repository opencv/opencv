Decision Trees
==============

The ML classes discussed in this section implement Classification and Regression Tree algorithms described in [Breiman84]_.

The class ``cv::ml::DTrees`` represents a single decision tree or a collection of decision trees. It's also a base class for ``RTrees`` and ``Boost``.

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


DTrees::Split
-------------
.. ocv:class:: DTrees::Split

  The class represents split in a decision tree. It has public members:

  .. ocv:member:: int varIdx

     Index of variable on which the split is created.

  .. ocv:member:: bool inversed

     If true, then the inverse split rule is used (i.e. left and right branches are exchanged in the rule expressions below).

  .. ocv:member:: float quality

     The split quality, a positive number. It is used to choose the best split.

  .. ocv:member:: int next

     Index of the next split in the list of splits for the node

  .. ocv:member:: float c

     The threshold value in case of split on an ordered variable. The rule is: ::

       if var_value < c
         then next_node<-left
         else next_node<-right

  .. ocv:member:: int subsetOfs

     Offset of the bitset used by the split on a categorical variable. The rule is: ::

        if bitset[var_value] == 1
          then next_node <- left
          else next_node <- right

DTrees::Node
------------
.. ocv:class:: DTrees::Node

  The class represents a decision tree node. It has public members:

  .. ocv:member:: double value

    Value at the node: a class label in case of classification or estimated function value in case of regression.

  .. ocv:member:: int classIdx

    Class index normalized to 0..class_count-1 range and assigned to the node. It is used internally in classification trees and tree ensembles.

  .. ocv:member:: int parent

    Index of the parent node

  .. ocv:member:: int left

    Index of the left child node

  .. ocv:member:: int right

    Index of right child node.

  .. ocv:member:: int defaultDir

    Default direction where to go (-1: left or +1: right). It helps in the case of missing values.

  .. ocv:member:: int split

    Index of the first split

DTrees::Params
---------------
.. ocv:class:: DTrees::Params

The structure contains all the decision tree training parameters. You can initialize it by default constructor and then override any parameters directly before training, or the structure may be fully initialized using the advanced variant of the constructor.

DTrees::Params::Params
----------------------------
The constructors

.. ocv:function:: DTrees::Params::Params()

.. ocv:function:: DTrees::Params::Params( int maxDepth, int minSampleCount, double regressionAccuracy, bool useSurrogates, int maxCategories, int CVFolds, bool use1SERule, bool truncatePrunedTree, const Mat& priors )

    :param maxDepth: The maximum possible depth of the tree. That is the training algorithms attempts to split a node while its depth is less than ``maxDepth``. The root node has zero depth. The actual depth may be smaller if the other termination criteria are met (see the outline of the training procedure in the beginning of the section), and/or if the tree is pruned.

    :param minSampleCount: If the number of samples in a node is less than this parameter then the node will not be split.

    :param regressionAccuracy: Termination criteria for regression trees. If all absolute differences between an estimated value in a node and values of train samples in this node are less than this parameter then the node will not be split further.

    :param useSurrogates: If true then surrogate splits will be built. These splits allow to work with missing data and compute variable importance correctly. .. note:: currently it's not implemented.

    :param maxCategories: Cluster possible values of a categorical variable into ``K<=maxCategories`` clusters to find a suboptimal split. If a discrete variable, on which the training procedure tries to make a split, takes more than ``maxCategories`` values, the precise best subset estimation may take a very long time because the algorithm is exponential. Instead, many decision trees engines (including our implementation) try to find sub-optimal split in this case by clustering all the samples into ``maxCategories`` clusters that is some categories are merged together. The clustering is applied only in ``n > 2``-class classification problems for categorical variables with ``N > max_categories`` possible values. In case of regression and 2-class classification the optimal split can be found efficiently without employing clustering, thus the parameter is not used in these cases.

    :param CVFolds: If ``CVFolds > 1`` then algorithms prunes the built decision tree using ``K``-fold cross-validation procedure where ``K`` is equal to ``CVFolds``.

    :param use1SERule: If true then a pruning will be harsher. This will make a tree more compact and more resistant to the training data noise but a bit less accurate.

    :param truncatePrunedTree: If true then pruned branches are physically removed from the tree. Otherwise they are retained and it is possible to get results from the original unpruned (or pruned less aggressively) tree.

    :param priors: The array of a priori class probabilities, sorted by the class label value. The parameter can be used to tune the decision tree preferences toward a certain class. For example, if you want to detect some rare anomaly occurrence, the training base will likely contain much more normal cases than anomalies, so a very good classification performance will be achieved just by considering every case as normal. To avoid this, the priors can be specified, where the anomaly probability is artificially increased (up to 0.5 or even greater), so the weight of the misclassified anomalies becomes much bigger, and the tree is adjusted properly. You can also think about this parameter as weights of prediction categories which determine relative weights that you give to misclassification. That is, if the weight of the first category is 1 and the weight of the second category is 10, then each mistake in predicting the second category is equivalent to making 10 mistakes in predicting the first category.

The default constructor initializes all the parameters with the default values tuned for the standalone classification tree:

::

    DTrees::Params::Params()
    {
        maxDepth = INT_MAX;
        minSampleCount = 10;
        regressionAccuracy = 0.01f;
        useSurrogates = false;
        maxCategories = 10;
        CVFolds = 10;
        use1SERule = true;
        truncatePrunedTree = true;
        priors = Mat();
    }


DTrees
------

.. ocv:class:: DTrees : public StatModel

The class represents a single decision tree or a collection of decision trees. The current public interface of the class allows user to train only a single decision tree, however the class is capable of storing multiple decision trees and using them for prediction (by summing responses or using a voting schemes), and the derived from DTrees classes (such as ``RTrees`` and ``Boost``) use this capability to implement decision tree ensembles.

DTrees::create
----------------
Creates the empty model

.. ocv:function:: Ptr<DTrees> DTrees::create(const Params& params=Params())

The static method creates empty decision tree with the specified parameters. It should be then trained using ``train`` method (see ``StatModel::train``). Alternatively, you can load the model from file using ``StatModel::load<DTrees>(filename)``.

DTrees::getDParams
------------------
Returns the training parameters

.. ocv:function:: Params DTrees::getDParams() const

The method returns the training parameters.

DTrees::setDParams
-------------------
Sets the training parameters

.. ocv:function:: void DTrees::setDParams( const Params& p )

    :param p: Training parameters of type DTrees::Params.

The method sets the training parameters.


DTrees::getRoots
-------------------
Returns indices of root nodes

.. ocv:function:: std::vector<int>& DTrees::getRoots() const

DTrees::getNodes
-------------------
Returns all the nodes

.. ocv:function:: std::vector<Node>& DTrees::getNodes() const

all the node indices, mentioned above (left, right, parent, root indices) are indices in the returned vector

DTrees::getSplits
-------------------
Returns all the splits

.. ocv:function:: std::vector<Split>& DTrees::getSplits() const

all the split indices, mentioned above (split, next etc.) are indices in the returned vector

DTrees::getSubsets
-------------------
Returns all the bitsets for categorical splits

.. ocv:function:: std::vector<int>& DTrees::getSubsets() const

``Split::subsetOfs`` is an offset in the returned vector

.. [Breiman84] Breiman, L., Friedman, J. Olshen, R. and Stone, C. (1984), *Classification and Regression Trees*, Wadsworth.
