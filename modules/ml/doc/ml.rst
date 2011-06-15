********************
ml. Machine Learning
********************

The Machine Learning Library (MLL) is a set of classes and functions for statistical classification, regression, and clustering of data.

Most of the classification and regression algorithms are implemented as C++ classes. As the algorithms have different sets of features (like an ability to handle missing measurements or categorical input variables), there is a little common ground between the classes. This common ground is defined by the class `CvStatModel` that all the other ML classes are derived from.

.. toctree::
    :maxdepth: 2

    statistical_models
    normal_bayes_classifier
    k_nearest_neighbors
    support_vector_machines
    decision_trees
    boosting
    gradient_boosted_trees
    random_trees
    expectation_maximization
    neural_networks
