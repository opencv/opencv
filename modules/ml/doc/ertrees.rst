Extremely randomized trees
==========================

Extremely randomized trees have been introduced by Pierre Geurts, Damien Ernst and Louis Wehenkel in the article "Extremely randomized trees", 2006 [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.65.7485&rep=rep1&type=pdf]. The algorithm of growing Extremely randomized trees is similar to :ref:`Random Trees` (Random Forest), but there are two differences:

#. Extremely randomized trees don't apply the bagging procedure to constract the training samples for each tree. The same input training set is used to train all trees.

#. Extremely randomized trees pick a node split very extremely (both a variable index and variable spliting value are chosen randomly), whereas Random Forest finds the best split (optimal one by variable index and variable spliting value) among random subset of variables.


CvERTrees
--------
.. ocv:class:: CvERTrees

    The class implements the Extremely randomized trees algorithm. ``CvERTrees`` is inherited from :ocv:class:`CvRTrees` and has the same interface, so see description of :ocv:class:`CvRTrees` class to get detailes. To set the training parameters of Extremely randomized trees the same class :ocv:class:`CvRTParams` is used.
