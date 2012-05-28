
.. _ML_Expectation Maximization:


Expectation Maximization
========================

.. highlight:: cpp

The Expectation Maximization(EM) algorithm estimates the parameters of the multivariate probability density function in the form of a Gaussian mixture distribution with a specified number of mixtures.

Consider the set of the N feature vectors
{ :math:`x_1, x_2,...,x_{N}` } from a d-dimensional Euclidean space drawn from a Gaussian mixture:

.. math::

    p(x;a_k,S_k, \pi _k) =  \sum _{k=1}^{m} \pi _kp_k(x),  \quad \pi _k  \geq 0,  \quad \sum _{k=1}^{m} \pi _k=1,

.. math::

    p_k(x)= \varphi (x;a_k,S_k)= \frac{1}{(2\pi)^{d/2}\mid{S_k}\mid^{1/2}} exp \left \{ - \frac{1}{2} (x-a_k)^TS_k^{-1}(x-a_k) \right \} ,

where
:math:`m` is the number of mixtures,
:math:`p_k` is the normal distribution
density with the mean
:math:`a_k` and covariance matrix
:math:`S_k`,
:math:`\pi_k` is the weight of the k-th mixture. Given the number of mixtures
:math:`M` and the samples
:math:`x_i`,
:math:`i=1..N` the algorithm finds the
maximum-likelihood estimates (MLE) of all the mixture parameters,
that is,
:math:`a_k`,
:math:`S_k` and
:math:`\pi_k` :

.. math::

    L(x, \theta )=logp(x, \theta )= \sum _{i=1}^{N}log \left ( \sum _{k=1}^{m} \pi _kp_k(x) \right ) \to \max _{ \theta \in \Theta },

.. math::

    \Theta = \left \{ (a_k,S_k, \pi _k): a_k  \in \mathbbm{R} ^d,S_k=S_k^T>0,S_k  \in \mathbbm{R} ^{d  \times d}, \pi _k \geq 0, \sum _{k=1}^{m} \pi _k=1 \right \} .

The EM algorithm is an iterative procedure. Each iteration includes
two steps. At the first step (Expectation step or E-step), you find a
probability
:math:`p_{i,k}` (denoted
:math:`\alpha_{i,k}` in the formula below) of
sample ``i`` to belong to mixture ``k`` using the currently
available mixture parameter estimates:

.. math::

    \alpha _{ki} =  \frac{\pi_k\varphi(x;a_k,S_k)}{\sum\limits_{j=1}^{m}\pi_j\varphi(x;a_j,S_j)} .

At the second step (Maximization step or M-step), the mixture parameter estimates are refined using the computed probabilities:

.. math::

    \pi _k= \frac{1}{N} \sum _{i=1}^{N} \alpha _{ki},  \quad a_k= \frac{\sum\limits_{i=1}^{N}\alpha_{ki}x_i}{\sum\limits_{i=1}^{N}\alpha_{ki}} ,  \quad S_k= \frac{\sum\limits_{i=1}^{N}\alpha_{ki}(x_i-a_k)(x_i-a_k)^T}{\sum\limits_{i=1}^{N}\alpha_{ki}}

Alternatively, the algorithm may start with the M-step when the initial values for
:math:`p_{i,k}` can be provided. Another alternative when
:math:`p_{i,k}` are unknown is to use a simpler clustering algorithm to pre-cluster the input samples and thus obtain initial
:math:`p_{i,k}` . Often (including machine learning) the
:ocv:func:`kmeans` algorithm is used for that purpose.

One of the main problems of the EM algorithm is a large number
of parameters to estimate. The majority of the parameters reside in
covariance matrices, which are
:math:`d \times d` elements each
where
:math:`d` is the feature space dimensionality. However, in
many practical problems, the covariance matrices are close to diagonal
or even to
:math:`\mu_k*I` , where
:math:`I` is an identity matrix and
:math:`\mu_k` is a mixture-dependent "scale" parameter. So, a robust computation
scheme could start with harder constraints on the covariance
matrices and then use the estimated parameters as an input for a less
constrained optimization problem (often a diagonal covariance matrix is
already a good enough approximation).

**References:**

*
    Bilmes98 J. A. Bilmes. *A Gentle Tutorial of the EM Algorithm and its Application to Parameter Estimation for Gaussian Mixture and Hidden Markov Models*. Technical Report TR-97-021, International Computer Science Institute and Computer Science Division, University of California at Berkeley, April 1998.

EM
--
.. ocv:class:: EM : public Algorithm

The class implements the EM algorithm as described in the beginning of this section. It is inherited from :ocv:class:`Algorithm`.


EM::EM
------
The constructor of the class

.. ocv:function:: EM::EM(int nclusters=EM::DEFAULT_NCLUSTERS, int covMatType=EM::COV_MAT_DIAGONAL, const TermCriteria& termCrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON) )

.. ocv:pyfunction:: cv2.EM([nclusters[, covMatType[, termCrit]]]) -> <EM object>

    :param nclusters: The number of mixture components in the Gaussian mixture model. Default value of the parameter is ``EM::DEFAULT_NCLUSTERS=5``. Some of EM implementation could determine the optimal number of mixtures within a specified value range, but that is not the case in ML yet.

    :param covMatType: Constraint on covariance matrices which defines type of matrices. Possible values are:

        * **EM::COV_MAT_SPHERICAL** A scaled identity matrix :math:`\mu_k * I`. There is the only parameter :math:`\mu_k` to be estimated for each matrix. The option may be used in special cases, when the constraint is relevant, or as a first step in the optimization (for example in case when the data is preprocessed with PCA). The results of such preliminary estimation may be passed again to the optimization procedure, this time with ``covMatType=EM::COV_MAT_DIAGONAL``.

        * **EM::COV_MAT_DIAGONAL** A diagonal matrix with positive diagonal elements. The number of free parameters is ``d`` for each matrix. This is most commonly used option yielding good estimation results.

        * **EM::COV_MAT_GENERIC** A symmetric positively defined matrix. The number of free parameters in each matrix is about :math:`d^2/2`. It is not recommended to use this option, unless there is pretty accurate initial estimation of the parameters and/or a huge number of training samples.

    :param termCrit: The termination criteria of the EM algorithm. The EM algorithm can be terminated by the number of iterations ``termCrit.maxCount`` (number of M-steps) or when relative change of likelihood logarithm is less than ``termCrit.epsilon``. Default maximum number of iterations is ``EM::DEFAULT_MAX_ITERS=100``.

EM::train
---------
Estimates the Gaussian mixture parameters from a samples set.

.. ocv:function:: bool EM::train(InputArray samples, OutputArray logLikelihoods=noArray(), OutputArray labels=noArray(), OutputArray probs=noArray())

.. ocv:function:: bool EM::trainE(InputArray samples, InputArray means0, InputArray covs0=noArray(), InputArray weights0=noArray(), OutputArray logLikelihoods=noArray(), OutputArray labels=noArray(), OutputArray probs=noArray())

.. ocv:function:: bool EM::trainM(InputArray samples, InputArray probs0, OutputArray logLikelihoods=noArray(), OutputArray labels=noArray(), OutputArray probs=noArray())

.. ocv:pyfunction:: cv2.EM.train(samples[, logLikelihoods[, labels[, probs]]]) -> retval, logLikelihoods, labels, probs

.. ocv:pyfunction:: cv2.EM.trainE(samples, means0[, covs0[, weights0[, logLikelihoods[, labels[, probs]]]]]) -> retval, logLikelihoods, labels, probs

.. ocv:pyfunction:: cv2.EM.trainM(samples, probs0[, logLikelihoods[, labels[, probs]]]) -> retval, logLikelihoods, labels, probs

    :param samples: Samples from which the Gaussian mixture model will be estimated. It should be a one-channel matrix, each row of which is a sample. If the matrix does not have ``CV_64F`` type it will be converted to the inner matrix of such type for the further computing.

    :param means0: Initial means :math:`a_k` of mixture components. It is a one-channel matrix of :math:`nclusters \times dims` size. If the matrix does not have ``CV_64F`` type it will be converted to the inner matrix of such type for the further computing.

    :param covs0: The vector of initial covariance matrices :math:`S_k` of mixture components. Each of covariance matrices is a one-channel matrix of :math:`dims \times dims` size. If the matrices do not have ``CV_64F`` type they will be converted to the inner matrices of such type for the further computing.

    :param weights0: Initial weights :math:`\pi_k` of mixture components. It should be a one-channel floating-point matrix with :math:`1 \times nclusters` or :math:`nclusters \times 1` size.

    :param probs0: Initial probabilities :math:`p_{i,k}` of sample :math:`i` to belong to mixture component :math:`k`. It is a  one-channel floating-point matrix of :math:`nsamples \times nclusters` size.

    :param logLikelihoods: The optional output matrix that contains a likelihood logarithm value for each sample. It has :math:`nsamples \times 1` size and ``CV_64FC1`` type.

    :param labels: The optional output "class label" for each sample: :math:`\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N` (indices of the most probable mixture component for each sample). It has :math:`nsamples \times 1` size and ``CV_32SC1`` type.

    :param probs: The optional output matrix that contains posterior probabilities of each Gaussian mixture component given the each sample. It has :math:`nsamples \times nclusters` size and ``CV_64FC1`` type.

Three versions of training method differ in the initialization of Gaussian mixture model parameters and start step:

* **train** - Starts with Expectation step. Initial values of the model parameters will be estimated by the k-means algorithm.

* **trainE** - Starts with Expectation step. You need to provide initial means :math:`a_k` of mixture components. Optionally you can pass initial weights :math:`\pi_k` and covariance matrices :math:`S_k` of mixture components.

* **trainM** - Starts with Maximization step. You need to provide initial probabilities :math:`p_{i,k}` to use this option.

The methods return ``true`` if the Gaussian mixture model was trained successfully, otherwise it returns ``false``.

Unlike many of the ML models, EM is an unsupervised learning algorithm and it does not take responses (class labels or function values) as input. Instead, it computes the
*Maximum Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the parameters inside the structure:
:math:`p_{i,k}` in ``probs``,
:math:`a_k` in ``means`` ,
:math:`S_k` in ``covs[k]``,
:math:`\pi_k` in ``weights`` , and optionally computes the output "class label" for each sample:
:math:`\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N` (indices of the most probable mixture component for each sample).

The trained model can be used further for prediction, just like any other classifier. The trained model is similar to the
:ocv:class:`CvNormalBayesClassifier`.

EM::predict
-----------
Returns a likelihood logarithm value and an index of the most probable mixture component for the given sample.

.. ocv:function:: Vec2d predict(InputArray sample, OutputArray probs=noArray()) const

.. ocv:pyfunction:: cv2.EM.predict(sample[, probs]) -> retval, probs

    :param sample: A sample for classification. It should be a one-channel matrix of :math:`1 \times dims` or :math:`dims \times 1` size.

    :param probs: Optional output matrix that contains posterior probabilities of each component given the sample. It has :math:`1 \times nclusters` size and ``CV_64FC1`` type.

The method returns a two-element ``double`` vector. Zero element is a likelihood logarithm value for the sample. First element is an index of the most probable mixture component for the given sample.

CvEM::isTrained
---------------
Returns ``true`` if the Gaussian mixture model was trained.

.. ocv:function:: bool EM::isTrained() const

.. ocv:pyfunction:: cv2.EM.isTrained() -> retval

EM::read, EM::write
-------------------
See :ocv:func:`Algorithm::read` and :ocv:func:`Algorithm::write`.

EM::get, EM::set
----------------
See :ocv:func:`Algorithm::get` and :ocv:func:`Algorithm::set`. The following parameters are available:

* ``"nclusters"``
* ``"covMatType"``
* ``"maxIters"``
* ``"epsilon"``
* ``"weights"`` *(read-only)*
* ``"means"`` *(read-only)*
* ``"covs"`` *(read-only)*

..
