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
:math:`p_{i,k}` . Often (including macnine learning) the
:ref:`kmeans` algorithm is used for that purpose.

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


CvEMParams
----------
.. ocv:class:: CvEMParams

Parameters of the EM algorithm. All parameters are public. You can initialize them by a constructor and then override some of them directly if you want.



CvEMParams::CvEMParams
----------------------
The constructors

.. ocv:function:: CvEMParams::CvEMParams()

.. ocv:function:: CvEMParams::CvEMParams( int nclusters, int cov_mat_type=CvEM::COV_MAT_DIAGONAL, int start_step=CvEM::START_AUTO_STEP, CvTermCriteria term_crit=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON), const CvMat* probs=0, const CvMat* weights=0, const CvMat* means=0, const CvMat** covs=0 ) 

    :param nclusters: The number of mixture components in the gaussian mixture model. Some of EM implementation could determine the optimal number of mixtures within a specified value range, but that is not the case in ML yet.
    
    :param cov_mat_type: Constraint on covariance matrices which defines type of matrices. Possible values are:

        * **CvEM::COV_MAT_SPHERICAL** A scaled identity matrix :math:`\mu_k * I`. There is the only parameter :math:`\mu_k` to be estimated for earch matrix. The option may be used in special cases, when the constraint is relevant, or as a first step in the optimization (for example in case when the data is preprocessed with PCA). The results of such preliminary estimation may be passed again to the optimization procedure, this time with ``cov_mat_type=CvEM::COV_MAT_DIAGONAL``.

        * **CvEM::COV_MAT_DIAGONAL** A diagonal matrix with positive diagonal elements. The number of free parameters is ``d`` for each matrix. This is most commonly used option yielding good estimation results.

        * **CvEM::COV_MAT_GENERIC** A symmetric positively defined matrix. The number of free parameters in each matrix is about :math:`d^2/2`. It is not recommended to use this option, unless there is pretty accurate initial estimation of the parameters and/or a huge number of training samples.

    :param start_step: The start step of the EM algorithm: 

        * **CvEM::START_E_STEP** Start with Expectation step. You need to provide means :math:`a_k` of mixtures to use this option. Optionally you can pass weights :math:`\pi_k` and covariance matrices :math:`S_k` of mixtures.
        * **CvEM::START_M_STEP** Start with Maximization step. You need to provide initial probabilites :math:`p_{i,k}` to use this option.
        * **CvEM::START_AUTO_STEP** Start with Expectation step. You need not provide any parameters because they will be estimated by the k-means algorithm.

    :param term_crit: The termination criteria of the EM algorithm. The EM algorithm can be terminated by the number of iterations ``term_crit.max_iter`` (number of M-steps) or when relative change of likelihood logarithm is less than ``term_crit.epsilon``.

    :param probs: Initial probabilities :math:`p_{i,k}` of sample :math:`i` to belong to mixture :math:`k`. It is a floating-point matrix of :math:`nsamples \times nclusters` size. It is used and must be not NULL only when ``start_step=CvEM::START_M_STEP``.

    :param weights: Initial weights of mixtures :math:`\pi_k`. It is a floating-point vector with :math:`nclusters` elements. It is used (if not NULL) only when ``start_step=CvEM::START_E_STEP``. 

    :param means: Initial means of mixtures :math:`a_k`. It is a floating-point matrix of :math:`nclusters \times dims` size. It is used used and must be not NULL only when ``start_step=CvEM::START_E_STEP``.

    :param covs: Initial covariance matrices of mixtures :math:`S_k`. Each of covariance matrices is a valid square floating-point matrix of :math:`dims \times dims` size. It is used (if not NULL) only when ``start_step=CvEM::START_E_STEP``.

The default constructor represents a rough rule-of-the-thumb:

::

    CvEMParams() : nclusters(10), cov_mat_type(1/*CvEM::COV_MAT_DIAGONAL*/),
        start_step(0/*CvEM::START_AUTO_STEP*/), probs(0), weights(0), means(0), covs(0)
    {
        term_crit=cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON );
    }


With another contstructor it is possible to override a variety of parameters from a single number of mixtures (the only essential problem-dependent parameter) to initial values for the mixture parameters.


CvEM
----
.. ocv:class:: CvEM

    The class implements the EM algorithm as described in the beginning of this section.


CvEM::train
-----------
Estimates the Gaussian mixture parameters from a sample set.

.. ocv:function:: void CvEM::train(  const Mat& samples,  const Mat&  sample_idx=Mat(),                    CvEMParams params=CvEMParams(),  Mat* labels=0 )

.. ocv:function:: bool CvEM::train( const CvMat* samples, const CvMat* sampleIdx=0, CvEMParams params=CvEMParams(), CvMat* labels=0 )

.. ocv:pyfunction:: cv2.CvEM.train(samples[, sampleIdx[, params]]) -> retval, labels

    :param samples: Samples from which the Gaussian mixture model will be estimated.

    :param sample_idx: Mask of samples to use. All samples are used by default.

    :param params: Parameters of the EM algorithm.

    :param labels: The optional output "class label" for each sample: :math:`\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N` (indices of the most probable mixture component for each sample).

    Estimates the Gaussian mixture parameters from a sample set.

Unlike many of the ML models, EM is an unsupervised learning algorithm and it does not take responses (class labels or function values) as input. Instead, it computes the
*Maximum Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the parameters inside the structure:
:math:`p_{i,k}` in ``probs``,
:math:`a_k` in ``means`` ,
:math:`S_k` in ``covs[k]``,
:math:`\pi_k` in ``weights`` , and optionally computes the output "class label" for each sample:
:math:`\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N` (indices of the most probable mixture for each sample).

The trained model can be used further for prediction, just like any other classifier. The trained model is similar to the
:ref:`Bayes classifier`.

For an example of clustering random samples of the multi-Gaussian distribution using EM, see ``em.cpp`` sample in the OpenCV distribution.


CvEM::predict
-------------
Returns a mixture component index of a sample.

.. ocv:function:: float CvEM::predict( const Mat& sample, Mat* probs=0 ) const

.. ocv:function:: float CvEM::predict( const CvMat* sample, CvMat* probs ) const

.. ocv:pyfunction:: cv2.CvEM.predict(sample) -> retval, probs

    :param sample: A sample for classification.

    :param probs: If it is not null then the method will write posterior probabilities of each component given the sample data to this parameter.


CvEM::getNClusters
------------------
Returns the number of mixture components :math:`M` in the gaussian mixture model.

.. ocv:function:: int CvEM::getNClusters() const

.. ocv:function:: int CvEM::get_nclusters() const


CvEM::getNClusters
------------------
Returns mixture means :math:`a_k`.

.. ocv:function:: Mat CvEM::getMeans() const

.. ocv:function:: const CvMat* CvEM::get_means() const


CvEM::getCovs
-------------
Returns mixture covariance matrices :math:`S_k`.

.. ocv:function:: void CvEM::getCovs(std::vector<cv::Mat>& covs) const

.. ocv:function:: const CvMat** CvEM::get_covs() const


CvEM::getWeights
----------------
Returns mixture weights :math:`\pi_k`.

.. ocv:function:: Mat CvEM::getWeights() const

.. ocv:function:: const CvMat* CvEM::get_weights() const


CvEM::getProbs
--------------
Returns vectors of probabilities for each training sample.

.. ocv:function:: Mat CvEM::getProbs() const

.. ocv:function:: const CvMat* CvEM::get_probs() const

Returns probabilites :math:`p_{i,k}` of sample :math:`i` (that have been passed to the constructor or to :ocv:func:`CvEM::train`) to belong to a mixture component :math:`k`.


CvEM::getLikelihood
-------------------
Returns logarithm of likelihood.

.. ocv:function:: double CvEM::getLikelihood() const

.. ocv:function:: double CvEM::get_log_likelihood() const


CvEM::getLikelihoodDelta
------------------------
Returns difference between logarithm of likelihood on the last iteration and logarithm of likelihood on the previous iteration.

.. ocv:function:: double CvEM::getLikelihoodDelta() const

.. ocv:function:: double CvEM::get_log_likelihood_delta() const 


CvEM::write_params
------------------
Writes used parameters of the EM algorithm to a file storage.

.. ocv:function:: void CvEM::write_params( CvFileStorage* fs ) const

    :param fs: A file storage where parameters will be written.


CvEM::read_params
-----------------
Reads parameters of the EM algorithm.

.. ocv:function:: void CvEM::read_params( CvFileStorage* fs, CvFileNode* node )

    :param fs: A file storage with parameters of the EM algorithm.

    :param node: The parent map. If it is NULL, the function searches a node with parameters in all the top-level nodes (streams), starting with the first one.

The function reads EM parameters from the specified file storage node. For example of clustering random samples of multi-Gaussian distribution using EM see em.cpp sample in OpenCV distribution.


