Expectation Maximization
========================

This section describes obsolete ``C`` interface of EM algorithm. Details of the algorithm and its ``C++`` interface can be found in the other section :ref:`ML_Expectation Maximization`.

.. highlight:: cpp


CvEMParams
----------
.. ocv:struct:: CvEMParams

Parameters of the EM algorithm. All parameters are public. You can initialize them by a constructor and then override some of them directly if you want.

CvEMParams::CvEMParams
----------------------
The constructors

.. ocv:function:: CvEMParams::CvEMParams()

.. ocv:function:: CvEMParams::CvEMParams( int nclusters, int cov_mat_type=EM::COV_MAT_DIAGONAL, int start_step=EM::START_AUTO_STEP, CvTermCriteria term_crit=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON), const CvMat* probs=0, const CvMat* weights=0, const CvMat* means=0, const CvMat** covs=0 )

    :param nclusters: The number of mixture components in the Gaussian mixture model. Some of EM implementation could determine the optimal number of mixtures within a specified value range, but that is not the case in ML yet.

    :param cov_mat_type: Constraint on covariance matrices which defines type of matrices. Possible values are:

        * **CvEM::COV_MAT_SPHERICAL** A scaled identity matrix :math:`\mu_k * I`. There is the only parameter :math:`\mu_k` to be estimated for each matrix. The option may be used in special cases, when the constraint is relevant, or as a first step in the optimization (for example in case when the data is preprocessed with PCA). The results of such preliminary estimation may be passed again to the optimization procedure, this time with ``cov_mat_type=CvEM::COV_MAT_DIAGONAL``.

        * **CvEM::COV_MAT_DIAGONAL** A diagonal matrix with positive diagonal elements. The number of free parameters is ``d`` for each matrix. This is most commonly used option yielding good estimation results.

        * **CvEM::COV_MAT_GENERIC** A symmetric positively defined matrix. The number of free parameters in each matrix is about :math:`d^2/2`. It is not recommended to use this option, unless there is pretty accurate initial estimation of the parameters and/or a huge number of training samples.

    :param start_step: The start step of the EM algorithm:

        * **CvEM::START_E_STEP** Start with Expectation step. You need to provide means :math:`a_k` of mixture components to use this option. Optionally you can pass weights :math:`\pi_k` and covariance matrices :math:`S_k` of mixture components.
        * **CvEM::START_M_STEP** Start with Maximization step. You need to provide initial probabilities :math:`p_{i,k}` to use this option.
        * **CvEM::START_AUTO_STEP** Start with Expectation step. You need not provide any parameters because they will be estimated by the kmeans algorithm.

    :param term_crit: The termination criteria of the EM algorithm. The EM algorithm can be terminated by the number of iterations ``term_crit.max_iter`` (number of M-steps) or when relative change of likelihood logarithm is less than ``term_crit.epsilon``.

    :param probs: Initial probabilities :math:`p_{i,k}` of sample :math:`i` to belong to mixture component :math:`k`. It is a floating-point matrix of :math:`nsamples \times nclusters` size. It is used and must be not NULL only when ``start_step=CvEM::START_M_STEP``.

    :param weights: Initial weights :math:`\pi_k` of mixture components. It is a floating-point vector with :math:`nclusters` elements. It is used (if not NULL) only when ``start_step=CvEM::START_E_STEP``.

    :param means: Initial means :math:`a_k` of mixture components. It is a floating-point matrix of :math:`nclusters \times dims` size. It is used used and must be not NULL only when ``start_step=CvEM::START_E_STEP``.

    :param covs: Initial covariance matrices :math:`S_k` of mixture components. Each of covariance matrices is a valid square floating-point matrix of :math:`dims \times dims` size. It is used (if not NULL) only when ``start_step=CvEM::START_E_STEP``.

The default constructor represents a rough rule-of-the-thumb:

::

    CvEMParams() : nclusters(10), cov_mat_type(1/*CvEM::COV_MAT_DIAGONAL*/),
        start_step(0/*CvEM::START_AUTO_STEP*/), probs(0), weights(0), means(0), covs(0)
    {
        term_crit=cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, FLT_EPSILON );
    }


With another constructor it is possible to override a variety of parameters from a single number of mixtures (the only essential problem-dependent parameter) to initial values for the mixture parameters.


CvEM
----
.. ocv:class:: CvEM : public CvStatModel

    The class implements the EM algorithm as described in the beginning of the section :ref:`ML_Expectation Maximization`.


CvEM::train
-----------
Estimates the Gaussian mixture parameters from a sample set.

.. ocv:function:: bool CvEM::train( const Mat& samples, const Mat& sampleIdx=Mat(), CvEMParams params=CvEMParams(), Mat* labels=0 )

.. ocv:function:: bool CvEM::train( const CvMat* samples, const CvMat* sampleIdx=0, CvEMParams params=CvEMParams(), CvMat* labels=0 )

.. ocv:pyfunction:: cv2.EM.train(samples[, sampleIdx[, params]]) -> retval, labels

    :param samples: Samples from which the Gaussian mixture model will be estimated.

    :param sample_idx: Mask of samples to use. All samples are used by default.

    :param params: Parameters of the EM algorithm.

    :param labels: The optional output "class label" for each sample: :math:`\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N` (indices of the most probable mixture component for each sample).

Unlike many of the ML models, EM is an unsupervised learning algorithm and it does not take responses (class labels or function values) as input. Instead, it computes the
*Maximum Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the parameters inside the structure:
:math:`p_{i,k}` in ``probs``,
:math:`a_k` in ``means`` ,
:math:`S_k` in ``covs[k]``,
:math:`\pi_k` in ``weights`` , and optionally computes the output "class label" for each sample:
:math:`\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N` (indices of the most probable mixture component for each sample).

The trained model can be used further for prediction, just like any other classifier. The trained model is similar to the
:ocv:class:`CvNormalBayesClassifier`.

For an example of clustering random samples of the multi-Gaussian distribution using EM, see ``em.cpp`` sample in the OpenCV distribution.


CvEM::predict
-------------
Returns a mixture component index of a sample.

.. ocv:function:: float CvEM::predict( const Mat& sample, Mat* probs=0 ) const

.. ocv:function:: float CvEM::predict( const CvMat* sample, CvMat* probs ) const

.. ocv:pyfunction:: cv2.EM.predict(sample) -> retval, probs

    :param sample: A sample for classification.

    :param probs: If it is not null then the method will write posterior probabilities of each component given the sample data to this parameter.


CvEM::getNClusters
------------------
Returns the number of mixture components :math:`M` in the Gaussian mixture model.

.. ocv:function:: int CvEM::getNClusters() const

.. ocv:function:: int CvEM::get_nclusters() const

.. ocv:pyfunction:: cv2.EM.getNClusters() -> retval


CvEM::getMeans
------------------
Returns mixture means :math:`a_k`.

.. ocv:function:: Mat CvEM::getMeans() const

.. ocv:function:: const CvMat* CvEM::get_means() const

.. ocv:pyfunction:: cv2.EM.getMeans() -> means


CvEM::getCovs
-------------
Returns mixture covariance matrices :math:`S_k`.

.. ocv:function:: void CvEM::getCovs(std::vector<cv::Mat>& covs) const

.. ocv:function:: const CvMat** CvEM::get_covs() const

.. ocv:pyfunction:: cv2.EM.getCovs([covs]) -> covs


CvEM::getWeights
----------------
Returns mixture weights :math:`\pi_k`.

.. ocv:function:: Mat CvEM::getWeights() const

.. ocv:function:: const CvMat* CvEM::get_weights() const

.. ocv:pyfunction:: cv2.EM.getWeights() -> weights


CvEM::getProbs
--------------
Returns vectors of probabilities for each training sample.

.. ocv:function:: Mat CvEM::getProbs() const

.. ocv:function:: const CvMat* CvEM::get_probs() const

.. ocv:pyfunction:: cv2.EM.getProbs() -> probs

For each training sample :math:`i` (that have been passed to the constructor or to :ocv:func:`CvEM::train`) returns probabilities :math:`p_{i,k}` to belong to a mixture component :math:`k`.


CvEM::getLikelihood
-------------------
Returns logarithm of likelihood.

.. ocv:function:: double CvEM::getLikelihood() const

.. ocv:function:: double CvEM::get_log_likelihood() const

.. ocv:pyfunction:: cv2.EM.getLikelihood() -> likelihood


CvEM::write
-----------
Writes the trained Gaussian mixture model to the file storage.

.. ocv:function:: void CvEM::write( CvFileStorage* fs, const char* name ) const

    :param fs: A file storage where the model will be written.
    :param name: A name of the file node where the model data will be written.


CvEM::read
-----------------
Reads the trained Gaussian mixture model from the file storage.

.. ocv:function:: void CvEM::read( CvFileStorage* fs, CvFileNode* node )

    :param fs: A file storage with the trained model.

    :param node: The parent map. If it is NULL, the function searches a node with parameters in all the top-level nodes (streams), starting with the first one.

