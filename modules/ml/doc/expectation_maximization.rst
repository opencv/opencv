Expectation Maximization
========================

The EM (Expectation Maximization) algorithm estimates the parameters of the multivariate probability density function in the form of a Gaussian mixture distribution with a specified number of mixtures.

Consider the set of the
:math:`x_1, x_2,...,x_{N}` : N feature vectors?? from a d-dimensional Euclidean space drawn from a Gaussian mixture:

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
:math:`p_{i,k}` . Often (including ML) the
:ref:`kmeans` algorithm is used for that purpose.

One of the main problems?? the EM algorithm should deal with is a large number
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

.. index:: CvEMParams

.. _CvEMParams:

CvEMParams
----------
.. c:type:: CvEMParams

Parameters of the EM algorithm ::

    struct CvEMParams
    {
        CvEMParams() : nclusters(10), cov_mat_type(CvEM::COV_MAT_DIAGONAL),
            start_step(CvEM::START_AUTO_STEP), probs(0), weights(0), means(0),
                                                         covs(0)
        {
            term_crit=cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                                                    100, FLT_EPSILON );
        }

        CvEMParams( int _nclusters, int _cov_mat_type=1/*CvEM::COV_MAT_DIAGONAL*/,
                    int _start_step=0/*CvEM::START_AUTO_STEP*/,
                    CvTermCriteria _term_crit=cvTermCriteria(
                                            CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                                            100, FLT_EPSILON),
                    const CvMat* _probs=0, const CvMat* _weights=0,
                    const CvMat* _means=0, const CvMat** _covs=0 ) :
                    nclusters(_nclusters), cov_mat_type(_cov_mat_type),
                    start_step(_start_step),
                    probs(_probs), weights(_weights), means(_means), covs(_covs),
                    term_crit(_term_crit)
        {}

        int nclusters;
        int cov_mat_type;
        int start_step;
        const CvMat* probs;
        const CvMat* weights;
        const CvMat* means;
        const CvMat** covs;
        CvTermCriteria term_crit;
    };


The structure has two constructors. The default one represents a rough rule-of-the-thumb. With another one it is possible to override a variety of parameters from a single number of mixtures (the only essential problem-dependent parameter) to initial values for the mixture parameters.

.. index:: CvEM

.. _CvEM:

CvEM
----
.. c:type:: CvEM

EM model ::

    class CV_EXPORTS CvEM : public CvStatModel
    {
    public:
        // Type of covariance matrices
        enum { COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2 };

        // Initial step
        enum { START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0 };

        CvEM();
        CvEM( const Mat& samples, const Mat& sample_idx=Mat(),
              CvEMParams params=CvEMParams(), Mat* labels=0 );
        virtual ~CvEM();

        virtual bool train( const Mat& samples, const Mat& sample_idx=Mat(),
                            CvEMParams params=CvEMParams(), Mat* labels=0 );

        virtual float predict( const Mat& sample, Mat& probs ) const;
        virtual void clear();

        int get_nclusters() const { return params.nclusters; }
        const Mat& get_means() const { return means; }
        const Mat&* get_covs() const { return covs; }
        const Mat& get_weights() const { return weights; }
        const Mat& get_probs() const { return probs; }

    protected:

        virtual void set_params( const CvEMParams& params,
                                 const CvVectors& train_data );
        virtual void init_em( const CvVectors& train_data );
        virtual double run_em( const CvVectors& train_data );
        virtual void init_auto( const CvVectors& samples );
        virtual void kmeans( const CvVectors& train_data, int nclusters,
                             Mat& labels, CvTermCriteria criteria,
                             const Mat& means );
        CvEMParams params;
        double log_likelihood;

        Mat& means;
        Mat&* covs;
        Mat& weights;
        Mat& probs;

        Mat& log_weight_div_det;
        Mat& inv_eigen_values;
        Mat&* cov_rotate_mats;
    };


.. index:: CvEM::train

.. _CvEM::train:

CvEM::train
-----------
.. cpp:function:: void CvEM::train(  const Mat& samples,  const Mat&  sample_idx=Mat(),                    CvEMParams params=CvEMParams(),  Mat* labels=0 )

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

For example of clustering random samples of multi-Gaussian distribution using EM see em.cpp sample in OpenCV distribution.



