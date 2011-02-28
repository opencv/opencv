Expectation-Maximization
========================

.. highlight:: cpp

The EM (Expectation-Maximization) algorithm estimates the parameters of the multivariate probability density function in the form of a Gaussian mixture distribution with a specified number of mixtures.

Consider the set of the feature vectors
:math:`x_1, x_2,...,x_{N}` : N vectors from a d-dimensional Euclidean space drawn from a Gaussian mixture:

.. math::

    p(x;a_k,S_k, \pi _k) =  \sum _{k=1}^{m} \pi _kp_k(x),  \quad \pi _k  \geq 0,  \quad \sum _{k=1}^{m} \pi _k=1,

.. math::

    p_k(x)= \varphi (x;a_k,S_k)= \frac{1}{(2\pi)^{d/2}\mid{S_k}\mid^{1/2}} exp \left \{ - \frac{1}{2} (x-a_k)^TS_k^{-1}(x-a_k) \right \} ,

where
:math:`m` is the number of mixtures,
:math:`p_k` is the normal distribution
density with the mean
:math:`a_k` and covariance matrix
:math:`S_k`,:math:`\pi_k` is the weight of the k-th mixture. Given the number of mixtures
:math:`M` and the samples
:math:`x_i`,:math:`i=1..N` the algorithm finds the
maximum-likelihood estimates (MLE) of the all the mixture parameters,
i.e.
:math:`a_k`,:math:`S_k` and
:math:`\pi_k` :

.. math::

    L(x, \theta )=logp(x, \theta )= \sum _{i=1}^{N}log \left ( \sum _{k=1}^{m} \pi _kp_k(x) \right ) \to \max _{ \theta \in \Theta },

.. math::

    \Theta = \left \{ (a_k,S_k, \pi _k): a_k  \in \mathbbm{R} ^d,S_k=S_k^T>0,S_k  \in \mathbbm{R} ^{d  \times d}, \pi _k \geq 0, \sum _{k=1}^{m} \pi _k=1 \right \} .

EM algorithm is an iterative procedure. Each iteration of it includes
two steps. At the first step (Expectation-step, or E-step), we find a
probability
:math:`p_{i,k}` (denoted
:math:`\alpha_{i,k}` in the formula below) of
sample ``i`` to belong to mixture ``k`` using the currently
available mixture parameter estimates:

.. math::

    \alpha _{ki} =  \frac{\pi_k\varphi(x;a_k,S_k)}{\sum\limits_{j=1}^{m}\pi_j\varphi(x;a_j,S_j)} .

At the second step (Maximization-step, or M-step) the mixture parameter estimates are refined using the computed probabilities:

.. math::

    \pi _k= \frac{1}{N} \sum _{i=1}^{N} \alpha _{ki},  \quad a_k= \frac{\sum\limits_{i=1}^{N}\alpha_{ki}x_i}{\sum\limits_{i=1}^{N}\alpha_{ki}} ,  \quad S_k= \frac{\sum\limits_{i=1}^{N}\alpha_{ki}(x_i-a_k)(x_i-a_k)^T}{\sum\limits_{i=1}^{N}\alpha_{ki}} ,

Alternatively, the algorithm may start with the M-step when the initial values for
:math:`p_{i,k}` can be provided. Another alternative when
:math:`p_{i,k}` are unknown, is to use a simpler clustering algorithm to pre-cluster the input samples and thus obtain initial
:math:`p_{i,k}` . Often (and in ML) the
:ref:`KMeans2` algorithm is used for that purpose.

One of the main that EM algorithm should deal with is the large number
of parameters to estimate. The majority of the parameters sits in
covariance matrices, which are
:math:`d \times d` elements each
(where
:math:`d` is the feature space dimensionality). However, in
many practical problems the covariance matrices are close to diagonal,
or even to
:math:`\mu_k*I` , where
:math:`I` is identity matrix and
:math:`\mu_k` is mixture-dependent "scale" parameter. So a robust computation
scheme could be to start with the harder constraints on the covariance
matrices and then use the estimated parameters as an input for a less
constrained optimization problem (often a diagonal covariance matrix is
already a good enough approximation).

**References:**

*
    Bilmes98 J. A. Bilmes. A Gentle Tutorial of the EM Algorithm and its Application to Parameter Estimation for Gaussian Mixture and Hidden Markov Models. Technical Report TR-97-021, International Computer Science Institute and Computer Science Division, University of California at Berkeley, April 1998.

.. index:: CvEMParams

.. _CvEMParams:

CvEMParams
----------
.. c:type:: CvEMParams

Parameters of the EM algorithm. ::

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
                    CvMat* _probs=0, CvMat* _weights=0,
                    CvMat* _means=0, CvMat** _covs=0 ) :
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
..

The structure has 2 constructors, the default one represents a rough rule-of-thumb, with another one it is possible to override a variety of parameters, from a single number of mixtures (the only essential problem-dependent parameter), to the initial values for the mixture parameters.

.. index:: CvEM

.. _CvEM:

CvEM
----
.. c:type:: CvEM

EM model. ::

    class CV_EXPORTS CvEM : public CvStatModel
    {
    public:
        // Type of covariance matrices
        enum { COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2 };

        // The initial step
        enum { START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0 };

        CvEM();
        CvEM( const CvMat* samples, const CvMat* sample_idx=0,
              CvEMParams params=CvEMParams(), CvMat* labels=0 );
        virtual ~CvEM();

        virtual bool train( const CvMat* samples, const CvMat* sample_idx=0,
                            CvEMParams params=CvEMParams(), CvMat* labels=0 );

        virtual float predict( const CvMat* sample, CvMat* probs ) const;
        virtual void clear();

        int get_nclusters() const { return params.nclusters; }
        const CvMat* get_means() const { return means; }
        const CvMat** get_covs() const { return covs; }
        const CvMat* get_weights() const { return weights; }
        const CvMat* get_probs() const { return probs; }

    protected:

        virtual void set_params( const CvEMParams& params,
                                 const CvVectors& train_data );
        virtual void init_em( const CvVectors& train_data );
        virtual double run_em( const CvVectors& train_data );
        virtual void init_auto( const CvVectors& samples );
        virtual void kmeans( const CvVectors& train_data, int nclusters,
                             CvMat* labels, CvTermCriteria criteria,
                             const CvMat* means );
        CvEMParams params;
        double log_likelihood;

        CvMat* means;
        CvMat** covs;
        CvMat* weights;
        CvMat* probs;

        CvMat* log_weight_div_det;
        CvMat* inv_eigen_values;
        CvMat** cov_rotate_mats;
    };
..

.. index:: CvEM::train

.. _CvEM::train:

CvEM::train
-----------
.. c:function:: void CvEM::train(  const CvMat* samples,  const CvMat*  sample_idx=0,                    CvEMParams params=CvEMParams(),  CvMat* labels=0 )

    Estimates the Gaussian mixture parameters from the sample set.

Unlike many of the ML models, EM is an unsupervised learning algorithm and it does not take responses (class labels or the function values) on input. Instead, it computes the
:ref:`MLE` of the Gaussian mixture parameters from the input sample set, stores all the parameters inside the structure:
:math:`p_{i,k}` in ``probs``,:math:`a_k` in ``means`` :math:`S_k` in ``covs[k]``,:math:`\pi_k` in ``weights`` and optionally computes the output "class label" for each sample:
:math:`\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N` (i.e. indices of the most-probable mixture for each sample).

The trained model can be used further for prediction, just like any other classifier. The model trained is similar to the
:ref:`Bayes classifier` .

Example: Clustering random samples of multi-Gaussian distribution using EM ::

    #include "ml.h"
    #include "highgui.h"

    int main( int argc, char** argv )
    {
        const int N = 4;
        const int N1 = (int)sqrt((double)N);
        const CvScalar colors[] = {{0,0,255}},{{0,255,0}},
                                        {{0,255,255}},{{255,255,0}
                                        ;
        int i, j;
        int nsamples = 100;
        CvRNG rng_state = cvRNG(-1);
        CvMat* samples = cvCreateMat( nsamples, 2, CV_32FC1 );
        CvMat* labels = cvCreateMat( nsamples, 1, CV_32SC1 );
        IplImage* img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
        float _sample[2];
        CvMat sample = cvMat( 1, 2, CV_32FC1, _sample );
        CvEM em_model;
        CvEMParams params;
        CvMat samples_part;

        cvReshape( samples, samples, 2, 0 );
        for( i = 0; i < N; i++ )
        {
            CvScalar mean, sigma;

            // form the training samples
            cvGetRows( samples, &samples_part, i*nsamples/N,
                                               (i+1)*nsamples/N );
            mean = cvScalar(((i
                           ((i/N1)+1.)*img->height/(N1+1));
            sigma = cvScalar(30,30);
            cvRandArr( &rng_state, &samples_part, CV_RAND_NORMAL,
                                                            mean, sigma );
        }
        cvReshape( samples, samples, 1, 0 );

        // initialize model's parameters
        params.covs      = NULL;
        params.means     = NULL;
        params.weights   = NULL;
        params.probs     = NULL;
        params.nclusters = N;
        params.cov_mat_type       = CvEM::COV_MAT_SPHERICAL;
        params.start_step         = CvEM::START_AUTO_STEP;
        params.term_crit.max_iter = 10;
        params.term_crit.epsilon  = 0.1;
        params.term_crit.type     = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;

        // cluster the data
        em_model.train( samples, 0, params, labels );

    #if 0
        // the piece of code shows how to repeatedly optimize the model
        // with less-constrained parameters
        //(COV_MAT_DIAGONAL instead of COV_MAT_SPHERICAL)
        // when the output of the first stage is used as input for the second.
        CvEM em_model2;
        params.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
        params.start_step = CvEM::START_E_STEP;
        params.means = em_model.get_means();
        params.covs = (const CvMat**)em_model.get_covs();
        params.weights = em_model.get_weights();

        em_model2.train( samples, 0, params, labels );
        // to use em_model2, replace em_model.predict()
        // with em_model2.predict() below
    #endif
        // classify every image pixel
        cvZero( img );
        for( i = 0; i < img->height; i++ )
        {
            for( j = 0; j < img->width; j++ )
            {
                CvPoint pt = cvPoint(j, i);
                sample.data.fl[0] = (float)j;
                sample.data.fl[1] = (float)i;
                int response = cvRound(em_model.predict( &sample, NULL ));
                CvScalar c = colors[response];

                cvCircle( img, pt, 1, cvScalar(c.val[0]*0.75,
                    c.val[1]*0.75,c.val[2]*0.75), CV_FILLED );
            }
        }

        //draw the clustered samples
        for( i = 0; i < nsamples; i++ )
        {
            CvPoint pt;
            pt.x = cvRound(samples->data.fl[i*2]);
            pt.y = cvRound(samples->data.fl[i*2+1]);
            cvCircle( img, pt, 1, colors[labels->data.i[i]], CV_FILLED );
        }

        cvNamedWindow( "EM-clustering result", 1 );
        cvShowImage( "EM-clustering result", img );
        cvWaitKey(0);

        cvReleaseMat( &samples );
        cvReleaseMat( &labels );
        return 0;
    }
..

