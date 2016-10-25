/*
  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.


                          BSD 3-Clause License

 Copyright (C) 2014, Olexa Bilaniuk, Hamid Bazargani & Robert Laganiere, all rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * The name of the copyright holders may not be used to endorse or promote products
     derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
*/

/**
 * Bilaniuk, Olexa, Hamid Bazargani, and Robert Laganiere. "Fast Target
 * Recognition on Mobile Devices: Revisiting Gaussian Elimination for the
 * Estimation of Planar Homographies." In Computer Vision and Pattern
 * Recognition Workshops (CVPRW), 2014 IEEE Conference on, pp. 119-125.
 * IEEE, 2014.
 */

/* Includes */
#include "precomp.hpp"
#include <opencv2/core.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <vector>
#include "rho.h"





/* For the sake of cv:: namespace ONLY: */
namespace cv{/* For C support, replace with extern "C" { */


/* Constants */
const int    MEM_ALIGN              = 32;
const size_t HSIZE                  = (3*3*sizeof(float));
const double MIN_DELTA_CHNG         = 0.1;
// const double CHI_STAT               = 2.706;
const double CHI_SQ                 = 1.645;
// const double RLO                    = 0.25;
// const double RHI                    = 0.75;
const int    MAXLEVMARQITERS        = 100;
const int    SMPL_SIZE              = 4;      /* 4 points required per model */
const int    SPRT_T_M               = 25;     /* Guessing 25 match evlauations / 1 model generation */
const int    SPRT_M_S               = 1;      /* 1 model per sample */
const double SPRT_EPSILON           = 0.1;    /* No explanation */
const double SPRT_DELTA             = 0.01;   /* No explanation */
const double LM_GAIN_LO             = 0.25;   /* See sacLMGain(). */
const double LM_GAIN_HI             = 0.75;   /* See sacLMGain(). */


/* Data Structures */

/**
 * Base Struct for RHO algorithm.
 *
 * A RHO estimator has initialization, finalization, capacity, seeding and
 * homography-estimation APIs that must be implemented.
 */

struct RHO_HEST{
    /* This is a virtual base class; It should have a virtual destructor. */
    virtual ~RHO_HEST(){}

    /* External Interface Methods */

    /**
     * Initialization work.
     *
     * @return 0 if initialization is unsuccessful; non-zero otherwise.
     */

    virtual inline int    initialize(void){return 1;}


    /**
     * Finalization work.
     */

    virtual inline void   finalize(void){}

    /**
     * Ensure that the estimator context's internal table for the non-randomness
     * criterion is at least of the given size, and uses the given beta. The table
     * should be larger than the maximum number of matches fed into the estimator.
     *
     * A value of N of 0 requests deallocation of the table.
     *
     * @param [in] N     If 0, deallocate internal table. If > 0, ensure that the
     *                   internal table is of at least this size, reallocating if
     *                   necessary.
     * @param [in] beta  The beta-factor to use within the table.
     * @return 0 if unsuccessful; non-zero otherwise.
     */

    virtual inline int    ensureCapacity(unsigned N, double beta){
        (void)N;
        (void)beta;

        return 1;
    }


    /**
     * Generates a random double uniformly distributed in the range [0, 1).
     *
     * The default implementation uses the xorshift128+ algorithm from
     * Sebastiano Vigna. Further scramblings of Marsaglia's xorshift generators.
     * CoRR, abs/1402.6246, 2014.
     * http://vigna.di.unimi.it/ftp/papers/xorshiftplus.pdf
     *
     * Source roughly as given in
     * http://en.wikipedia.org/wiki/Xorshift#Xorshift.2B
     */

    virtual inline double fastRandom(void){
        uint64_t x = prng.s[0];
        uint64_t y = prng.s[1];
        x ^= x << 23; // a
        x ^= x >> 17; // b
        x ^= y ^ (y >> 26); // c
        prng.s[0] = y;
        prng.s[1] = x;
        uint64_t s = x + y;

        return s * 5.421010862427522e-20;/* 2^-64 */
    }


    /**
     * Seeds the context's PRNG.
     *
     * @param [in] seed  A 64-bit unsigned integer seed.
     */

    virtual inline void   fastSeed(uint64_t seed){
        int i;

        prng.s[0] =  seed;
        prng.s[1] = ~seed;/* Guarantees one of the elements will be non-zero. */

        /**
         * Escape from zero-land (see xorshift128+ paper). Approximately 20
         * iterations required according to the graph.
         */

        for(i=0;i<20;i++){
            fastRandom();
        }
    }


    /**
     * Estimates the homography using the given context, matches and parameters to
     * PROSAC.
     *
     * @param [in]     src     The pointer to the source points of the matches.
     *                             Cannot be NULL.
     * @param [in]     dst     The pointer to the destination points of the matches.
     *                             Cannot be NULL.
     * @param [out]    inl     The pointer to the output mask of inlier matches.
     *                             May be NULL.
     * @param [in]     N       The number of matches.
     * @param [in]     maxD    The maximum distance.
     * @param [in]     maxI    The maximum number of PROSAC iterations.
     * @param [in]     rConvg  The RANSAC convergence parameter.
     * @param [in]     cfd     The required confidence in the solution.
     * @param [in]     minInl  The minimum required number of inliers.
     * @param [in]     beta    The beta-parameter for the non-randomness criterion.
     * @param [in]     flags   A union of flags to control the estimation.
     * @param [in]     guessH  An extrinsic guess at the solution H, or NULL if
     *                         none provided.
     * @param [out]    finalH  The final estimation of H, or the zero matrix if
     *                         the minimum number of inliers was not met.
     *                         Cannot be NULL.
     * @return                 The number of inliers if the minimum number of
     *                         inliers for acceptance was reached; 0 otherwise.
     */

    virtual unsigned      rhoHest(const float*   src,     /* Source points */
                                  const float*   dst,     /* Destination points */
                                  char*          inl,     /* Inlier mask */
                                  unsigned       N,       /*  = src.length = dst.length = inl.length */
                                  float          maxD,    /* Works:     3.0 */
                                  unsigned       maxI,    /* Works:    2000 */
                                  unsigned       rConvg,  /* Works:    2000 */
                                  double         cfd,     /* Works:   0.995 */
                                  unsigned       minInl,  /* Minimum:     4 */
                                  double         beta,    /* Works:    0.35 */
                                  unsigned       flags,   /* Works:       0 */
                                  const float*   guessH,  /* Extrinsic guess, NULL if none provided */
                                  float*         finalH) = 0; /* Final result. */



    /* PRNG XORshift128+ */
    struct{
        uint64_t  s[2];            /* PRNG state */
    } prng;
};



/**
 * Generic C implementation of RHO algorithm.
 */

struct RHO_HEST_REFC : RHO_HEST{
    /**
     * Virtual Arguments.
     *
     * Exactly the same as at function call, except:
     * - minInl is enforced to be >= 4.
     */

    struct{
        const float* src;
        const float* dst;
        char*        inl;
        unsigned     N;
        float        maxD;
        unsigned     maxI;
        unsigned     rConvg;
        double       cfd;
        unsigned     minInl;
        double       beta;
        unsigned     flags;
        const float* guessH;
        float*       finalH;
    } arg;

    /* PROSAC Control */
    struct{
        unsigned  i;               /* Iteration Number */
        unsigned  phNum;           /* Phase Number */
        unsigned  phEndI;          /* Phase End Iteration */
        double    phEndFpI;        /* Phase floating-point End Iteration */
        unsigned  phMax;           /* Termination phase number */
        unsigned  phNumInl;        /* Number of inliers for termination phase */
        unsigned  numModels;       /* Number of models tested */
        unsigned* smpl;            /* Sample of match indexes */
    } ctrl;

    /* Current model being tested */
    struct{
        float*    pkdPts;          /* Packed points */
        float*    H;               /* Homography */
        char*     inl;             /* Mask of inliers */
        unsigned  numInl;          /* Number of inliers */
    } curr;

    /* Best model (so far) */
    struct{
        float*    H;               /* Homography */
        char*     inl;             /* Mask of inliers */
        unsigned  numInl;          /* Number of inliers */
    } best;

    /* Non-randomness criterion */
    struct{
        std::vector<unsigned> tbl; /* Non-Randomness: Table */
        unsigned  size;            /* Non-Randomness: Size */
        double    beta;            /* Non-Randomness: Beta */
    } nr;

    /* SPRT Evaluator */
    struct{
        double    t_M;             /* t_M */
        double    m_S;             /* m_S */
        double    epsilon;         /* Epsilon */
        double    delta;           /* delta */
        double    A;               /* SPRT Threshold */
        unsigned  Ntested;         /* Number of points tested */
        unsigned  Ntestedtotal;    /* Number of points tested in total */
        int       good;            /* Good/bad flag */
        double    lambdaAccept;    /* Accept multiplier */
        double    lambdaReject;    /* Reject multiplier */
    } eval;

    /* Levenberg-Marquardt Refinement */
    struct{
        float  (* JtJ)[8];         /* JtJ matrix */
        float  (* tmp1)[8];        /* Temporary 1 */
        float*    Jte;             /* Jte vector */
    } lm;

    /* Memory Management */
    struct{
        cv::Mat perObj;
        cv::Mat perRun;
    } mem;

    /* Initialized? */
    int initialized;


    /* Empty constructors and destructors */
    public:
    RHO_HEST_REFC();
    private: /* Forbid copying. */
    RHO_HEST_REFC(const RHO_HEST_REFC&);
    public:
    ~RHO_HEST_REFC();

    /* Methods to implement external interface */
    inline int    initialize(void);
    inline void   finalize(void);
    inline int    ensureCapacity(unsigned N, double beta);
    unsigned      rhoHest(const float*   src,     /* Source points */
                          const float*   dst,     /* Destination points */
                          char*          inl,     /* Inlier mask */
                          unsigned       N,       /*  = src.length = dst.length = inl.length */
                          float          maxD,    /* Works:     3.0 */
                          unsigned       maxI,    /* Works:    2000 */
                          unsigned       rConvg,  /* Works:    2000 */
                          double         cfd,     /* Works:   0.995 */
                          unsigned       minInl,  /* Minimum:     4 */
                          double         beta,    /* Works:    0.35 */
                          unsigned       flags,   /* Works:       0 */
                          const float*   guessH,  /* Extrinsic guess, NULL if none provided */
                          float*         finalH); /* Final result. */



    /* Methods to implement internals */
    inline void   allocatePerObj(void);
    inline void   allocatePerRun(void);
    inline void   deallocatePerRun(void);
    inline void   deallocatePerObj(void);
    inline int    initRun(void);
    inline void   finiRun(void);
    inline int    haveExtrinsicGuess(void);
    inline int    hypothesize(void);
    inline int    verify(void);
    inline int    isNREnabled(void);
    inline int    isRefineEnabled(void);
    inline int    isFinalRefineEnabled(void);
    inline int    PROSACPhaseEndReached(void);
    inline void   PROSACGoToNextPhase(void);
    inline void   getPROSACSample(void);
    inline void   rndSmpl(unsigned  sampleSize,
                          unsigned* currentSample,
                          unsigned  dataSetSize);
    inline int    isSampleDegenerate(void);
    inline void   generateModel(void);
    inline int    isModelDegenerate(void);
    inline void   evaluateModelSPRT(void);
    inline void   updateSPRT(void);
    inline void   designSPRTTest(void);
    inline int    isBestModel(void);
    inline int    isBestModelGoodEnough(void);
    inline void   saveBestModel(void);
    inline void   nStarOptimize(void);
    inline void   updateBounds(void);
    inline void   outputModel(void);
    inline void   outputZeroH(void);
    inline int    canRefine(void);
    inline void   refine(void);
};




/**
 * Prototypes for purely-computational code.
 */

static inline void   sacInitNonRand       (double    beta,
                                           unsigned  start,
                                           unsigned  N,
                                           unsigned* nonRandMinInl);
static inline double sacInitPEndFpI       (const unsigned ransacConvg,
                                           const unsigned n,
                                           const unsigned s);
static inline unsigned sacCalcIterBound   (double   confidence,
                                           double   inlierRate,
                                           unsigned sampleSize,
                                           unsigned maxIterBound);
static inline void   hFuncRefC            (float* packedPoints, float* H);
static inline void   sacCalcJacobianErrors(const float* H,
                                           const float* src,
                                           const float* dst,
                                           const char*  inl,
                                           unsigned     N,
                                           float     (* JtJ)[8],
                                           float*       Jte,
                                           float*       Sp);
static inline float  sacLMGain            (const float*  dH,
                                           const float*  Jte,
                                           const float   S,
                                           const float   newS,
                                           const float   lambda);
static inline int    sacChol8x8Damped     (const float (*A)[8],
                                           float         lambda,
                                           float       (*L)[8]);
static inline void   sacTRInv8x8          (const float (*L)[8],
                                           float       (*M)[8]);
static inline void   sacTRISolve8x8       (const float (*L)[8],
                                           const float*  Jte,
                                           float*        dH);
static inline void   sacSub8x1            (float*       Hout,
                                           const float* H,
                                           const float* dH);



/* Functions */

/**
 * External access to context constructor.
 *
 * @return A pointer to the context if successful; NULL if an error occured.
 */

Ptr<RHO_HEST> rhoInit(void){
    /* Select an optimized implementation of RHO here. */

#if 1
    /**
     * For now, only the generic C implementation is available. In the future,
     * SSE2/AVX/AVX2/FMA/NEON versions may be added, and they will be selected
     * depending on cv::checkHardwareSupport()'s return values.
     */

    Ptr<RHO_HEST> p = Ptr<RHO_HEST>(new RHO_HEST_REFC);
#endif

    /* Initialize it. */
    if(p){
        if(!p->initialize()){
            p.release();
        }
    }

    /* Return it. */
    return p;
}


/**
 * External access to non-randomness table resize.
 */

int  rhoEnsureCapacity(Ptr<RHO_HEST> p, unsigned N, double beta){
    return p->ensureCapacity(N, beta);
}


/**
 * Seeds the internal PRNG with the given seed.
 */

void rhoSeed(Ptr<RHO_HEST> p, uint64_t seed){
    p->fastSeed(seed);
}


/**
 * Estimates the homography using the given context, matches and parameters to
 * PROSAC.
 *
 * @param [in/out] p       The context to use for homography estimation. Must
 *                             be already initialized. Cannot be NULL.
 * @param [in]     src     The pointer to the source points of the matches.
 *                             Must be aligned to 4 bytes. Cannot be NULL.
 * @param [in]     dst     The pointer to the destination points of the matches.
 *                             Must be aligned to 16 bytes. Cannot be NULL.
 * @param [out]    inl     The pointer to the output mask of inlier matches.
 *                             Must be aligned to 16 bytes. May be NULL.
 * @param [in]     N       The number of matches.
 * @param [in]     maxD    The maximum distance.
 * @param [in]     maxI    The maximum number of PROSAC iterations.
 * @param [in]     rConvg  The RANSAC convergence parameter.
 * @param [in]     cfd     The required confidence in the solution.
 * @param [in]     minInl  The minimum required number of inliers.
 * @param [in]     beta    The beta-parameter for the non-randomness criterion.
 * @param [in]     flags   A union of flags to control the estimation.
 * @param [in]     guessH  An extrinsic guess at the solution H, or NULL if
 *                         none provided.
 * @param [out]    finalH  The final estimation of H, or the zero matrix if
 *                         the minimum number of inliers was not met.
 *                         Cannot be NULL.
 * @return                 The number of inliers if the minimum number of
 *                         inliers for acceptance was reached; 0 otherwise.
 */

unsigned rhoHest(Ptr<RHO_HEST> p,       /* Homography estimation context. */
                 const float*  src,     /* Source points */
                 const float*  dst,     /* Destination points */
                 char*         inl,     /* Inlier mask */
                 unsigned      N,       /*  = src.length = dst.length = inl.length */
                 float         maxD,    /* Works:     3.0 */
                 unsigned      maxI,    /* Works:    2000 */
                 unsigned      rConvg,  /* Works:    2000 */
                 double        cfd,     /* Works:   0.995 */
                 unsigned      minInl,  /* Minimum:     4 */
                 double        beta,    /* Works:    0.35 */
                 unsigned      flags,   /* Works:       0 */
                 const float*  guessH,  /* Extrinsic guess, NULL if none provided */
                 float*        finalH){ /* Final result. */
    return p->rhoHest(src, dst, inl, N, maxD, maxI, rConvg, cfd, minInl, beta,
                      flags, guessH, finalH);
}












/*********************** RHO_HEST_REFC implementation **********************/

/**
 * Constructor for RHO_HEST_REFC.
 *
 * Does nothing. True initialization is done by initialize().
 */

RHO_HEST_REFC::RHO_HEST_REFC() : initialized(0){

}

/**
 * Private copy constructor for RHO_HEST_REFC. Disabled.
 */

RHO_HEST_REFC::RHO_HEST_REFC(const RHO_HEST_REFC&) : initialized(0){

}

/**
 * Destructor for RHO_HEST_REFC.
 */

RHO_HEST_REFC::~RHO_HEST_REFC(){
    if(initialized){
        finalize();
    }
}



/**
 * Initialize the estimator context, by allocating the aligned buffers
 * internally needed.
 *
 * Currently there are 5 per-estimator buffers:
 * - The buffer of m indexes representing a sample
 * - The buffer of 16 floats representing m matches (x,y) -> (X,Y).
 * - The buffer for the current homography
 * - The buffer for the best-so-far homography
 * - Optionally, the non-randomness criterion table
 *
 * Returns 0 if unsuccessful and non-0 otherwise.
 */

inline int    RHO_HEST_REFC::initialize(void){
    initialized = 0;


    allocatePerObj();

    curr.inl    = NULL;
    curr.numInl = 0;

    best.inl    = NULL;
    best.numInl = 0;

    nr.size     = 0;
    nr.beta     = 0.0;


    fastSeed((uint64_t)~0);


    int areAllAllocsSuccessful = !mem.perObj.empty();

    if(!areAllAllocsSuccessful){
        finalize();
    }else{
        initialized = 1;
    }

    return areAllAllocsSuccessful;
}

/**
 * Finalize.
 *
 * Finalize the estimator context, by freeing the aligned buffers used
 * internally.
 */

inline void   RHO_HEST_REFC::finalize(void){
    if(initialized){
        deallocatePerObj();

        initialized = 0;
    }
}

/**
 * Ensure that the estimator context's internal table for non-randomness
 * criterion is at least of the given size, and uses the given beta. The table
 * should be larger than the maximum number of matches fed into the estimator.
 *
 * A value of N of 0 requests deallocation of the table.
 *
 * @param [in] N     If 0, deallocate internal table. If > 0, ensure that the
 *                   internal table is of at least this size, reallocating if
 *                   necessary.
 * @param [in] beta  The beta-factor to use within the table.
 * @return 0 if unsuccessful; non-zero otherwise.
 *
 * Reads:  nr.*
 * Writes: nr.*
 */

inline int    RHO_HEST_REFC::ensureCapacity(unsigned N, double beta){
    if(N == 0){
        /* Clear. */
        nr.tbl.clear();
        nr.size = 0;
    }else if(nr.beta != beta){
        /* Beta changed. Redo all the work. */
        nr.tbl.resize(N);
        nr.beta = beta;
        sacInitNonRand(nr.beta, 0, N, &nr.tbl[0]);
        nr.size = N;
    }else if(N > nr.size){
        /* Work is partially done. Do rest of it. */
        nr.tbl.resize(N);
        sacInitNonRand(nr.beta, nr.size, N, &nr.tbl[nr.size]);
        nr.size = N;
    }else{
        /* Work is already done. Do nothing. */
    }

    return 1;
}


/**
 * Estimates the homography using the given context, matches and parameters to
 * PROSAC.
 *
 * @param [in]     src     The pointer to the source points of the matches.
 *                             Must be aligned to 4 bytes. Cannot be NULL.
 * @param [in]     dst     The pointer to the destination points of the matches.
 *                             Must be aligned to 4 bytes. Cannot be NULL.
 * @param [out]    inl     The pointer to the output mask of inlier matches.
 *                             Must be aligned to 4 bytes. May be NULL.
 * @param [in]     N       The number of matches.
 * @param [in]     maxD    The maximum distance.
 * @param [in]     maxI    The maximum number of PROSAC iterations.
 * @param [in]     rConvg  The RANSAC convergence parameter.
 * @param [in]     cfd     The required confidence in the solution.
 * @param [in]     minInl  The minimum required number of inliers.
 * @param [in]     beta    The beta-parameter for the non-randomness criterion.
 * @param [in]     flags   A union of flags to control the estimation.
 * @param [in]     guessH  An extrinsic guess at the solution H, or NULL if
 *                         none provided.
 * @param [out]    finalH  The final estimation of H, or the zero matrix if
 *                         the minimum number of inliers was not met.
 *                         Cannot be NULL.
 * @return                 The number of inliers if the minimum number of
 *                         inliers for acceptance was reached; 0 otherwise.
 */

unsigned RHO_HEST_REFC::rhoHest(const float*   src,     /* Source points */
                                const float*   dst,     /* Destination points */
                                char*          inl,     /* Inlier mask */
                                unsigned       N,       /*  = src.length = dst.length = inl.length */
                                float          maxD,    /* Works:     3.0 */
                                unsigned       maxI,    /* Works:    2000 */
                                unsigned       rConvg,  /* Works:    2000 */
                                double         cfd,     /* Works:   0.995 */
                                unsigned       minInl,  /* Minimum:     4 */
                                double         beta,    /* Works:    0.35 */
                                unsigned       flags,   /* Works:       0 */
                                const float*   guessH,  /* Extrinsic guess, NULL if none provided */
                                float*         finalH){ /* Final result. */

    /**
     * Setup
     */

    arg.src     = src;
    arg.dst     = dst;
    arg.inl     = inl;
    arg.N       = N;
    arg.maxD    = maxD;
    arg.maxI    = maxI;
    arg.rConvg  = rConvg;
    arg.cfd     = cfd;
    arg.minInl  = minInl;
    arg.beta    = beta;
    arg.flags   = flags;
    arg.guessH  = guessH;
    arg.finalH  = finalH;
    if(!initRun()){
        outputZeroH();
        finiRun();
        return 0;
    }

    /**
     * Extrinsic Guess
     */

    if(haveExtrinsicGuess()){
        verify();
    }


    /**
     * PROSAC Loop
     */

    for(ctrl.i=0; ctrl.i < arg.maxI || ctrl.i < 100; ctrl.i++){
        hypothesize() && verify();
    }


    /**
     * Teardown
     */

    if(isFinalRefineEnabled() && canRefine()){
        refine();
    }

    outputModel();
    finiRun();
    return isBestModelGoodEnough() ? best.numInl : 0;
}


/**
 * Allocate per-object dynamic storage.
 *
 * This includes aligned, fixed-size internal buffers, but excludes any buffers
 * whose size cannot be determined ahead-of-time (before the number of matches
 * is known).
 *
 * All buffer memory is allocated in one single shot, and all pointers are
 * initialized.
 */

inline void   RHO_HEST_REFC::allocatePerObj(void){
    /* We have known sizes */
    size_t ctrl_smpl_sz   = SMPL_SIZE*sizeof(*ctrl.smpl);
    size_t curr_pkdPts_sz = SMPL_SIZE*2*2*sizeof(*curr.pkdPts);
    size_t curr_H_sz      = HSIZE;
    size_t best_H_sz      = HSIZE;
    size_t lm_JtJ_sz      = 8*8*sizeof(float);
    size_t lm_tmp1_sz     = 8*8*sizeof(float);
    size_t lm_Jte_sz      = 1*8*sizeof(float);

    /* We compute offsets */
    size_t total = 0;
#define MK_OFFSET(v)                                     \
    size_t v ## _of = total;                             \
    total = alignSize(v ## _of  +  v ## _sz, MEM_ALIGN)

    MK_OFFSET(ctrl_smpl);
    MK_OFFSET(curr_pkdPts);
    MK_OFFSET(curr_H);
    MK_OFFSET(best_H);
    MK_OFFSET(lm_JtJ);
    MK_OFFSET(lm_tmp1);
    MK_OFFSET(lm_Jte);

#undef MK_OFFSET

    /* Allocate dynamic memory managed by cv::Mat */
    mem.perObj.create(1, (int)(total + MEM_ALIGN), CV_8UC1);

    /* Extract aligned pointer */
    unsigned char* ptr = alignPtr(mem.perObj.data, MEM_ALIGN);

    /* Assign pointers */
    ctrl.smpl   = (unsigned*)  (ptr + ctrl_smpl_of);
    curr.pkdPts = (float*)     (ptr + curr_pkdPts_of);
    curr.H      = (float*)     (ptr + curr_H_of);
    best.H      = (float*)     (ptr + best_H_of);
    lm.JtJ      = (float(*)[8])(ptr + lm_JtJ_of);
    lm.tmp1     = (float(*)[8])(ptr + lm_tmp1_of);
    lm.Jte      = (float*)     (ptr + lm_Jte_of);
}


/**
 * Allocate per-run dynamic storage.
 *
 * This includes storage that is proportional to the number of points, such as
 * the inlier mask.
 */

inline void   RHO_HEST_REFC::allocatePerRun(void){
    /* We have known sizes */
    size_t best_inl_sz = arg.N;
    size_t curr_inl_sz = arg.N;

    /* We compute offsets */
    size_t total = 0;
#define MK_OFFSET(v)                                     \
    size_t v ## _of = total;                             \
    total = alignSize(v ## _of  +  v ## _sz, MEM_ALIGN)

    MK_OFFSET(best_inl);
    MK_OFFSET(curr_inl);

#undef MK_OFFSET

    /* Allocate dynamic memory managed by cv::Mat */
    mem.perRun.create(1, (int)(total + MEM_ALIGN), CV_8UC1);

    /* Extract aligned pointer */
    unsigned char* ptr = alignPtr(mem.perRun.data, MEM_ALIGN);

    /* Assign pointers */
    best.inl  = (char*)(ptr + best_inl_of);
    curr.inl  = (char*)(ptr + curr_inl_of);
}


/**
 * Deallocate per-run dynamic storage.
 *
 * Undoes the work by allocatePerRun().
 */

inline void   RHO_HEST_REFC::deallocatePerRun(void){
    best.inl  = NULL;
    curr.inl  = NULL;

    mem.perRun.release();
}


/**
 * Deallocate per-object dynamic storage.
 *
 * Undoes the work by allocatePerObj().
 */

inline void   RHO_HEST_REFC::deallocatePerObj(void){
    ctrl.smpl   = NULL;
    curr.pkdPts = NULL;
    curr.H      = NULL;
    best.H      = NULL;
    lm.JtJ      = NULL;
    lm.tmp1     = NULL;
    lm.Jte      = NULL;

    mem.perObj.release();
}


/**
 * Initialize SAC for a run given its arguments.
 *
 * Performs sanity-checks and memory allocations. Also initializes the state.
 *
 * @returns 0 if per-run initialization failed at any point; non-zero
 *          otherwise.
 *
 * Reads:  arg.*, nr.*
 * Writes: curr.*, best.*, ctrl.*, eval.*
 */

inline int    RHO_HEST_REFC::initRun(void){
    /**
     * Sanitize arguments.
     *
     * Runs zeroth because these are easy-to-check errors and unambiguously
     * mean something or other.
     */

    if(!arg.src || !arg.dst){
        /* Arguments src or dst are insane, must be != NULL */
        return 0;
    }
    if(arg.N < (unsigned)SMPL_SIZE){
        /* Argument N is insane, must be >= 4. */
        return 0;
    }
    if(arg.maxD < 0){
        /* Argument maxD is insane, must be >= 0. */
        return 0;
    }
    if(arg.cfd < 0 || arg.cfd > 1){
        /* Argument cfd is insane, must be in [0, 1]. */
        return 0;
    }
    /* Clamp minInl to 4 or higher. */
    arg.minInl = arg.minInl < (unsigned)SMPL_SIZE ? SMPL_SIZE : arg.minInl;
    if(isNREnabled() && (arg.beta <= 0 || arg.beta >= 1)){
        /* Argument beta is insane, must be in (0, 1). */
        return 0;
    }
    if(!arg.finalH){
        /* Argument finalH is insane, must be != NULL */
        return 0;
    }

    /**
     * Optional NR setup.
     *
     * Runs first because it is decoupled from most other things (*) and if it
     * fails, it is easy to recover from.
     *
     * (*) The only things this code depends on is the flags argument, the nr.*
     *     substruct and the sanity-checked N and beta arguments from above.
     */

    if(isNREnabled() && !ensureCapacity(arg.N, arg.beta)){
        return 0;
    }

    /**
     * Inlier mask alloc.
     *
     * Runs second because we want to quit as fast as possible if we can't even
     * allocate the two masks.
     */

    allocatePerRun();

    memset(best.inl, 0, arg.N);
    memset(curr.inl, 0, arg.N);

    /**
     * Reset scalar per-run state.
     *
     * Runs third because there's no point in resetting/calculating a large
     * number of fields if something in the above junk failed.
     */

    ctrl.i            = 0;
    ctrl.phNum        = SMPL_SIZE;
    ctrl.phEndI       = 1;
    ctrl.phEndFpI     = sacInitPEndFpI(arg.rConvg, arg.N, SMPL_SIZE);
    ctrl.phMax        = arg.N;
    ctrl.phNumInl     = 0;
    ctrl.numModels    = 0;

    if(haveExtrinsicGuess()){
        memcpy(curr.H, arg.guessH, HSIZE);
    }else{
        memset(curr.H, 0, HSIZE);
    }
    curr.numInl       = 0;

    memset(best.H, 0, HSIZE);
    best.numInl       = 0;

    eval.Ntested      = 0;
    eval.Ntestedtotal = 0;
    eval.good         = 1;
    eval.t_M          = SPRT_T_M;
    eval.m_S          = SPRT_M_S;
    eval.epsilon      = SPRT_EPSILON;
    eval.delta        = SPRT_DELTA;
    designSPRTTest();

    return 1;
}

/**
 * Finalize SAC run.
 *
 * Deallocates per-run allocatable resources. Currently this consists only of
 * the best and current inlier masks, which are equal in size to p->arg.N
 * bytes.
 *
 * Reads:  arg.bestInl, curr.inl, best.inl
 * Writes: curr.inl, best.inl
 */

inline void   RHO_HEST_REFC::finiRun(void){
    deallocatePerRun();
}

/**
 * Hypothesize a model.
 *
 * Selects randomly a sample (within the rules of PROSAC) and generates a
 * new current model, and applies degeneracy tests to it.
 *
 * @returns 0 if hypothesized model could be rejected early as degenerate, and
 * non-zero otherwise.
 */

inline int    RHO_HEST_REFC::hypothesize(void){
    if(PROSACPhaseEndReached()){
        PROSACGoToNextPhase();
    }

    getPROSACSample();
    if(isSampleDegenerate()){
        return 0;
    }

    generateModel();
    if(isModelDegenerate()){
        return 0;
    }

    return 1;
}

/**
 * Verify the hypothesized model.
 *
 * Given the current model, evaluate its quality. If it is better than
 * everything before, save as new best model (and possibly refine it).
 *
 * Returns 1.
 */

inline int    RHO_HEST_REFC::verify(void){
    evaluateModelSPRT();
    updateSPRT();

    if(isBestModel()){
        saveBestModel();

        if(isRefineEnabled() && canRefine()){
            refine();
        }

        updateBounds();

        if(isNREnabled()){
            nStarOptimize();
        }
    }

    return 1;
}

/**
 * Check whether extrinsic guess was provided or not.
 *
 * @return Zero if no extrinsic guess was provided; non-zero otherwiseEE.
 */

inline int    RHO_HEST_REFC::haveExtrinsicGuess(void){
    return !!arg.guessH;
}

/**
 * Check whether non-randomness criterion is enabled.
 *
 * @return Zero if non-randomness criterion disabled; non-zero if not.
 */

inline int    RHO_HEST_REFC::isNREnabled(void){
    return arg.flags & RHO_FLAG_ENABLE_NR;
}

/**
 * Check whether best-model-so-far refinement is enabled.
 *
 * @return Zero if best-model-so-far refinement disabled; non-zero if not.
 */

inline int    RHO_HEST_REFC::isRefineEnabled(void){
    return arg.flags & RHO_FLAG_ENABLE_REFINEMENT;
}

/**
 * Check whether final-model refinement is enabled.
 *
 * @return Zero if final-model refinement disabled; non-zero if not.
 */

inline int    RHO_HEST_REFC::isFinalRefineEnabled(void){
    return arg.flags & RHO_FLAG_ENABLE_FINAL_REFINEMENT;
}

/**
 * Computes whether the end of the current PROSAC phase has been reached. At
 * PROSAC phase phNum, only matches [0, phNum) are sampled from.
 *
 * Reads    (direct): ctrl.i, ctrl.phEndI, ctrl.phNum, ctrl.phMax
 * Reads   (callees): None.
 * Writes   (direct): None.
 * Writes  (callees): None.
 */

inline int    RHO_HEST_REFC::PROSACPhaseEndReached(void){
    return ctrl.i >= ctrl.phEndI && ctrl.phNum < ctrl.phMax;
}

/**
 * Updates unconditionally the necessary fields to move to the next PROSAC
 * stage.
 *
 * Not idempotent.
 *
 * Reads    (direct): ctrl.phNum, ctrl.phEndFpI, ctrl.phEndI
 * Reads   (callees): None.
 * Writes   (direct): ctrl.phNum, ctrl.phEndFpI, ctrl.phEndI
 * Writes  (callees): None.
 */

inline void   RHO_HEST_REFC::PROSACGoToNextPhase(void){
    double next;

    ctrl.phNum++;
    next = (ctrl.phEndFpI * ctrl.phNum)/(ctrl.phNum - SMPL_SIZE);
    ctrl.phEndI  += (unsigned)ceil(next - ctrl.phEndFpI);
    ctrl.phEndFpI = next;
}

/**
 * Get a sample according to PROSAC rules. Namely:
 * - If we're past the phase end interation, select randomly 4 out of the first
 *   phNum matches.
 * - Otherwise, select match phNum-1 and select randomly the 3 others out of
 *   the first phNum-1 matches.
 *
 * Reads    (direct): ctrl.i, ctrl.phEndI, ctrl.phNum
 * Reads   (callees): prng.s
 * Writes   (direct): ctrl.smpl
 * Writes  (callees): prng.s
 */

inline void   RHO_HEST_REFC::getPROSACSample(void){
    if(ctrl.i > ctrl.phEndI){
        /* FIXME: Dubious. Review. */
        rndSmpl(4, ctrl.smpl, ctrl.phNum);/* Used to be phMax */
    }else{
        rndSmpl(3, ctrl.smpl, ctrl.phNum-1);
        ctrl.smpl[3] = ctrl.phNum-1;
    }
}

/**
 * Choose, without repetition, sampleSize integers in the range [0, numDataPoints).
 *
 * Reads    (direct): None.
 * Reads   (callees): prng.s
 * Writes   (direct): None.
 * Writes  (callees): prng.s
 */

inline void   RHO_HEST_REFC::rndSmpl(unsigned  sampleSize,
                                     unsigned* currentSample,
                                     unsigned  dataSetSize){
    /**
     * If sampleSize is very close to dataSetSize, we use selection sampling.
     * Otherwise we use the naive sampling technique wherein we select random
     * indexes until sampleSize of them are distinct.
     */

    if(sampleSize*2>dataSetSize){
        /**
         * Selection Sampling:
         *
         * Algorithm S (Selection sampling technique). To select n records at random from a set of N, where 0 < n ≤ N.
         * S1. [Initialize.] Set t ← 0, m ← 0. (During this algorithm, m represents the number of records selected so far,
         *                                      and t is the total number of input records that we have dealt with.)
         * S2. [Generate U.] Generate a random number U, uniformly distributed between zero and one.
         * S3. [Test.] If (N – t)U ≥ n – m, go to step S5.
         * S4. [Select.] Select the next record for the sample, and increase m and t by 1. If m < n, go to step S2;
         *               otherwise the sample is complete and the algorithm terminates.
         * S5. [Skip.] Skip the next record (do not include it in the sample), increase t by 1, and go back to step S2.
         *
         * Replaced m with i and t with j in the below code.
         */

        unsigned i=0,j=0;

        for(i=0;i<sampleSize;j++){
            double U=fastRandom();
            if((dataSetSize-j)*U < (sampleSize-i)){
                currentSample[i++]=j;
            }
        }
    }else{
        /**
         * Naive sampling technique. Generate indexes until sampleSize of them are distinct.
         */

        unsigned i, j;
        for(i=0;i<sampleSize;i++){
            int inList;

            do{
                currentSample[i] = (unsigned)(dataSetSize*fastRandom());

                inList=0;
                for(j=0;j<i;j++){
                    if(currentSample[i] == currentSample[j]){
                        inList=1;
                        break;
                    }
                }
            }while(inList);
        }
    }
}

/**
 * Checks whether the *sample* is degenerate prior to model generation.
 * - First, the extremely cheap numerical degeneracy test is run, which weeds
 *   out bad samples to the optimized GE implementation.
 * - Second, the geometrical degeneracy test is run, which weeds out most other
 *   bad samples.
 *
 * Reads    (direct): ctrl.smpl, arg.src, arg.dst
 * Reads   (callees): None.
 * Writes   (direct): curr.pkdPts
 * Writes  (callees): None.
 */

inline int    RHO_HEST_REFC::isSampleDegenerate(void){
    unsigned i0 = ctrl.smpl[0], i1 = ctrl.smpl[1], i2 = ctrl.smpl[2], i3 = ctrl.smpl[3];
    typedef struct{float x,y;} MyPt2f;
    MyPt2f* pkdPts = (MyPt2f*)curr.pkdPts, *src = (MyPt2f*)arg.src, *dst = (MyPt2f*)arg.dst;

    /**
     * Pack the matches selected by the SAC algorithm.
     * Must be packed  points[0:7]  = {srcx0, srcy0, srcx1, srcy1, srcx2, srcy2, srcx3, srcy3}
     *                 points[8:15] = {dstx0, dsty0, dstx1, dsty1, dstx2, dsty2, dstx3, dsty3}
     * Gather 4 points into the vector
     */

    pkdPts[0] = src[i0];
    pkdPts[1] = src[i1];
    pkdPts[2] = src[i2];
    pkdPts[3] = src[i3];
    pkdPts[4] = dst[i0];
    pkdPts[5] = dst[i1];
    pkdPts[6] = dst[i2];
    pkdPts[7] = dst[i3];

    /**
     * If the matches' source points have common x and y coordinates, abort.
     */

    if(pkdPts[0].x == pkdPts[1].x || pkdPts[1].x == pkdPts[2].x ||
       pkdPts[2].x == pkdPts[3].x || pkdPts[0].x == pkdPts[2].x ||
       pkdPts[1].x == pkdPts[3].x || pkdPts[0].x == pkdPts[3].x ||
       pkdPts[0].y == pkdPts[1].y || pkdPts[1].y == pkdPts[2].y ||
       pkdPts[2].y == pkdPts[3].y || pkdPts[0].y == pkdPts[2].y ||
       pkdPts[1].y == pkdPts[3].y || pkdPts[0].y == pkdPts[3].y){
        return 1;
    }

    /* If the matches do not satisfy the strong geometric constraint, abort. */
    /* (0 x 1) * 2 */
    float cross0s0 = pkdPts[0].y-pkdPts[1].y;
    float cross0s1 = pkdPts[1].x-pkdPts[0].x;
    float cross0s2 = pkdPts[0].x*pkdPts[1].y-pkdPts[0].y*pkdPts[1].x;
    float dots0    = cross0s0*pkdPts[2].x + cross0s1*pkdPts[2].y + cross0s2;
    float cross0d0 = pkdPts[4].y-pkdPts[5].y;
    float cross0d1 = pkdPts[5].x-pkdPts[4].x;
    float cross0d2 = pkdPts[4].x*pkdPts[5].y-pkdPts[4].y*pkdPts[5].x;
    float dotd0    = cross0d0*pkdPts[6].x + cross0d1*pkdPts[6].y + cross0d2;
    if(((int)dots0^(int)dotd0) < 0){
        return 1;
    }
    /* (0 x 1) * 3 */
    float cross1s0 = cross0s0;
    float cross1s1 = cross0s1;
    float cross1s2 = cross0s2;
    float dots1    = cross1s0*pkdPts[3].x + cross1s1*pkdPts[3].y + cross1s2;
    float cross1d0 = cross0d0;
    float cross1d1 = cross0d1;
    float cross1d2 = cross0d2;
    float dotd1    = cross1d0*pkdPts[7].x + cross1d1*pkdPts[7].y + cross1d2;
    if(((int)dots1^(int)dotd1) < 0){
        return 1;
    }
    /* (2 x 3) * 0 */
    float cross2s0 = pkdPts[2].y-pkdPts[3].y;
    float cross2s1 = pkdPts[3].x-pkdPts[2].x;
    float cross2s2 = pkdPts[2].x*pkdPts[3].y-pkdPts[2].y*pkdPts[3].x;
    float dots2    = cross2s0*pkdPts[0].x + cross2s1*pkdPts[0].y + cross2s2;
    float cross2d0 = pkdPts[6].y-pkdPts[7].y;
    float cross2d1 = pkdPts[7].x-pkdPts[6].x;
    float cross2d2 = pkdPts[6].x*pkdPts[7].y-pkdPts[6].y*pkdPts[7].x;
    float dotd2    = cross2d0*pkdPts[4].x + cross2d1*pkdPts[4].y + cross2d2;
    if(((int)dots2^(int)dotd2) < 0){
        return 1;
    }
    /* (2 x 3) * 1 */
    float cross3s0 = cross2s0;
    float cross3s1 = cross2s1;
    float cross3s2 = cross2s2;
    float dots3    = cross3s0*pkdPts[1].x + cross3s1*pkdPts[1].y + cross3s2;
    float cross3d0 = cross2d0;
    float cross3d1 = cross2d1;
    float cross3d2 = cross2d2;
    float dotd3    = cross3d0*pkdPts[5].x + cross3d1*pkdPts[5].y + cross3d2;
    if(((int)dots3^(int)dotd3) < 0){
        return 1;
    }

    /* Otherwise, accept */
    return 0;
}

/**
 * Compute homography of matches in gathered, packed sample and output the
 * current homography.
 *
 * Reads    (direct): None.
 * Reads   (callees): curr.pkdPts
 * Writes   (direct): None.
 * Writes  (callees): curr.H
 */

inline void   RHO_HEST_REFC::generateModel(void){
    hFuncRefC(curr.pkdPts, curr.H);
}

/**
 * Checks whether the model is itself degenerate.
 * - One test: All elements of the homography are added, and if the result is
 *   NaN the homography is rejected.
 *
 * Reads    (direct): curr.H
 * Reads   (callees): None.
 * Writes   (direct): None.
 * Writes  (callees): None.
 */

inline int    RHO_HEST_REFC::isModelDegenerate(void){
    int degenerate;
    float* H = curr.H;
    float f=H[0]+H[1]+H[2]+H[3]+H[4]+H[5]+H[6]+H[7];

    /* degenerate = isnan(f); */
    /* degenerate = f!=f;// Only NaN is not equal to itself. */
    degenerate = cvIsNaN(f);
    /* degenerate = 0; */


    return degenerate;
}

/**
 * Evaluates the current model using SPRT for early exiting.
 *
 * Reads    (direct): arg.maxD, arg.src, arg.dst, arg.N, curr.inl, curr.H,
 *                    ctrl.numModels, eval.Ntestedtotal, eval.lambdaAccept,
 *                    eval.lambdaReject, eval.A
 * Reads   (callees): None.
 * Writes   (direct): ctrl.numModels, curr.numInl, eval.Ntested, eval.good,
 *                    eval.Ntestedtotal
 * Writes  (callees): None.
 */

inline void   RHO_HEST_REFC::evaluateModelSPRT(void){
    unsigned i;
    unsigned isInlier;
    double   lambda  = 1.0;
    float    distSq  = arg.maxD*arg.maxD;
    const float* src = arg.src;
    const float* dst = arg.dst;
    char*    inl     = curr.inl;
    const float*   H = curr.H;


    ctrl.numModels++;

    curr.numInl   = 0;
    eval.Ntested  = 0;
    eval.good     = 1;


    /* SCALAR */
    for(i=0;i<arg.N && eval.good;i++){
        /* Backproject */
        float x=src[i*2],y=src[i*2+1];
        float X=dst[i*2],Y=dst[i*2+1];

        float reprojX=H[0]*x+H[1]*y+H[2]; /*  ( X_1 )     ( H_11 H_12    H_13  ) (x_1)       */
        float reprojY=H[3]*x+H[4]*y+H[5]; /*  ( X_2 )  =  ( H_21 H_22    H_23  ) (x_2)       */
        float reprojZ=H[6]*x+H[7]*y+1.0f; /*  ( X_3 )     ( H_31 H_32 H_33=1.0 ) (x_3 = 1.0) */

        /* reproj is in homogeneous coordinates. To bring back to "regular" coordinates, divide by Z. */
        reprojX/=reprojZ;
        reprojY/=reprojZ;

        /* Compute distance */
        reprojX-=X;
        reprojY-=Y;
        reprojX*=reprojX;
        reprojY*=reprojY;
        float reprojDist = reprojX+reprojY;

        /* ... */
        isInlier   = reprojDist <= distSq;
        curr.numInl += isInlier;
        *inl++     = (char)isInlier;


        /* SPRT */
        lambda *= isInlier ? eval.lambdaAccept : eval.lambdaReject;
        eval.good = lambda <= eval.A;
        /* If !good, the threshold A was exceeded, so we're rejecting */
    }


    eval.Ntested       = i;
    eval.Ntestedtotal += i;
}

/**
 * Update either the delta or epsilon SPRT parameters, depending on the events
 * that transpired in the previous evaluation.
 *
 * Reads    (direct): eval.good, curr.numInl, arg.N, eval.Ntested, eval.delta
 * Reads   (callees): eval.delta, eval.epsilon, eval.t_M, eval.m_S
 * Writes   (direct): eval.epsilon, eval.delta
 * Writes  (callees): eval.A, eval.lambdaReject, eval.lambdaAccept
 */

inline void   RHO_HEST_REFC::updateSPRT(void){
    if(eval.good){
        if(isBestModel()){
            eval.epsilon = (double)curr.numInl/arg.N;
            designSPRTTest();
        }
    }else{
        double newDelta = (double)curr.numInl/eval.Ntested;

        if(newDelta > 0){
            double relChange = fabs(eval.delta - newDelta)/ eval.delta;
            if(relChange > MIN_DELTA_CHNG){
                eval.delta = newDelta;
                designSPRTTest();
            }
        }
    }
}

/**
 * Numerically compute threshold A from the estimated delta, epsilon, t_M and
 * m_S values.
 *
 * Epsilon:  Denotes the probability that a randomly chosen data point is
 *           consistent with a good model.
 * Delta:    Denotes the probability that a randomly chosen data point is
 *           consistent with a bad model.
 * t_M:      Time needed to instantiate a model hypotheses given a sample.
 *           (Computing model parameters from a sample takes the same time
 *            as verification of t_M data points)
 * m_S:      The number of models that are verified per sample.
 */

static inline double sacDesignSPRTTest(double delta, double epsilon, double t_M, double m_S){
    double An, C, K, prevAn;
    unsigned i;

    /**
     * Randomized RANSAC with Sequential Probability Ratio Test, ICCV 2005
     * Eq (2)
     */

    C = (1-delta)  *  log((1-delta)/(1-epsilon)) +
        delta      *  log(  delta  /  epsilon  );

    /**
     * Randomized RANSAC with Sequential Probability Ratio Test, ICCV 2005
     * Eq (6)
     * K = K_1/K_2 + 1 = (t_M*C)/m_S + 1
     */

    K = t_M*C/m_S + 1;

    /**
     * Randomized RANSAC with Sequential Probability Ratio Test, ICCV 2005
     * Paragraph below Eq (6)
     *
     * A* = lim_{n -> infty} A_n, where
     *     A_0     = K1/K2 + 1             and
     *     A_{n+1} = K1/K2 + 1 + log(A_n)
     * The series converges fast, typically within four iterations.
     */

    An = K;
    i  = 0;

    do{
        prevAn = An;
        An = K + log(An);
    }while((An-prevAn > 1.5e-8)  &&  (++i < 10));

    /**
     * Return A = An_stopping, with n_stopping < 10
     */

    return An;
}

/**
 * Design the SPRT test. Shorthand for
 *     A = sprt(delta, epsilon, t_M, m_S);
 *
 * Idempotent.
 *
 * Reads    (direct): eval.delta, eval.epsilon, eval.t_M, eval.m_S
 * Reads   (callees): None.
 * Writes   (direct): eval.A, eval.lambdaReject, eval.lambdaAccept.
 * Writes  (callees): None.
 */

inline void   RHO_HEST_REFC::designSPRTTest(void){
    eval.A = sacDesignSPRTTest(eval.delta, eval.epsilon, eval.t_M, eval.m_S);
    eval.lambdaReject = ((1.0 - eval.delta) / (1.0 - eval.epsilon));
    eval.lambdaAccept = ((   eval.delta   ) / (    eval.epsilon  ));
}

/**
 * Return whether the current model is the best model so far.
 *
 * @return Non-zero if this is the model with the most inliers seen so far;
 *         0 otherwise.
 *
 * Reads    (direct): curr.numInl, best.numInl
 * Reads   (callees): None.
 * Writes   (direct): None.
 * Writes  (callees): None.
 */

inline int    RHO_HEST_REFC::isBestModel(void){
    return curr.numInl > best.numInl;
}

/**
 * Returns whether the current-best model is good enough to be an
 * acceptable best model, by checking whether it meets the minimum
 * number of inliers.
 *
 * @return Non-zero if the current model is "good enough" to save;
 *         0 otherwise.
 *
 * Reads    (direct): best.numInl, arg.minInl
 * Reads   (callees): None.
 * Writes   (direct): None.
 * Writes  (callees): None.
 */

inline int    RHO_HEST_REFC::isBestModelGoodEnough(void){
    return best.numInl >= arg.minInl;
}

/**
 * Make current model new best model by swapping the homography, inlier mask
 * and count of inliers between the current and best models.
 *
 * Reads    (direct): curr.H, curr.inl, curr.numInl,
 *                    best.H, best.inl, best.numInl
 * Reads   (callees): None.
 * Writes   (direct): curr.H, curr.inl, curr.numInl,
 *                    best.H, best.inl, best.numInl
 * Writes  (callees): None.
 */

inline void   RHO_HEST_REFC::saveBestModel(void){
    float*   H      = curr.H;
    char*    inl    = curr.inl;
    unsigned numInl = curr.numInl;

    curr.H       = best.H;
    curr.inl     = best.inl;
    curr.numInl  = best.numInl;

    best.H       = H;
    best.inl     = inl;
    best.numInl  = numInl;
}

/**
 * Compute NR table entries [start, N) for given beta.
 */

static inline void   sacInitNonRand(double    beta,
                                    unsigned  start,
                                    unsigned  N,
                                    unsigned* nonRandMinInl){
    unsigned n = SMPL_SIZE+1 > start ? SMPL_SIZE+1 : start;
    double   beta_beta1_sq_chi = sqrt(beta*(1.0-beta)) * CHI_SQ;

    for(; n < N; n++){
        double   mu      = n * beta;
        double   sigma   = sqrt((double)n)* beta_beta1_sq_chi;
        unsigned i_min   = (unsigned)ceil(SMPL_SIZE + mu + sigma);

        nonRandMinInl[n] = i_min;
    }
}

/**
 * Optimize the stopping criterion to account for the non-randomness criterion
 * of PROSAC.
 *
 * Reads    (direct): arg.N, best.numInl, nr.tbl, arg.inl, ctrl.phMax,
 *                    ctrl.phNumInl, arg.cfd, arg.maxI
 * Reads   (callees): None.
 * Writes   (direct): arg.maxI, ctrl.phMax, ctrl.phNumInl
 * Writes  (callees): None.
 */

inline void   RHO_HEST_REFC::nStarOptimize(void){
    unsigned min_sample_length = 10*2; /*(N * INLIERS_RATIO) */
    unsigned best_n       = arg.N;
    unsigned test_n       = best_n;
    unsigned bestNumInl   = best.numInl;
    unsigned testNumInl   = bestNumInl;

    for(;test_n > min_sample_length && testNumInl;test_n--){
        if(testNumInl*best_n > bestNumInl*test_n){
            if(testNumInl < nr.tbl[test_n]){
                break;
            }
            best_n      = test_n;
            bestNumInl  = testNumInl;
        }
        testNumInl -= !!best.inl[test_n-1];
    }

    if(bestNumInl*ctrl.phMax > ctrl.phNumInl*best_n){
        ctrl.phMax    = best_n;
        ctrl.phNumInl = bestNumInl;
        arg.maxI      = sacCalcIterBound(arg.cfd,
                                         (double)ctrl.phNumInl/ctrl.phMax,
                                         SMPL_SIZE,
                                         arg.maxI);
    }
}

/**
 * Classic RANSAC iteration bound based on largest # of inliers.
 *
 * Reads    (direct): arg.maxI, arg.cfd, best.numInl, arg.N
 * Reads   (callees): None.
 * Writes   (direct): arg.maxI
 * Writes  (callees): None.
 */

inline void   RHO_HEST_REFC::updateBounds(void){
    arg.maxI = sacCalcIterBound(arg.cfd,
                                (double)best.numInl/arg.N,
                                SMPL_SIZE,
                                arg.maxI);
}

/**
 * Ouput the best model so far to the output argument.
 *
 * Reads    (direct): arg.finalH, best.H, arg.inl, best.inl, arg.N
 * Reads   (callees): arg.finalH, arg.inl, arg.N
 * Writes   (direct): arg.finalH, arg.inl
 * Writes  (callees): arg.finalH, arg.inl
 */

inline void   RHO_HEST_REFC::outputModel(void){
    if(isBestModelGoodEnough()){
        memcpy(arg.finalH, best.H, HSIZE);
        if(arg.inl){
            memcpy(arg.inl, best.inl, arg.N);
        }
    }else{
        outputZeroH();
    }
}

/**
 * Ouput a zeroed H to the output argument.
 *
 * Reads    (direct): arg.finalH, arg.inl, arg.N
 * Reads   (callees): None.
 * Writes   (direct): arg.finalH, arg.inl
 * Writes  (callees): None.
 */

inline void   RHO_HEST_REFC::outputZeroH(void){
    if(arg.finalH){
        memset(arg.finalH, 0, HSIZE);
    }
    if(arg.inl){
        memset(arg.inl,    0, arg.N);
    }
}

/**
 * Compute the real-valued number of samples per phase, given the RANSAC convergence speed,
 * data set size and sample size.
 */

static inline double sacInitPEndFpI(const unsigned ransacConvg,
                                    const unsigned n,
                                    const unsigned s){
    double numer=1, denom=1;

    unsigned i;
    for(i=0;i<s;i++){
        numer *= s-i;
        denom *= n-i;
    }

    return ransacConvg*numer/denom;
}

/**
 * Estimate the number of iterations required based on the requested confidence,
 * proportion of inliers in the best model so far and sample size.
 *
 * Clamp return value at maxIterationBound.
 */

static inline unsigned sacCalcIterBound(double   confidence,
                                        double   inlierRate,
                                        unsigned sampleSize,
                                        unsigned maxIterBound){
    unsigned retVal;

    /**
     * Formula chosen from http://en.wikipedia.org/wiki/RANSAC#The_parameters :
     *
     * \[ k = \frac{\log{(1-confidence)}}{\log{(1-inlierRate**sampleSize)}} \]
     */

    double atLeastOneOutlierProbability = 1.-pow(inlierRate, (double)sampleSize);

    /**
     * There are two special cases: When argument to log() is 0 and when it is 1.
     * Each has a special meaning.
     */

    if(atLeastOneOutlierProbability>=1.){
        /**
         * A certainty of picking at least one outlier means that we will need
         * an infinite amount of iterations in order to find a correct solution.
         */

        retVal = maxIterBound;
    }else if(atLeastOneOutlierProbability<=0.){
        /**
         * The certainty of NOT picking an outlier means that only 1 iteration
         * is needed to find a solution.
         */

        retVal = 1;
    }else{
        /**
         * Since 1-confidence (the probability of the model being based on at
         * least one outlier in the data) is equal to
         * (1-inlierRate**sampleSize)**numIterations (the probability of picking
         * at least one outlier in numIterations samples), we can isolate
         * numIterations (the return value) into
         */

        retVal = (unsigned)ceil(log(1.-confidence)/log(atLeastOneOutlierProbability));
    }

    /**
     * Clamp to maxIterationBound.
     */

    return retVal <= maxIterBound ? retVal : maxIterBound;
}


/**
 * Given 4 matches, computes the homography that relates them using Gaussian
 * Elimination. The row operations are as given in the paper.
 *
 * TODO: Clean this up. The code is hideous, and might even conceal sign bugs
 *       (specifically relating to whether the last column should be negated,
 *        or not).
 */

static void hFuncRefC(float* packedPoints,/* Source (four x,y float coordinates) points followed by
                                             destination (four x,y float coordinates) points, aligned by 32 bytes */
                      float* H){          /* Homography (three 16-byte aligned rows of 3 floats) */
    float x0=*packedPoints++;
    float y0=*packedPoints++;
    float x1=*packedPoints++;
    float y1=*packedPoints++;
    float x2=*packedPoints++;
    float y2=*packedPoints++;
    float x3=*packedPoints++;
    float y3=*packedPoints++;
    float X0=*packedPoints++;
    float Y0=*packedPoints++;
    float X1=*packedPoints++;
    float Y1=*packedPoints++;
    float X2=*packedPoints++;
    float Y2=*packedPoints++;
    float X3=*packedPoints++;
    float Y3=*packedPoints++;

    float x0X0=x0*X0, x1X1=x1*X1, x2X2=x2*X2, x3X3=x3*X3;
    float x0Y0=x0*Y0, x1Y1=x1*Y1, x2Y2=x2*Y2, x3Y3=x3*Y3;
    float y0X0=y0*X0, y1X1=y1*X1, y2X2=y2*X2, y3X3=y3*X3;
    float y0Y0=y0*Y0, y1Y1=y1*Y1, y2Y2=y2*Y2, y3Y3=y3*Y3;


    /**
     *  [0]   [1] Hidden   Prec
     *  x0    y0    1       x1
     *  x1    y1    1       x1
     *  x2    y2    1       x1
     *  x3    y3    1       x1
     *
     * Eliminate ones in column 2 and 5.
     * R(0)-=R(2)
     * R(1)-=R(2)
     * R(3)-=R(2)
     *
     *  [0]   [1] Hidden   Prec
     * x0-x2 y0-y2  0       x1+1
     * x1-x2 y1-y2  0       x1+1
     *  x2    y2    1       x1
     * x3-x2 y3-y2  0       x1+1
     *
     * Eliminate column 0 of rows 1 and 3
     * R(1)=(x0-x2)*R(1)-(x1-x2)*R(0),     y1'=(y1-y2)(x0-x2)-(x1-x2)(y0-y2)
     * R(3)=(x0-x2)*R(3)-(x3-x2)*R(0),     y3'=(y3-y2)(x0-x2)-(x3-x2)(y0-y2)
     *
     *  [0]   [1] Hidden   Prec
     * x0-x2 y0-y2  0      x1+1
     *   0    y1'   0      x2+3
     *  x2    y2    1       x1
     *   0    y3'   0      x2+3
     *
     * Eliminate column 1 of rows 0 and 3
     * R(3)=y1'*R(3)-y3'*R(1)
     * R(0)=y1'*R(0)-(y0-y2)*R(1)
     *
     *  [0]   [1] Hidden   Prec
     *  x0'    0    0      x3+5
     *   0    y1'   0      x2+3
     *  x2    y2    1       x1
     *   0     0    0      x4+7
     *
     * Eliminate columns 0 and 1 of row 2
     * R(0)/=x0'
     * R(1)/=y1'
     * R(2)-= (x2*R(0) + y2*R(1))
     *
     *  [0]   [1] Hidden   Prec
     *   1     0    0      x6+10
     *   0     1    0      x4+6
     *   0     0    1      x4+7
     *   0     0    0      x4+7
     */

    /**
     * Eliminate ones in column 2 and 5.
     * R(0)-=R(2)
     * R(1)-=R(2)
     * R(3)-=R(2)
     */

    /*float minor[4][2] = {{x0-x2,y0-y2},
                         {x1-x2,y1-y2},
                         {x2   ,y2   },
                         {x3-x2,y3-y2}};*/
    /*float major[8][3] = {{x2X2-x0X0,y2X2-y0X0,(X0-X2)},
                         {x2X2-x1X1,y2X2-y1X1,(X1-X2)},
                         {-x2X2    ,-y2X2    ,(X2   )},
                         {x2X2-x3X3,y2X2-y3X3,(X3-X2)},
                         {x2Y2-x0Y0,y2Y2-y0Y0,(Y0-Y2)},
                         {x2Y2-x1Y1,y2Y2-y1Y1,(Y1-Y2)},
                         {-x2Y2    ,-y2Y2    ,(Y2   )},
                         {x2Y2-x3Y3,y2Y2-y3Y3,(Y3-Y2)}};*/
    float minor[2][4] = {{x0-x2,x1-x2,x2   ,x3-x2},
                         {y0-y2,y1-y2,y2   ,y3-y2}};
    float major[3][8] = {{x2X2-x0X0,x2X2-x1X1,-x2X2    ,x2X2-x3X3,x2Y2-x0Y0,x2Y2-x1Y1,-x2Y2    ,x2Y2-x3Y3},
                         {y2X2-y0X0,y2X2-y1X1,-y2X2    ,y2X2-y3X3,y2Y2-y0Y0,y2Y2-y1Y1,-y2Y2    ,y2Y2-y3Y3},
                         { (X0-X2) , (X1-X2) , (X2   ) , (X3-X2) , (Y0-Y2) , (Y1-Y2) , (Y2   ) , (Y3-Y2) }};

    /**
     * int i;
     * for(i=0;i<8;i++) major[2][i]=-major[2][i];
     * Eliminate column 0 of rows 1 and 3
     * R(1)=(x0-x2)*R(1)-(x1-x2)*R(0),     y1'=(y1-y2)(x0-x2)-(x1-x2)(y0-y2)
     * R(3)=(x0-x2)*R(3)-(x3-x2)*R(0),     y3'=(y3-y2)(x0-x2)-(x3-x2)(y0-y2)
     */

    float scalar1=minor[0][0], scalar2=minor[0][1];
    minor[1][1]=minor[1][1]*scalar1-minor[1][0]*scalar2;

    major[0][1]=major[0][1]*scalar1-major[0][0]*scalar2;
    major[1][1]=major[1][1]*scalar1-major[1][0]*scalar2;
    major[2][1]=major[2][1]*scalar1-major[2][0]*scalar2;

    major[0][5]=major[0][5]*scalar1-major[0][4]*scalar2;
    major[1][5]=major[1][5]*scalar1-major[1][4]*scalar2;
    major[2][5]=major[2][5]*scalar1-major[2][4]*scalar2;

    scalar2=minor[0][3];
    minor[1][3]=minor[1][3]*scalar1-minor[1][0]*scalar2;

    major[0][3]=major[0][3]*scalar1-major[0][0]*scalar2;
    major[1][3]=major[1][3]*scalar1-major[1][0]*scalar2;
    major[2][3]=major[2][3]*scalar1-major[2][0]*scalar2;

    major[0][7]=major[0][7]*scalar1-major[0][4]*scalar2;
    major[1][7]=major[1][7]*scalar1-major[1][4]*scalar2;
    major[2][7]=major[2][7]*scalar1-major[2][4]*scalar2;

    /**
     * Eliminate column 1 of rows 0 and 3
     * R(3)=y1'*R(3)-y3'*R(1)
     * R(0)=y1'*R(0)-(y0-y2)*R(1)
     */

    scalar1=minor[1][1];scalar2=minor[1][3];
    major[0][3]=major[0][3]*scalar1-major[0][1]*scalar2;
    major[1][3]=major[1][3]*scalar1-major[1][1]*scalar2;
    major[2][3]=major[2][3]*scalar1-major[2][1]*scalar2;

    major[0][7]=major[0][7]*scalar1-major[0][5]*scalar2;
    major[1][7]=major[1][7]*scalar1-major[1][5]*scalar2;
    major[2][7]=major[2][7]*scalar1-major[2][5]*scalar2;

    scalar2=minor[1][0];
    minor[0][0]=minor[0][0]*scalar1-minor[0][1]*scalar2;

    major[0][0]=major[0][0]*scalar1-major[0][1]*scalar2;
    major[1][0]=major[1][0]*scalar1-major[1][1]*scalar2;
    major[2][0]=major[2][0]*scalar1-major[2][1]*scalar2;

    major[0][4]=major[0][4]*scalar1-major[0][5]*scalar2;
    major[1][4]=major[1][4]*scalar1-major[1][5]*scalar2;
    major[2][4]=major[2][4]*scalar1-major[2][5]*scalar2;

    /**
     * Eliminate columns 0 and 1 of row 2
     * R(0)/=x0'
     * R(1)/=y1'
     * R(2)-= (x2*R(0) + y2*R(1))
     */

    scalar1=1.0f/minor[0][0];
    major[0][0]*=scalar1;
    major[1][0]*=scalar1;
    major[2][0]*=scalar1;
    major[0][4]*=scalar1;
    major[1][4]*=scalar1;
    major[2][4]*=scalar1;

    scalar1=1.0f/minor[1][1];
    major[0][1]*=scalar1;
    major[1][1]*=scalar1;
    major[2][1]*=scalar1;
    major[0][5]*=scalar1;
    major[1][5]*=scalar1;
    major[2][5]*=scalar1;


    scalar1=minor[0][2];scalar2=minor[1][2];
    major[0][2]-=major[0][0]*scalar1+major[0][1]*scalar2;
    major[1][2]-=major[1][0]*scalar1+major[1][1]*scalar2;
    major[2][2]-=major[2][0]*scalar1+major[2][1]*scalar2;

    major[0][6]-=major[0][4]*scalar1+major[0][5]*scalar2;
    major[1][6]-=major[1][4]*scalar1+major[1][5]*scalar2;
    major[2][6]-=major[2][4]*scalar1+major[2][5]*scalar2;

    /* Only major matters now. R(3) and R(7) correspond to the hollowed-out rows. */
    scalar1=major[0][7];
    major[1][7]/=scalar1;
    major[2][7]/=scalar1;

    scalar1=major[0][0];major[1][0]-=scalar1*major[1][7];major[2][0]-=scalar1*major[2][7];
    scalar1=major[0][1];major[1][1]-=scalar1*major[1][7];major[2][1]-=scalar1*major[2][7];
    scalar1=major[0][2];major[1][2]-=scalar1*major[1][7];major[2][2]-=scalar1*major[2][7];
    scalar1=major[0][3];major[1][3]-=scalar1*major[1][7];major[2][3]-=scalar1*major[2][7];
    scalar1=major[0][4];major[1][4]-=scalar1*major[1][7];major[2][4]-=scalar1*major[2][7];
    scalar1=major[0][5];major[1][5]-=scalar1*major[1][7];major[2][5]-=scalar1*major[2][7];
    scalar1=major[0][6];major[1][6]-=scalar1*major[1][7];major[2][6]-=scalar1*major[2][7];


    /* One column left (Two in fact, but the last one is the homography) */
    scalar1=major[1][3];

    major[2][3]/=scalar1;
    scalar1=major[1][0];major[2][0]-=scalar1*major[2][3];
    scalar1=major[1][1];major[2][1]-=scalar1*major[2][3];
    scalar1=major[1][2];major[2][2]-=scalar1*major[2][3];
    scalar1=major[1][4];major[2][4]-=scalar1*major[2][3];
    scalar1=major[1][5];major[2][5]-=scalar1*major[2][3];
    scalar1=major[1][6];major[2][6]-=scalar1*major[2][3];
    scalar1=major[1][7];major[2][7]-=scalar1*major[2][3];


    /* Homography is done. */
    H[0]=major[2][0];
    H[1]=major[2][1];
    H[2]=major[2][2];

    H[3]=major[2][4];
    H[4]=major[2][5];
    H[5]=major[2][6];

    H[6]=major[2][7];
    H[7]=major[2][3];
    H[8]=1.0;
}


/**
 * Returns whether refinement is possible.
 *
 * NB This is separate from whether it is *enabled*.
 *
 * @return 0 if refinement isn't possible, non-zero otherwise.
 *
 * Reads    (direct): best.numInl
 * Reads   (callees): None.
 * Writes   (direct): None.
 * Writes  (callees): None.
 */

inline int    RHO_HEST_REFC::canRefine(void){
    /**
     * If we only have 4 matches, GE's result is already optimal and cannot
     * be refined any further.
     */

    return best.numInl > (unsigned)SMPL_SIZE;
}


/**
 * Refines the best-so-far homography (p->best.H).
 *
 * Reads    (direct): best.H, arg.src, arg.dst, best.inl, arg.N, lm.JtJ,
 *                    lm.Jte, lm.tmp1
 * Reads   (callees): None.
 * Writes   (direct): best.H, lm.JtJ, lm.Jte, lm.tmp1
 * Writes  (callees): None.
 */

inline void   RHO_HEST_REFC::refine(void){
    int         i;
    float       S, newS;  /* Sum of squared errors */
    float       gain;     /* Gain-parameter. */
    float       L  = 100.0f;/* Lambda of LevMarq */
    float dH[8], newH[8];

    /**
     * Iteratively refine the homography.
     */
    /* Find initial conditions */
    sacCalcJacobianErrors(best.H, arg.src, arg.dst, best.inl, arg.N,
                          lm.JtJ, lm.Jte,  &S);

    /*Levenberg-Marquardt Loop.*/
    for(i=0;i<MAXLEVMARQITERS;i++){
        /**
         * Attempt a step given current state
         *   - Jacobian-x-Jacobian   (JtJ)
         *   - Jacobian-x-error      (Jte)
         *   - Sum of squared errors (S)
         * and current parameter
         *   - Lambda (L)
         * .
         *
         * This is done by solving the system of equations
         *     Ax = b
         * where A (JtJ) and b (Jte) are sourced from our current state, and
         * the solution x becomes a step (dH) that is applied to best.H in
         * order to compute a candidate homography (newH).
         *
         * The system above is solved by Cholesky decomposition of a
         * sufficently-damped JtJ into a lower-triangular matrix (and its
         * transpose), whose inverse is then computed. This inverse (and its
         * transpose) then multiply Jte in order to find dH.
         */

        while(!sacChol8x8Damped(lm.JtJ, L, lm.tmp1)){
            L *= 2.0f;
        }
        sacTRInv8x8   (lm.tmp1, lm.tmp1);
        sacTRISolve8x8(lm.tmp1, lm.Jte,  dH);
        sacSub8x1     (newH,       best.H,  dH);
        sacCalcJacobianErrors(newH, arg.src, arg.dst, best.inl, arg.N,
                              NULL, NULL, &newS);
        gain = sacLMGain(dH, lm.Jte, S, newS, L);
        /*printf("Lambda: %12.6f  S: %12.6f  newS: %12.6f  Gain: %12.6f\n",
                 L, S, newS, gain);*/

        /**
         * If the gain is positive (i.e., the new Sum of Square Errors (newS)
         * corresponding to newH is lower than the previous one (S) ), save
         * the current state and accept the new step dH.
         *
         * If the gain is below LM_GAIN_LO, damp more (increase L), even if the
         * gain was positive. If the gain is above LM_GAIN_HI, damp less
         * (decrease L). Otherwise the gain is left unchanged.
         */

        if(gain < LM_GAIN_LO){
            L *= 8;
            if(L>1000.0f/FLT_EPSILON){
                break;/* FIXME: Most naive termination criterion imaginable. */
            }
        }else if(gain > LM_GAIN_HI){
            L *= 0.5;
        }

        if(gain > 0){
            S = newS;
            memcpy(best.H, newH, sizeof(newH));
            sacCalcJacobianErrors(best.H, arg.src, arg.dst, best.inl, arg.N,
                                  lm.JtJ, lm.Jte,  &S);
        }
    }
}


/**
 * Compute directly the JtJ, Jte and sum-of-squared-error for a given
 * homography and set of inliers.
 *
 * This is possible because the product of J and its transpose as well as with
 * the error and the sum-of-squared-error can all be computed additively
 * (match-by-match), as one would intuitively expect; All matches make
 * contributions to the error independently of each other.
 *
 * What this allows is a constant-space implementation of Lev-Marq that can
 * nevertheless be vectorized if need be.
 */

static inline void   sacCalcJacobianErrors(const float* H,
                                           const float* src,
                                           const float* dst,
                                           const char*  inl,
                                           unsigned     N,
                                           float     (* JtJ)[8],
                                           float*       Jte,
                                           float*       Sp){
    unsigned i;
    float    S;

    /* Zero out JtJ, Jte and S */
    if(JtJ){memset(JtJ, 0, 8*8*sizeof(float));}
    if(Jte){memset(Jte, 0, 8*1*sizeof(float));}
    S = 0.0f;

    /* Additively compute JtJ and Jte */
    for(i=0;i<N;i++){
        /* Skip outliers */
        if(!inl[i]){
            continue;
        }

        /**
         * Otherwise, compute additively the upper triangular matrix JtJ and
         * the Jtd vector within the following formula:
         *
         *     LaTeX:
         *     (J^{T}J + \lambda \diag( J^{T}J )) \beta = J^{T}[ y - f(\Beta) ]
         *     Simplified ASCII:
         *     (JtJ + L*diag(JtJ)) beta = Jt e, where e (error) is y-f(Beta).
         *
         * For this we need to calculate
         *     1) The 2D error (e) of the homography on the current point i
         *        using the current parameters Beta.
         *     2) The derivatives (J) of the error on the current point i under
         *        perturbations of the current parameters Beta.
         * Accumulate products of the error times the derivative to Jte, and
         * products of the derivatives to JtJ.
         */

        /* Compute Squared Error */
        float x       = src[2*i+0];
        float y       = src[2*i+1];
        float X       = dst[2*i+0];
        float Y       = dst[2*i+1];
        float W       = (H[6]*x + H[7]*y + 1.0f);
        float iW      = fabs(W) > FLT_EPSILON ? 1.0f/W : 0;

        float reprojX = (H[0]*x + H[1]*y + H[2]) * iW;
        float reprojY = (H[3]*x + H[4]*y + H[5]) * iW;

        float eX      = reprojX - X;
        float eY      = reprojY - Y;
        float e       = eX*eX + eY*eY;
        S            += e;

        /* Compute Jacobian */
        if(JtJ || Jte){
            float dxh11 = x          * iW;
            float dxh12 = y          * iW;
            float dxh13 =              iW;
          /*float dxh21 = 0.0f;*/
          /*float dxh22 = 0.0f;*/
          /*float dxh23 = 0.0f;*/
            float dxh31 = -reprojX*x * iW;
            float dxh32 = -reprojX*y * iW;

          /*float dyh11 = 0.0f;*/
          /*float dyh12 = 0.0f;*/
          /*float dyh13 = 0.0f;*/
            float dyh21 = x          * iW;
            float dyh22 = y          * iW;
            float dyh23 =              iW;
            float dyh31 = -reprojY*x * iW;
            float dyh32 = -reprojY*y * iW;

            /* Update Jte:          X             Y   */
            if(Jte){
                Jte[0]    += eX   *dxh11              ;/*  +0 */
                Jte[1]    += eX   *dxh12              ;/*  +0 */
                Jte[2]    += eX   *dxh13              ;/*  +0 */
                Jte[3]    +=               eY   *dyh21;/* 0+  */
                Jte[4]    +=               eY   *dyh22;/* 0+  */
                Jte[5]    +=               eY   *dyh23;/* 0+  */
                Jte[6]    += eX   *dxh31 + eY   *dyh31;/*  +  */
                Jte[7]    += eX   *dxh32 + eY   *dyh32;/*  +  */
            }

            /* Update JtJ:          X             Y    */
            if(JtJ){
                JtJ[0][0] += dxh11*dxh11              ;/*  +0 */

                JtJ[1][0] += dxh11*dxh12              ;/*  +0 */
                JtJ[1][1] += dxh12*dxh12              ;/*  +0 */

                JtJ[2][0] += dxh11*dxh13              ;/*  +0 */
                JtJ[2][1] += dxh12*dxh13              ;/*  +0 */
                JtJ[2][2] += dxh13*dxh13              ;/*  +0 */

              /*JtJ[3][0] +=                          ;   0+0 */
              /*JtJ[3][1] +=                          ;   0+0 */
              /*JtJ[3][2] +=                          ;   0+0 */
                JtJ[3][3] +=               dyh21*dyh21;/* 0+  */

              /*JtJ[4][0] +=                          ;   0+0 */
              /*JtJ[4][1] +=                          ;   0+0 */
              /*JtJ[4][2] +=                          ;   0+0 */
                JtJ[4][3] +=               dyh21*dyh22;/* 0+  */
                JtJ[4][4] +=               dyh22*dyh22;/* 0+  */

              /*JtJ[5][0] +=                          ;   0+0 */
              /*JtJ[5][1] +=                          ;   0+0 */
              /*JtJ[5][2] +=                          ;   0+0 */
                JtJ[5][3] +=               dyh21*dyh23;/* 0+  */
                JtJ[5][4] +=               dyh22*dyh23;/* 0+  */
                JtJ[5][5] +=               dyh23*dyh23;/* 0+  */

                JtJ[6][0] += dxh11*dxh31              ;/*  +0 */
                JtJ[6][1] += dxh12*dxh31              ;/*  +0 */
                JtJ[6][2] += dxh13*dxh31              ;/*  +0 */
                JtJ[6][3] +=               dyh21*dyh31;/* 0+  */
                JtJ[6][4] +=               dyh22*dyh31;/* 0+  */
                JtJ[6][5] +=               dyh23*dyh31;/* 0+  */
                JtJ[6][6] += dxh31*dxh31 + dyh31*dyh31;/*  +  */

                JtJ[7][0] += dxh11*dxh32              ;/*  +0 */
                JtJ[7][1] += dxh12*dxh32              ;/*  +0 */
                JtJ[7][2] += dxh13*dxh32              ;/*  +0 */
                JtJ[7][3] +=               dyh21*dyh32;/* 0+  */
                JtJ[7][4] +=               dyh22*dyh32;/* 0+  */
                JtJ[7][5] +=               dyh23*dyh32;/* 0+  */
                JtJ[7][6] += dxh31*dxh32 + dyh31*dyh32;/*  +  */
                JtJ[7][7] += dxh32*dxh32 + dyh32*dyh32;/*  +  */
            }
        }
    }

    if(Sp){*Sp = S;}
}


/**
 * Compute the Levenberg-Marquardt "gain" obtained by the given step dH.
 *
 * Drawn from http://www2.imm.dtu.dk/documents/ftp/tr99/tr05_99.pdf.
 */

static inline float  sacLMGain(const float*  dH,
                               const float*  Jte,
                               const float   S,
                               const float   newS,
                               const float   lambda){
    float dS = S-newS;
    float dL = 0;
    int i;

    /* Compute h^t h... */
    for(i=0;i<8;i++){
        dL += dH[i]*dH[i];
    }
    /* Compute mu * h^t h... */
    dL *= lambda;
    /* Subtract h^t F'... */
    for(i=0;i<8;i++){
        dL += dH[i]*Jte[i];/* += as opposed to -=, since dH we compute is
                              opposite sign. */
    }
    /* Multiply by 1/2... */
    dL *= 0.5;

    /* Return gain as S-newS / L0 - LH. */
    return fabs(dL) < FLT_EPSILON ? dS : dS / dL;
}


/**
 * Cholesky decomposition on 8x8 real positive-definite matrix defined by its
 * lower-triangular half. Outputs L, the lower triangular part of the
 * decomposition.
 *
 * A and L can overlap fully (in-place) or not at all, but may not partially
 * overlap.
 *
 * For damping, the diagonal elements are scaled by 1.0 + lambda.
 *
 * Returns zero if decomposition unsuccessful, and non-zero otherwise.
 *
 * Source: http://en.wikipedia.org/wiki/Cholesky_decomposition#
 * The_Cholesky.E2.80.93Banachiewicz_and_Cholesky.E2.80.93Crout_algorithms
 */

static inline int    sacChol8x8Damped(const float (*A)[8],
                                      float         lambda,
                                      float       (*L)[8]){
    const int N = 8;
    int i, j, k;
    float  lambdap1 = lambda + 1.0f;
    float  x;

    for(i=0;i<N;i++){/* Row */
        /* Pre-diagonal elements */
        for(j=0;j<i;j++){
            x = A[i][j];               /* Aij */
            for(k=0;k<j;k++){
                x -= L[i][k] * L[j][k];/* - Sum_{k=0..j-1} Lik*Ljk */
            }
            L[i][j] = x / L[j][j];     /* Lij = ... / Ljj */
        }

        /* Diagonal element */
        {j = i;
            x = A[j][j] * lambdap1;    /* Ajj */
            for(k=0;k<j;k++){
                x -= L[j][k] * L[j][k];/* - Sum_{k=0..j-1} Ljk^2 */
            }
            if(x<0){
                return 0;
            }
            L[j][j] = sqrtf(x);        /* Ljj = sqrt( ... ) */
        }
    }

    return 1;
}


/**
 * Invert lower-triangular 8x8 matrix L into lower-triangular matrix M.
 *
 * L and M can overlap fully (in-place) or not at all, but may not partially
 * overlap.
 *
 * Uses formulation from
 * http://www.cs.berkeley.edu/~knight/knight_math221_poster.pdf
 * , adjusted for the fact that A^T^-1 = A^-1^T. Thus:
 *
 * U11    U12                   U11^-1   -U11^-1*U12*U22^-1
 *                ->
 *  0     U22                     0            U22^-1
 *
 * Becomes
 *
 * L11     0                    L11^-1           0
 *                ->
 * L21    L22            -L22^-1*L21*L11^-1    L22^-1
 *
 * Since
 *
 * ( -L11^T^-1*L21^T*L22^T^-1 )^T = -L22^T^-1^T*L21^T^T*L11^T^-1^T
 *                                = -L22^T^T^-1*L21^T^T*L11^T^T^-1
 *                                = -L22^-1*L21*L11^-1
 */

static inline void   sacTRInv8x8(const float (*L)[8],
                                 float       (*M)[8]){
    float s[2][2], t[2][2];
    float u[4][4], v[4][4];

    /*
        L00  0   0   0   0   0   0   0
        L10 L11  0   0   0   0   0   0
        L20 L21 L22  0   0   0   0   0
        L30 L31 L32 L33  0   0   0   0
        L40 L41 L42 L43 L44  0   0   0
        L50 L51 L52 L53 L54 L55  0   0
        L60 L61 L62 L63 L64 L65 L66  0
        L70 L71 L72 L73 L74 L75 L76 L77
    */

    /* Invert 4*2 1x1 matrices; Starts recursion. */
    M[0][0] = 1.0f/L[0][0];
    M[1][1] = 1.0f/L[1][1];
    M[2][2] = 1.0f/L[2][2];
    M[3][3] = 1.0f/L[3][3];
    M[4][4] = 1.0f/L[4][4];
    M[5][5] = 1.0f/L[5][5];
    M[6][6] = 1.0f/L[6][6];
    M[7][7] = 1.0f/L[7][7];

    /*
        M00  0   0   0   0   0   0   0
        L10 M11  0   0   0   0   0   0
        L20 L21 M22  0   0   0   0   0
        L30 L31 L32 M33  0   0   0   0
        L40 L41 L42 L43 M44  0   0   0
        L50 L51 L52 L53 L54 M55  0   0
        L60 L61 L62 L63 L64 L65 M66  0
        L70 L71 L72 L73 L74 L75 L76 M77
    */

    /* 4*2 Matrix products of 1x1 matrices */
    M[1][0] = -M[1][1]*L[1][0]*M[0][0];
    M[3][2] = -M[3][3]*L[3][2]*M[2][2];
    M[5][4] = -M[5][5]*L[5][4]*M[4][4];
    M[7][6] = -M[7][7]*L[7][6]*M[6][6];

    /*
        M00  0   0   0   0   0   0   0
        M10 M11  0   0   0   0   0   0
        L20 L21 M22  0   0   0   0   0
        L30 L31 M32 M33  0   0   0   0
        L40 L41 L42 L43 M44  0   0   0
        L50 L51 L52 L53 M54 M55  0   0
        L60 L61 L62 L63 L64 L65 M66  0
        L70 L71 L72 L73 L74 L75 M76 M77
    */

    /* 2*2 Matrix products of 2x2 matrices */

    /*
       (M22  0 )   (L20 L21)   (M00  0 )
     - (M32 M33) x (L30 L31) x (M10 M11)
    */

    s[0][0] = M[2][2]*L[2][0];
    s[0][1] = M[2][2]*L[2][1];
    s[1][0] = M[3][2]*L[2][0]+M[3][3]*L[3][0];
    s[1][1] = M[3][2]*L[2][1]+M[3][3]*L[3][1];

    t[0][0] = s[0][0]*M[0][0]+s[0][1]*M[1][0];
    t[0][1] =                 s[0][1]*M[1][1];
    t[1][0] = s[1][0]*M[0][0]+s[1][1]*M[1][0];
    t[1][1] =                 s[1][1]*M[1][1];

    M[2][0] = -t[0][0];
    M[2][1] = -t[0][1];
    M[3][0] = -t[1][0];
    M[3][1] = -t[1][1];

    /*
       (M66  0 )   (L64 L65)   (M44  0 )
     - (L76 M77) x (L74 L75) x (M54 M55)
    */

    s[0][0] = M[6][6]*L[6][4];
    s[0][1] = M[6][6]*L[6][5];
    s[1][0] = M[7][6]*L[6][4]+M[7][7]*L[7][4];
    s[1][1] = M[7][6]*L[6][5]+M[7][7]*L[7][5];

    t[0][0] = s[0][0]*M[4][4]+s[0][1]*M[5][4];
    t[0][1] =                 s[0][1]*M[5][5];
    t[1][0] = s[1][0]*M[4][4]+s[1][1]*M[5][4];
    t[1][1] =                 s[1][1]*M[5][5];

    M[6][4] = -t[0][0];
    M[6][5] = -t[0][1];
    M[7][4] = -t[1][0];
    M[7][5] = -t[1][1];

    /*
        M00  0   0   0   0   0   0   0
        M10 M11  0   0   0   0   0   0
        M20 M21 M22  0   0   0   0   0
        M30 M31 M32 M33  0   0   0   0
        L40 L41 L42 L43 M44  0   0   0
        L50 L51 L52 L53 M54 M55  0   0
        L60 L61 L62 L63 M64 M65 M66  0
        L70 L71 L72 L73 M74 M75 M76 M77
    */

    /* 1*2 Matrix products of 4x4 matrices */

    /*
       (M44  0   0   0 )   (L40 L41 L42 L43)   (M00  0   0   0 )
       (M54 M55  0   0 )   (L50 L51 L52 L53)   (M10 M11  0   0 )
       (M64 M65 M66  0 )   (L60 L61 L62 L63)   (M20 M21 M22  0 )
     - (M74 M75 M76 M77) x (L70 L71 L72 L73) x (M30 M31 M32 M33)
    */

    u[0][0] = M[4][4]*L[4][0];
    u[0][1] = M[4][4]*L[4][1];
    u[0][2] = M[4][4]*L[4][2];
    u[0][3] = M[4][4]*L[4][3];
    u[1][0] = M[5][4]*L[4][0]+M[5][5]*L[5][0];
    u[1][1] = M[5][4]*L[4][1]+M[5][5]*L[5][1];
    u[1][2] = M[5][4]*L[4][2]+M[5][5]*L[5][2];
    u[1][3] = M[5][4]*L[4][3]+M[5][5]*L[5][3];
    u[2][0] = M[6][4]*L[4][0]+M[6][5]*L[5][0]+M[6][6]*L[6][0];
    u[2][1] = M[6][4]*L[4][1]+M[6][5]*L[5][1]+M[6][6]*L[6][1];
    u[2][2] = M[6][4]*L[4][2]+M[6][5]*L[5][2]+M[6][6]*L[6][2];
    u[2][3] = M[6][4]*L[4][3]+M[6][5]*L[5][3]+M[6][6]*L[6][3];
    u[3][0] = M[7][4]*L[4][0]+M[7][5]*L[5][0]+M[7][6]*L[6][0]+M[7][7]*L[7][0];
    u[3][1] = M[7][4]*L[4][1]+M[7][5]*L[5][1]+M[7][6]*L[6][1]+M[7][7]*L[7][1];
    u[3][2] = M[7][4]*L[4][2]+M[7][5]*L[5][2]+M[7][6]*L[6][2]+M[7][7]*L[7][2];
    u[3][3] = M[7][4]*L[4][3]+M[7][5]*L[5][3]+M[7][6]*L[6][3]+M[7][7]*L[7][3];

    v[0][0] = u[0][0]*M[0][0]+u[0][1]*M[1][0]+u[0][2]*M[2][0]+u[0][3]*M[3][0];
    v[0][1] =                 u[0][1]*M[1][1]+u[0][2]*M[2][1]+u[0][3]*M[3][1];
    v[0][2] =                                 u[0][2]*M[2][2]+u[0][3]*M[3][2];
    v[0][3] =                                                 u[0][3]*M[3][3];
    v[1][0] = u[1][0]*M[0][0]+u[1][1]*M[1][0]+u[1][2]*M[2][0]+u[1][3]*M[3][0];
    v[1][1] =                 u[1][1]*M[1][1]+u[1][2]*M[2][1]+u[1][3]*M[3][1];
    v[1][2] =                                 u[1][2]*M[2][2]+u[1][3]*M[3][2];
    v[1][3] =                                                 u[1][3]*M[3][3];
    v[2][0] = u[2][0]*M[0][0]+u[2][1]*M[1][0]+u[2][2]*M[2][0]+u[2][3]*M[3][0];
    v[2][1] =                 u[2][1]*M[1][1]+u[2][2]*M[2][1]+u[2][3]*M[3][1];
    v[2][2] =                                 u[2][2]*M[2][2]+u[2][3]*M[3][2];
    v[2][3] =                                                 u[2][3]*M[3][3];
    v[3][0] = u[3][0]*M[0][0]+u[3][1]*M[1][0]+u[3][2]*M[2][0]+u[3][3]*M[3][0];
    v[3][1] =                 u[3][1]*M[1][1]+u[3][2]*M[2][1]+u[3][3]*M[3][1];
    v[3][2] =                                 u[3][2]*M[2][2]+u[3][3]*M[3][2];
    v[3][3] =                                                 u[3][3]*M[3][3];

    M[4][0] = -v[0][0];
    M[4][1] = -v[0][1];
    M[4][2] = -v[0][2];
    M[4][3] = -v[0][3];
    M[5][0] = -v[1][0];
    M[5][1] = -v[1][1];
    M[5][2] = -v[1][2];
    M[5][3] = -v[1][3];
    M[6][0] = -v[2][0];
    M[6][1] = -v[2][1];
    M[6][2] = -v[2][2];
    M[6][3] = -v[2][3];
    M[7][0] = -v[3][0];
    M[7][1] = -v[3][1];
    M[7][2] = -v[3][2];
    M[7][3] = -v[3][3];

    /*
        M00  0   0   0   0   0   0   0
        M10 M11  0   0   0   0   0   0
        M20 M21 M22  0   0   0   0   0
        M30 M31 M32 M33  0   0   0   0
        M40 M41 M42 M43 M44  0   0   0
        M50 M51 M52 M53 M54 M55  0   0
        M60 M61 M62 M63 M64 M65 M66  0
        M70 M71 M72 M73 M74 M75 M76 M77
    */
}


/**
 * Solves dH = inv(JtJ) Jte. The argument lower-triangular matrix is the
 * inverse of L as produced by the Cholesky decomposition LL^T of the matrix
 * JtJ; Thus the operation performed here is a left-multiplication of a vector
 * by two triangular matrices. The math is below:
 *
 * JtJ      = LL^T
 * Linv     = L^-1
 * (JtJ)^-1 = (LL^T)^-1
 *          = (L^T^-1)(Linv)
 *          = (Linv^T)(Linv)
 * dH       = ((JtJ)^-1) (Jte)
 *          = (Linv^T)(Linv) (Jte)
 *
 * where J is nx8, Jt is 8xn, JtJ is 8x8 PD, e is nx1, Jte is 8x1, L is lower
 * triangular 8x8 and dH is 8x1.
 */

static inline void   sacTRISolve8x8(const float (*L)[8],
                                    const float*  Jte,
                                    float*        dH){
    float t[8];

    t[0]  = L[0][0]*Jte[0];
    t[1]  = L[1][0]*Jte[0]+L[1][1]*Jte[1];
    t[2]  = L[2][0]*Jte[0]+L[2][1]*Jte[1]+L[2][2]*Jte[2];
    t[3]  = L[3][0]*Jte[0]+L[3][1]*Jte[1]+L[3][2]*Jte[2]+L[3][3]*Jte[3];
    t[4]  = L[4][0]*Jte[0]+L[4][1]*Jte[1]+L[4][2]*Jte[2]+L[4][3]*Jte[3]+L[4][4]*Jte[4];
    t[5]  = L[5][0]*Jte[0]+L[5][1]*Jte[1]+L[5][2]*Jte[2]+L[5][3]*Jte[3]+L[5][4]*Jte[4]+L[5][5]*Jte[5];
    t[6]  = L[6][0]*Jte[0]+L[6][1]*Jte[1]+L[6][2]*Jte[2]+L[6][3]*Jte[3]+L[6][4]*Jte[4]+L[6][5]*Jte[5]+L[6][6]*Jte[6];
    t[7]  = L[7][0]*Jte[0]+L[7][1]*Jte[1]+L[7][2]*Jte[2]+L[7][3]*Jte[3]+L[7][4]*Jte[4]+L[7][5]*Jte[5]+L[7][6]*Jte[6]+L[7][7]*Jte[7];


    dH[0] = L[0][0]*t[0]+L[1][0]*t[1]+L[2][0]*t[2]+L[3][0]*t[3]+L[4][0]*t[4]+L[5][0]*t[5]+L[6][0]*t[6]+L[7][0]*t[7];
    dH[1] =              L[1][1]*t[1]+L[2][1]*t[2]+L[3][1]*t[3]+L[4][1]*t[4]+L[5][1]*t[5]+L[6][1]*t[6]+L[7][1]*t[7];
    dH[2] =                           L[2][2]*t[2]+L[3][2]*t[3]+L[4][2]*t[4]+L[5][2]*t[5]+L[6][2]*t[6]+L[7][2]*t[7];
    dH[3] =                                        L[3][3]*t[3]+L[4][3]*t[4]+L[5][3]*t[5]+L[6][3]*t[6]+L[7][3]*t[7];
    dH[4] =                                                     L[4][4]*t[4]+L[5][4]*t[5]+L[6][4]*t[6]+L[7][4]*t[7];
    dH[5] =                                                                  L[5][5]*t[5]+L[6][5]*t[6]+L[7][5]*t[7];
    dH[6] =                                                                               L[6][6]*t[6]+L[7][6]*t[7];
    dH[7] =                                                                                            L[7][7]*t[7];
}


/**
 * Subtract dH from H.
 */

static inline void   sacSub8x1(float* Hout, const float* H, const float* dH){
    Hout[0] = H[0] - dH[0];
    Hout[1] = H[1] - dH[1];
    Hout[2] = H[2] - dH[2];
    Hout[3] = H[3] - dH[3];
    Hout[4] = H[4] - dH[4];
    Hout[5] = H[5] - dH[5];
    Hout[6] = H[6] - dH[6];
    Hout[7] = H[7] - dH[7];
}


/* End namespace cv */
}
