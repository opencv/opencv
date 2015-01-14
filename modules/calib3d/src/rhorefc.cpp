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
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include "rhorefc.h"



/* Defines */
#define MEM_ALIGN               32
#define HSIZE                   (3*3*sizeof(float))
#define MIN_DELTA_CHNG          0.1
#define REL_CHNG(a, b)          (fabs((a) - (b))/(a))
#define CHNG_SIGNIFICANT(a, b)  (REL_CHNG(a, b) > MIN_DELTA_CHNG)
#define CHI_STAT                2.706
#define CHI_SQ                  1.645
#define RLO                     0.25
#define RHI                     0.75
#define MAXLEVMARQITERS         100
#define m                       4       /* 4 points required per model */
#define SPRT_T_M                25      /* Guessing 25 match evlauations / 1 model generation */
#define SPRT_M_S                1       /* 1 model per sample */
#define SPRT_EPSILON            0.1     /* No explanation */
#define SPRT_DELTA              0.01    /* No explanation */



/* For the sake of cv:: namespace ONLY: */
#ifdef __cplusplus
namespace cv{/* For C support, replace with extern "C" { */
#endif


/* Data Structures */



/* Prototypes */
static inline void*  almalloc(size_t nBytes);
static inline void   alfree(void* ptr);

static inline int    sacInitRun(RHO_HEST_REFC* p);
static inline void   sacFiniRun(RHO_HEST_REFC* p);
static inline int    sacHaveExtrinsicGuess(RHO_HEST_REFC* p);
static inline int    sacHypothesize(RHO_HEST_REFC* p);
static inline int    sacVerify(RHO_HEST_REFC* p);
static inline int    sacIsNREnabled(RHO_HEST_REFC* p);
static inline int    sacIsRefineEnabled(RHO_HEST_REFC* p);
static inline int    sacIsFinalRefineEnabled(RHO_HEST_REFC* p);
static inline int    sacPROSACPhaseEndReached(RHO_HEST_REFC* p);
static inline void   sacPROSACGoToNextPhase(RHO_HEST_REFC* p);
static inline void   sacGetPROSACSample(RHO_HEST_REFC* p);
static inline int    sacIsSampleDegenerate(RHO_HEST_REFC* p);
static inline void   sacGenerateModel(RHO_HEST_REFC* p);
static inline int    sacIsModelDegenerate(RHO_HEST_REFC* p);
static inline void   sacEvaluateModelSPRT(RHO_HEST_REFC* p);
static inline void   sacUpdateSPRT(RHO_HEST_REFC* p);
static inline void   sacDesignSPRTTest(RHO_HEST_REFC* p);
static inline int    sacIsBestModel(RHO_HEST_REFC* p);
static inline int    sacIsBestModelGoodEnough(RHO_HEST_REFC* p);
static inline void   sacSaveBestModel(RHO_HEST_REFC* p);
static inline void   sacInitNonRand(double    beta,
                                    unsigned  start,
                                    unsigned  N,
                                    unsigned* nonRandMinInl);
static inline void   sacNStarOptimize(RHO_HEST_REFC* p);
static inline void   sacUpdateBounds(RHO_HEST_REFC* p);
static inline void   sacOutputModel(RHO_HEST_REFC* p);
static inline void   sacOutputZeroH(RHO_HEST_REFC* p);

static inline double sacInitPEndFpI(const unsigned ransacConvg,
                                    const unsigned n,
                                    const unsigned s);
static inline void   sacRndSmpl(unsigned  sampleSize,
                                unsigned* currentSample,
                                unsigned  dataSetSize);
static inline double sacRandom(void);
static inline unsigned sacCalcIterBound(double   confidence,
                                        double   inlierRate,
                                        unsigned sampleSize,
                                        unsigned maxIterBound);
static inline void   hFuncRefC(float* packedPoints, float* H);
static inline int    sacCanRefine(RHO_HEST_REFC* p);
static inline void   sacRefine(RHO_HEST_REFC* p);
static inline void   sacCalcJacobianErrors(const float* restrict H,
                                           const float* restrict src,
                                           const float* restrict dst,
                                           const char*  restrict inl,
                                           unsigned              N,
                                           float     (* restrict JtJ)[8],
                                           float*       restrict Jte,
                                           float*       restrict Sp);
static inline float  sacDs(const float (*JtJ)[8],
                           const float*  dH,
                           const float*  Jte);
static inline int    sacChol8x8Damped (const float (*A)[8],
                                       float         lambda,
                                       float       (*L)[8]);
static inline void   sacTRInv8x8(const float (*L)[8],
                                 float       (*M)[8]);
static inline void   sacTRISolve8x8(const float (*L)[8],
                                    const float*  Jte,
                                    float*        dH);
static inline void   sacSub8x1(float* Hout, const float* H, const float* dH);



/* Functions */

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
 * @param [in/out] p  The uninitialized estimator context to initialize.
 * @return 0 if successful; non-zero if an error occured.
 */

int  rhoRefCInit(RHO_HEST_REFC* p){
    memset(p, 0, sizeof(*p));

    p->ctrl.smpl   = (unsigned*)almalloc(m*sizeof(*p->ctrl.smpl));

    p->curr.pkdPts = (float*)   almalloc(m*2*2*sizeof(*p->curr.pkdPts));
    p->curr.H      = (float*)   almalloc(HSIZE);
    p->curr.inl    = NULL;
    p->curr.numInl = 0;

    p->best.H      = (float*)   almalloc(HSIZE);
    p->best.inl    = NULL;
    p->best.numInl = 0;

    p->nr.tbl      = NULL;/* By default this table is not computed. */
    p->nr.size     = 0;
    p->nr.beta     = 0.0;

    p->lm.ws       = NULL;
    p->lm.JtJ      = NULL;
    p->lm.tmp1     = NULL;
    p->lm.Jte      = NULL;


    int areAllAllocsSuccessful = p->ctrl.smpl   &&
                                 p->curr.H      &&
                                 p->best.H      &&
                                 p->curr.pkdPts;

    if(!areAllAllocsSuccessful){
        rhoRefCFini(p);
    }

    return areAllAllocsSuccessful;
}


/**
 * Ensure that the estimator context's internal table for non-randomness
 * criterion is at least of the given size, and uses the given beta. The table
 * should be larger than the maximum number of matches fed into the estimator.
 *
 * A value of N of 0 requests deallocation of the table.
 *
 * @param [in] p     The initialized estimator context
 * @param [in] N     If 0, deallocate internal table. If > 0, ensure that the
 *                   internal table is of at least this size, reallocating if
 *                   necessary.
 * @param [in] beta  The beta-factor to use within the table.
 * @return 1 if successful; 0 if an error occured.
 *
 * Reads:  nr.*
 * Writes: nr.*
 */

int  rhoRefCEnsureCapacity(RHO_HEST_REFC* p, unsigned N, double beta){
    unsigned* tmp;


    if(N == 0){
        /* Deallocate table */
        alfree(p->nr.tbl);
        p->nr.tbl  = NULL;
        p->nr.size = 0;
    }else{
        /* Ensure table at least as big as N and made for correct beta. */
        if(p->nr.tbl && p->nr.beta == beta && p->nr.size >= N){
            /* Table already correctly set up */
        }else{
            if(p->nr.size < N){
                /* Reallocate table because it is too small. */
                tmp = (unsigned*)almalloc(N*sizeof(unsigned));
                if(!tmp){
                    return 0;
                }

                /* Must recalculate in whole or part. */
                if(p->nr.beta != beta){
                    /* Beta changed; recalculate in whole. */
                    sacInitNonRand(beta, 0, N, tmp);
                    alfree(p->nr.tbl);
                }else{
                    /* Beta did not change; Copy over any work already done. */
                    memcpy(tmp, p->nr.tbl, p->nr.size*sizeof(unsigned));
                    sacInitNonRand(beta, p->nr.size, N, tmp);
                    alfree(p->nr.tbl);
                }

                p->nr.tbl  = tmp;
                p->nr.size = N;
                p->nr.beta = beta;
            }else{
                /* Might recalculate in whole, or not at all. */
                if(p->nr.beta != beta){
                    /* Beta changed; recalculate in whole. */
                    sacInitNonRand(beta, 0, p->nr.size, p->nr.tbl);
                    p->nr.beta = beta;
                }else{
                    /* Beta did not change; Table was already big enough. Do nothing. */
                    /* Besides, this is unreachable. */
                }
            }
        }
    }

    return 1;
}


/**
 * Finalize the estimator context, by freeing the aligned buffers used
 * internally.
 *
 * @param [in] p  The initialized estimator context to finalize.
 */

void rhoRefCFini(RHO_HEST_REFC* p){
    alfree(p->ctrl.smpl);
    alfree(p->curr.H);
    alfree(p->best.H);
    alfree(p->curr.pkdPts);
    alfree(p->nr.tbl);

    memset(p, 0, sizeof(*p));
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

unsigned rhoRefC(RHO_HEST_REFC* restrict p,       /* Homography estimation context. */
                 const float*   restrict src,     /* Source points */
                 const float*   restrict dst,     /* Destination points */
                 char*          restrict inl,     /* Inlier mask */
                 unsigned                N,       /*  = src.length = dst.length = inl.length */
                 float                   maxD,    /* Works:     3.0 */
                 unsigned                maxI,    /* Works:    2000 */
                 unsigned                rConvg,  /* Works:    2000 */
                 double                  cfd,     /* Works:   0.995 */
                 unsigned                minInl,  /* Minimum:     4 */
                 double                  beta,    /* Works:    0.35 */
                 unsigned                flags,   /* Works:       0 */
                 const float*            guessH,  /* Extrinsic guess, NULL if none provided */
                 float*                  finalH){ /* Final result. */

    /**
     * Setup
     */

    p->arg.src     = src;
    p->arg.dst     = dst;
    p->arg.inl     = inl;
    p->arg.N       = N;
    p->arg.maxD    = maxD;
    p->arg.maxI    = maxI;
    p->arg.rConvg  = rConvg;
    p->arg.cfd     = cfd;
    p->arg.minInl  = minInl;
    p->arg.beta    = beta;
    p->arg.flags   = flags;
    p->arg.guessH  = guessH;
    p->arg.finalH  = finalH;
    if(!sacInitRun(p)){
        sacOutputZeroH(p);
        sacFiniRun(p);
        return 0;
    }

    /**
     * Extrinsic Guess
     */

    if(sacHaveExtrinsicGuess(p)){
        sacVerify(p);
    }


    /**
     * PROSAC Loop
     */

    for(p->ctrl.i=0; p->ctrl.i < p->arg.maxI; p->ctrl.i++){
        sacHypothesize(p) && sacVerify(p);
    }


    /**
     * Teardown
     */

    if(sacIsFinalRefineEnabled(p) && sacCanRefine(p)){
        sacRefine(p);
    }

    sacOutputModel(p);
    sacFiniRun(p);
    return sacIsBestModelGoodEnough(p) ? p->best.numInl : 0;
}


/**
 * Allocate memory aligned to a boundary of MEMALIGN.
 */

static inline void*  almalloc(size_t nBytes){
    if(nBytes){
        unsigned char* ptr = (unsigned char*)malloc(MEM_ALIGN + nBytes);
        if(ptr){
            unsigned char* adj = (unsigned char*)(((intptr_t)(ptr+MEM_ALIGN))&((intptr_t)(-MEM_ALIGN)));
            ptrdiff_t diff = adj - ptr;
            adj[-1] = diff - 1;
            return adj;
        }
    }

    return NULL;
}

/**
 * Free aligned memory.
 *
 * If argument is NULL, do nothing in accordance with free() semantics.
 */

static inline void   alfree(void* ptr){
    if(ptr){
        unsigned char* cptr = (unsigned char*)ptr;
        free(cptr - (ptrdiff_t)cptr[-1] - 1);
    }
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

static inline int    sacInitRun(RHO_HEST_REFC* p){
    /**
     * Sanitize arguments.
     *
     * Runs zeroth because these are easy-to-check errors and unambiguously
     * mean something or other.
     */

    if(!p->arg.src || !p->arg.dst){
        /* Arguments src or dst are insane, must be != NULL */
        return 0;
    }
    if(p->arg.N < m){
        /* Argument N is insane, must be >= 4. */
        return 0;
    }
    if(p->arg.maxD < 0){
        /* Argument maxD is insane, must be >= 0. */
        return 0;
    }
    if(p->arg.cfd < 0 || p->arg.cfd > 1){
        /* Argument cfd is insane, must be in [0, 1]. */
        return 0;
    }
    /* Clamp minInl to 4 or higher. */
    p->arg.minInl = p->arg.minInl < m ? m : p->arg.minInl;
    if(sacIsNREnabled(p) && (p->arg.beta <= 0 || p->arg.beta >= 1)){
        /* Argument beta is insane, must be in (0, 1). */
        return 0;
    }
    if(!p->arg.finalH){
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

    if(sacIsNREnabled(p) && !rhoRefCEnsureCapacity(p, p->arg.N, p->arg.beta)){
        return 0;
    }

    /**
     * Inlier mask alloc.
     *
     * Runs second because we want to quit as fast as possible if we can't even
     * allocate the up tp two masks.
     *
     * If the calling software wants an output mask, use buffer provided. If
     * not, allocate one anyways internally.
     */

    p->best.inl = p->arg.inl ? p->arg.inl : (char*)almalloc(p->arg.N);
    p->curr.inl = (char*)almalloc(p->arg.N);

    if(!p->curr.inl || !p->best.inl){
        return 0;
    }

    /**
     * LevMarq workspace alloc.
     *
     * Runs third, consists only in a few conditional mallocs. If malloc fails
     * we wish to quit before doing any serious work.
     */

    if(sacIsRefineEnabled(p) || sacIsFinalRefineEnabled(p)){
        p->lm.ws   = (float*)almalloc(2*8*8*sizeof(float) + 1*8*sizeof(float));
        if(!p->lm.ws){
            return 0;
        }

        p->lm.JtJ  = (float(*)[8])(p->lm.ws + 0*8*8);
        p->lm.tmp1 = (float(*)[8])(p->lm.ws + 1*8*8);
        p->lm.Jte  = (float*)     (p->lm.ws + 2*8*8);
    }else{
        p->lm.ws = NULL;
    }

    /**
     * Reset scalar per-run state.
     *
     * Runs fourth because there's no point in resetting/calculating a large
     * number of fields if something in the above junk failed.
     */

    p->ctrl.i            = 0;
    p->ctrl.phNum        = m;
    p->ctrl.phEndI       = 1;
    p->ctrl.phEndFpI     = sacInitPEndFpI(p->arg.rConvg, p->arg.N, m);
    p->ctrl.phMax        = p->arg.N;
    p->ctrl.phNumInl     = 0;
    p->ctrl.numModels    = 0;

    if(sacHaveExtrinsicGuess(p)){
        memcpy(p->curr.H, p->arg.guessH, HSIZE);
    }else{
        memset(p->curr.H, 0, HSIZE);
    }
    p->curr.numInl       = 0;

    memset(p->best.H, 0, HSIZE);
    p->best.numInl       = 0;

    p->eval.Ntested      = 0;
    p->eval.Ntestedtotal = 0;
    p->eval.good         = 1;
    p->eval.t_M          = SPRT_T_M;
    p->eval.m_S          = SPRT_M_S;
    p->eval.epsilon      = SPRT_EPSILON;
    p->eval.delta        = SPRT_DELTA;
    sacDesignSPRTTest(p);

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

static inline void   sacFiniRun(RHO_HEST_REFC* p){
    /**
     * If no output inlier mask was required, free both (internal) masks.
     * Else if an (external) mask was provided as argument, find the other
     * (the internal one) and free it.
     */

    if(p->arg.inl){
        if(p->arg.inl == p->best.inl){
            alfree(p->curr.inl);
        }else{
            alfree(p->best.inl);
        }
    }else{
        alfree(p->best.inl);
        alfree(p->curr.inl);
    }

    p->best.inl = NULL;
    p->curr.inl = NULL;

    /**
     * ₣ree the Levenberg-Marquardt workspace.
     */

    alfree(p->lm.ws);
    p->lm.ws = NULL;
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

static inline int    sacHypothesize(RHO_HEST_REFC* p){
    if(sacPROSACPhaseEndReached(p)){
        sacPROSACGoToNextPhase(p);
    }

    sacGetPROSACSample(p);
    if(sacIsSampleDegenerate(p)){
        return 0;
    }

    sacGenerateModel(p);
    if(sacIsModelDegenerate(p)){
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

static inline int    sacVerify(RHO_HEST_REFC* p){
    sacEvaluateModelSPRT(p);
    sacUpdateSPRT(p);

    if(sacIsBestModel(p)){
        sacSaveBestModel(p);

        if(sacIsRefineEnabled(p) && sacCanRefine(p)){
            sacRefine(p);
        }

        sacUpdateBounds(p);

        if(sacIsNREnabled(p)){
            sacNStarOptimize(p);
        }
    }

    return 1;
}

/**
 * Check whether extrinsic guess was provided or not.
 *
 * @return Zero if no extrinsic guess was provided; non-zero otherwiseEE.
 */

static inline int    sacHaveExtrinsicGuess(RHO_HEST_REFC* p){
    return !!p->arg.guessH;
}

/**
 * Check whether non-randomness criterion is enabled.
 *
 * @return Zero if non-randomness criterion disabled; non-zero if not.
 */

static inline int    sacIsNREnabled(RHO_HEST_REFC* p){
    return p->arg.flags & RHO_FLAG_ENABLE_NR;
}

/**
 * Check whether best-model-so-far refinement is enabled.
 *
 * @return Zero if best-model-so-far refinement disabled; non-zero if not.
 */

static inline int    sacIsRefineEnabled(RHO_HEST_REFC* p){
    return p->arg.flags & RHO_FLAG_ENABLE_REFINEMENT;
}

/**
 * Check whether final-model refinement is enabled.
 *
 * @return Zero if final-model refinement disabled; non-zero if not.
 */

static inline int    sacIsFinalRefineEnabled(RHO_HEST_REFC* p){
    return p->arg.flags & RHO_FLAG_ENABLE_FINAL_REFINEMENT;
}

/**
 * Computes whether the end of the current PROSAC phase has been reached. At
 * PROSAC phase phNum, only matches [0, phNum) are sampled from.
 *
 * Accesses:
 * Read: i, phEndI, phNum, phMax.
 */

static inline int    sacPROSACPhaseEndReached(RHO_HEST_REFC* p){
    return p->ctrl.i >= p->ctrl.phEndI && p->ctrl.phNum < p->ctrl.phMax;
}

/**
 * Updates unconditionally the necessary fields to move to the next PROSAC
 * stage.
 *
 * Not idempotent.
 *
 * Accesses:
 * Read:  phNum, phEndFpI, phEndI
 * Write: phNum, phEndFpI, phEndI
 */

static inline void   sacPROSACGoToNextPhase(RHO_HEST_REFC* p){
    double next;

    p->ctrl.phNum++;
    next        = (p->ctrl.phEndFpI * p->ctrl.phNum)/(p->ctrl.phNum - m);
    p->ctrl.phEndI  += ceil(next - p->ctrl.phEndFpI);
    p->ctrl.phEndFpI = next;
}

/**
 * Get a sample according to PROSAC rules. Namely:
 * - If we're past the phase end interation, select randomly 4 out of the first
 *   phNum matches.
 * - Otherwise, select match phNum-1 and select randomly the 3 others out of
 *   the first phNum-1 matches.
 */

static inline void   sacGetPROSACSample(RHO_HEST_REFC* p){
    if(p->ctrl.i > p->ctrl.phEndI){
        sacRndSmpl(4, p->ctrl.smpl, p->ctrl.phNum);/* Used to be phMax */
    }else{
        sacRndSmpl(3, p->ctrl.smpl, p->ctrl.phNum-1);
        p->ctrl.smpl[3] = p->ctrl.phNum-1;
    }
}

/**
 * Checks whether the *sample* is degenerate prior to model generation.
 * - First, the extremely cheap numerical degeneracy test is run, which weeds
 *   out bad samples to the optimized GE implementation.
 * - Second, the geometrical degeneracy test is run, which weeds out most other
 *   bad samples.
 */

static inline int    sacIsSampleDegenerate(RHO_HEST_REFC* p){
    unsigned i0 = p->ctrl.smpl[0], i1 = p->ctrl.smpl[1], i2 = p->ctrl.smpl[2], i3 = p->ctrl.smpl[3];
    typedef struct{float x,y;} MyPt2f;
    MyPt2f* pkdPts = (MyPt2f*)p->curr.pkdPts, *src = (MyPt2f*)p->arg.src, *dst = (MyPt2f*)p->arg.dst;

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
 */

static inline void   sacGenerateModel(RHO_HEST_REFC* p){
    hFuncRefC(p->curr.pkdPts, p->curr.H);
}

/**
 * Checks whether the model is itself degenerate.
 * - One test: All elements of the homography are added, and if the result is
 *   NaN the homography is rejected.
 */

static inline int    sacIsModelDegenerate(RHO_HEST_REFC* p){
    int degenerate;
    float* H = p->curr.H;
    float f=H[0]+H[1]+H[2]+H[3]+H[4]+H[5]+H[6]+H[7];

    /* degenerate = isnan(f); */
    degenerate = f!=f;/* Only NaN is not equal to itself. */
    /* degenerate = 0; */


    return degenerate;
}

/**
 * Evaluates the current model using SPRT for early exiting.
 *
 * Reads:  arg.maxD, arg.src, arg.dst, curr.H, eval.*
 * Writes: eval.*, curr.inl, curr.numInl
 */

static inline void   sacEvaluateModelSPRT(RHO_HEST_REFC* p){
    unsigned i;
    unsigned isInlier;
    double   lambda  = 1.0;
    float    distSq  = p->arg.maxD*p->arg.maxD;
    const float* src = p->arg.src;
    const float* dst = p->arg.dst;
    char*    inl     = p->curr.inl;
    const float*   H = p->curr.H;


    p->ctrl.numModels++;

    p->curr.numInl   = 0;
    p->eval.Ntested  = 0;
    p->eval.good     = 1;


    /* SCALAR */
    for(i=0;i<p->arg.N && p->eval.good;i++){
        /* Backproject */
        float x=src[i*2],y=src[i*2+1];
        float X=dst[i*2],Y=dst[i*2+1];

        float reprojX=H[0]*x+H[1]*y+H[2]; /*  ( X_1 )     ( H_11 H_12    H_13  ) (x_1)       */
        float reprojY=H[3]*x+H[4]*y+H[5]; /*  ( X_2 )  =  ( H_21 H_22    H_23  ) (x_2)       */
        float reprojZ=H[6]*x+H[7]*y+1.0;  /*  ( X_3 )     ( H_31 H_32 H_33=1.0 ) (x_3 = 1.0) */

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
        p->curr.numInl += isInlier;
        *inl++     = isInlier;


        /* SPRT */
        lambda *= isInlier ? p->eval.lambdaAccept : p->eval.lambdaReject;
        p->eval.good = lambda <= p->eval.A;
        /* If !p->good, the threshold A was exceeded, so we're rejecting */
    }


    p->eval.Ntested       = i;
    p->eval.Ntestedtotal += i;
}

/**
 * Update either the delta or epsilon SPRT parameters, depending on the events
 * that transpired in the previous evaluation.
 *
 * If a "good" model that is also the best was encountered, update epsilon,
 * since
 *
 * Reads:  eval.good, eval.delta, eval.epsilon, eval.t_M, eval.m_S,
 *         curr.numInl, best.numInl
 * Writes: eval.epsilon, eval.delta, eval.A, eval.lambdaAccept,
 *         eval.lambdaReject
 */

static inline void   sacUpdateSPRT(RHO_HEST_REFC* p){
    if(p->eval.good){
        if(sacIsBestModel(p)){
            p->eval.epsilon = (double)p->curr.numInl/p->arg.N;
            sacDesignSPRTTest(p);
        }
    }else{
        double newDelta = (double)p->curr.numInl/p->eval.Ntested;

        if(newDelta > 0 && CHNG_SIGNIFICANT(p->eval.delta, newDelta)){
            p->eval.delta = newDelta;
            sacDesignSPRTTest(p);
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

static inline double designSPRTTest(double delta, double epsilon, double t_M, double m_S){
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
 * Idempotent, reentrant, thread-safe.
 *
 * Reads:  eval.delta, eval.epsilon, eval.t_M, eval.m_S
 * Writes: eval.A, eval.lambdaAccept, eval.lambdaReject
 */

static inline void   sacDesignSPRTTest(RHO_HEST_REFC* p){
    p->eval.A = designSPRTTest(p->eval.delta, p->eval.epsilon, p->eval.t_M, p->eval.m_S);
    p->eval.lambdaReject = ((1.0 - p->eval.delta) / (1.0 - p->eval.epsilon));
    p->eval.lambdaAccept = ((   p->eval.delta   ) / (    p->eval.epsilon  ));
}

/**
 * Return whether the current model is the best model so far.
 */

static inline int    sacIsBestModel(RHO_HEST_REFC* p){
    return p->curr.numInl > p->best.numInl;
}

/**
 * Returns whether the current-best model is good enough to be an
 * acceptable best model, by checking whether it meets the minimum
 * number of inliers.
 */

static inline int    sacIsBestModelGoodEnough(RHO_HEST_REFC* p){
    return p->best.numInl >= p->arg.minInl;
}

/**
 * Make current model new best model by swapping the homography, inlier mask
 * and count of inliers between the current and best models.
 */

static inline void   sacSaveBestModel(RHO_HEST_REFC* p){
    float*   H      = p->curr.H;
    char*    inl    = p->curr.inl;
    unsigned numInl = p->curr.numInl;

    p->curr.H       = p->best.H;
    p->curr.inl     = p->best.inl;
    p->curr.numInl  = p->best.numInl;

    p->best.H       = H;
    p->best.inl     = inl;
    p->best.numInl  = numInl;
}

/**
 * Compute NR table entries [start, N) for given beta.
 */

static inline void   sacInitNonRand(double    beta,
                                    unsigned  start,
                                    unsigned  N,
                                    unsigned* nonRandMinInl){
    unsigned n = m+1 > start ? m+1 : start;
    double   beta_beta1_sq_chi = sqrt(beta*(1.0-beta)) * CHI_SQ;

    for(; n < N; n++){
        double   mu      = n * beta;
        double   sigma   = sqrt(n)* beta_beta1_sq_chi;
        unsigned i_min   = ceil(m + mu + sigma);

        nonRandMinInl[n] = i_min;
    }
}

/**
 * Optimize the stopping criterion to account for the non-randomness criterion
 * of PROSAC.
 */

static inline void   sacNStarOptimize(RHO_HEST_REFC* p){
    unsigned min_sample_length = 10*2; /*(p->N * INLIERS_RATIO) */
    unsigned best_n       = p->arg.N;
    unsigned test_n       = best_n;
    unsigned bestNumInl   = p->best.numInl;
    unsigned testNumInl   = bestNumInl;

    for(;test_n > min_sample_length && testNumInl;test_n--){
        if(testNumInl*best_n > bestNumInl*test_n){
            if(testNumInl < p->nr.tbl[test_n]){
                break;
            }
            best_n      = test_n;
            bestNumInl  = testNumInl;
        }
        testNumInl -= !!p->arg.inl[test_n-1];
    }

    if(bestNumInl*p->ctrl.phMax > p->ctrl.phNumInl*best_n){
        p->ctrl.phMax    = best_n;
        p->ctrl.phNumInl = bestNumInl;
        p->arg.maxI      = sacCalcIterBound(p->arg.cfd,
                                            (double)p->ctrl.phNumInl/p->ctrl.phMax,
                                            m,
                                            p->arg.maxI);
    }
}

/**
 * Classic RANSAC iteration bound based on largest # of inliers.
 */

static inline void   sacUpdateBounds(RHO_HEST_REFC* p){
    p->arg.maxI = sacCalcIterBound(p->arg.cfd,
                                   (double)p->best.numInl/p->arg.N,
                                   m,
                                   p->arg.maxI);
}

/**
 * Ouput the best model so far to the output argument.
 */

static inline void   sacOutputModel(RHO_HEST_REFC* p){
    if(sacIsBestModelGoodEnough(p)){
        memcpy(p->arg.finalH, p->best.H, HSIZE);
    }else{
        sacOutputZeroH(p);
    }
}

/**
 * Ouput a zeroed H to the output argument.
 */

static inline void   sacOutputZeroH(RHO_HEST_REFC* p){
    memset(p->arg.finalH, 0, HSIZE);
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
 * Choose, without repetition, sampleSize integers in the range [0, numDataPoints).
 */

static inline void sacRndSmpl(unsigned  sampleSize,
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
            double U=sacRandom();
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
                currentSample[i]=dataSetSize*sacRandom();

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
 * Generates a random double uniformly distributed in the range [0, 1].
 */

static inline double sacRandom(void){
#ifdef _WIN32
    return ((double)rand())/RAND_MAX;
#else
    return ((double)random())/INT_MAX;
#endif
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

        retVal = ceil(log(1.-confidence)/log(atLeastOneOutlierProbability));
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

    scalar1=minor[0][0];
    major[0][0]/=scalar1;
    major[1][0]/=scalar1;
    major[2][0]/=scalar1;
    major[0][4]/=scalar1;
    major[1][4]/=scalar1;
    major[2][4]/=scalar1;

    scalar1=minor[1][1];
    major[0][1]/=scalar1;
    major[1][1]/=scalar1;
    major[2][1]/=scalar1;
    major[0][5]/=scalar1;
    major[1][5]/=scalar1;
    major[2][5]/=scalar1;


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
 */

static inline int    sacCanRefine(RHO_HEST_REFC* p){
    /**
     * If we only have 4 matches, GE's result is already optimal and cannot
     * be refined any further.
     */

    return p->best.numInl > m;
}



static inline void dumpMat8x8(float (*M)[8]){
    for(int i=0;i<8;i++){
        printf("\t");
        for(int j=0;j<=i;j++){
            printf("%14.6e%s", M[i][j], j==i?"\n":", ");
        }
    }
    printf("\n");
}

/**
 * Refines the best-so-far homography (p->best.H).
 */

static inline void   sacRefine(RHO_HEST_REFC* p){
    int         i = 0;
    float       S = 0, newS = 0, dS = 0;/* Sum of squared errors */
    float       R = 0;          /* R - Parameter */
    float       L  = 1.0f;  /* Lambda of LevMarq */
    float dH[8], newH[8];

    /**
     * Iteratively refine the homography.
     */

    /* Find initial conditions */
    sacCalcJacobianErrors(p->best.H, p->arg.src, p->arg.dst, p->arg.inl, p->arg.N,
                          p->lm.JtJ, p->lm.Jte,  &S);
    /*printf("Initial Error: %12.6f\n", S);*/

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
         * transpose) then multiply JtJ in order to find dH.
         */

        /*printf("Lambda: %f\n", L);*/
        while(!sacChol8x8Damped(p->lm.JtJ, L, p->lm.tmp1)){
            L *= 2.0f;/*printf("CholFail! Lambda: %f\n", L);*/
        }
        sacTRInv8x8   (p->lm.tmp1, p->lm.tmp1);
        sacTRISolve8x8(p->lm.tmp1, p->lm.Jte,  dH);
        sacSub8x1     (newH,       p->best.H,  dH);
        sacCalcJacobianErrors(newH, p->arg.src, p->arg.dst, p->arg.inl, p->arg.N,
                              NULL, NULL, &newS);

        /**
         * If the new Sum of Square Errors (newS) corresponding to newH is
         * lower than the previous one, save the current state and undamp
         * sligthly (decrease L). Else damp more (increase L).
         */

        if(newS < S){
            S = newS;
            memcpy(p->best.H, newH, sizeof(newH));
            sacCalcJacobianErrors(p->best.H, p->arg.src, p->arg.dst, p->arg.inl, p->arg.N,
                                  p->lm.JtJ, p->lm.Jte,  &S);
            /*printf("Error: %12.6f\n", S);*/
            L *= 0.5;
            if(L<FLT_EPSILON){
                L = 1;
            }
        }else{
            L *= 2.0;
            if(L>1.0f/FLT_EPSILON){
                break;
            }
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

static inline void   sacCalcJacobianErrors(const float* restrict H,
                                           const float* restrict src,
                                           const float* restrict dst,
                                           const char*  restrict inl,
                                           unsigned              N,
                                           float     (* restrict JtJ)[8],
                                           float*       restrict Jte,
                                           float*       restrict Sp){
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
 * Compute the derivative of the rate of change of the SSE.
 *
 * Inspired entirely by OpenCV's levmarq.cpp. To be reviewed.
 */

static inline float  sacDs(const float (*JtJ)[8],
                           const float*  dH,
                           const float*  Jte){
    float tdH[8];
    int   i, j;
    float dS = 0;

    /* Perform tdH = -JtJ*dH + 2*Jte. */
    for(i=0;i<8;i++){
        tdH[i] = 0;

        for(j=0;j<8;j++){
            tdH[i] -= JtJ[i][j] * dH[j];
        }

        tdH[i] += 2*Jte[i];
    }

    /* Perform dS = dH.dot(tdH).  */
    for(i=0;i<8;i++){
        dS += dH[i]*tdH[i];
    }

    return dS;
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
    const register int N = 8;
    int i, j, k;
    float  lambdap1 = lambda + 1.0f;
    double x;

    for(i=0;i<N;i++){/* Row */
        /* Pre-diagonal elements */
        for(j=0;j<i;j++){
            x = A[i][j];                       /* Aij */
            for(k=0;k<j;k++){
                x -= (double)L[i][k] * L[j][k];/* - Sum_{k=0..j-1} Lik*Ljk */
            }
            L[i][j] = x / L[j][j];             /* Lij = ... / Ljj */
        }

        /* Diagonal element */
        {j = i;
            x = A[j][j] * lambdap1;            /* Ajj */
            for(k=0;k<j;k++){
                x -= (double)L[j][k] * L[j][k];/* - Sum_{k=0..j-1} Ljk^2 */
            }
            if(x<0){
                return 0;
            }
            L[j][j] = sqrt(x);                 /* Ljj = sqrt( ... ) */
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



#ifdef __cplusplus
}
#endif
