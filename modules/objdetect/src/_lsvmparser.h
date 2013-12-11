#ifndef LSVM_PARSER
#define LSVM_PARSER
#include "opencv2/objdetect/objdetect_c.h"

#include "_lsvm_types.h"

#define MODEL    1
#define P        2
#define COMP     3
#define SCORE    4
#define RFILTER  100
#define PFILTERs 101
#define PFILTER  200
#define SIZEX    150
#define SIZEY    151
#define WEIGHTS  152
#define TAGV     300
#define Vx       350
#define Vy       351
#define TAGD     400
#define Dx       451
#define Dy       452
#define Dxx      453
#define Dyy      454
#define BTAG     500

#define STEP_END 1000

#define EMODEL    (STEP_END + MODEL)
#define EP        (STEP_END + P)
#define ECOMP     (STEP_END + COMP)
#define ESCORE    (STEP_END + SCORE)
#define ERFILTER  (STEP_END + RFILTER)
#define EPFILTERs (STEP_END + PFILTERs)
#define EPFILTER  (STEP_END + PFILTER)
#define ESIZEX    (STEP_END + SIZEX)
#define ESIZEY    (STEP_END + SIZEY)
#define EWEIGHTS  (STEP_END + WEIGHTS)
#define ETAGV     (STEP_END + TAGV)
#define EVx       (STEP_END + Vx)
#define EVy       (STEP_END + Vy)
#define ETAGD     (STEP_END + TAGD)
#define EDx       (STEP_END + Dx)
#define EDy       (STEP_END + Dy)
#define EDxx      (STEP_END + Dxx)
#define EDyy      (STEP_END + Dyy)
#define EBTAG     (STEP_END + BTAG)

//extern "C" {
    int LSVMparser(const char * filename, CvLSVMFilterObject *** model, int *last, int *max,
                   int **comp, float **b, int *count, float * score);
#ifdef __cplusplus
extern "C"
#endif
    int loadModel(

              const char *modelPath,

              CvLSVMFilterObject ***filters,
              int *kFilters,
              int *kComponents,
              int **kPartFilters,
              float **b,
              float *scoreThreshold);
//};
#endif
