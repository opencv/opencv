#include <complex.h>
#include "opencv_lapack.h"

#if defined(LAPACK_GLOBAL)
/* This means LAPACK is netlib's reference implementation with version > 3.4.0 */
#define OCV_LAPACK_GLOBAL(name,NAME) LAPACK_GLOBAL(name,NAME)
#elif defined(LAPACK_NAME)
/* This means LAPACK is netlib's reference implementation version 3.4.0 */
#define OCV_LAPACK_GLOBAL(name,NAME) LAPACK_NAME(name,NAME)
#else
/* This means 1 of 2 things:
 *  - either LAPACK is netlib's reference implementation with version < 3.4.0
 *  - or another LAPACK implementation is used (Apple's Accelerate for instance)
 *
 *  Fall back to what opencv always assumed until now
 */
#define OCV_LAPACK_GLOBAL(name,NAME) name##_
#endif

static char* check_fn1 = (char*)OCV_LAPACK_GLOBAL(sgesv,SGESV);
static char* check_fn2 = (char*)OCV_LAPACK_GLOBAL(sposv,SPOSV);
static char* check_fn3 = (char*)OCV_LAPACK_GLOBAL(spotrf,SPOTRF);
static char* check_fn4 = (char*)OCV_LAPACK_GLOBAL(sgesdd,SGESDD);

int main(int argc, char* argv[])
{
    (void)argv;
    if(argc > 1000)
        return check_fn1[0] + check_fn2[0] + check_fn3[0] + check_fn4[0];
    return 0;
}
