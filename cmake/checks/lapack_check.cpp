#include <complex.h>
#include "opencv_lapack.h"

#ifndef LAPACK_GLOBAL
// If lapack version <= 3.4.0 
#ifdef LAPACK_NAME
// If lapack version == 3.4.0
#define LAPACK_GLOBAL(name,NAME) LAPACK_NAME(name,NAME)
#else
// If lapack version < 3.4.0 
#error Developper needs to figure this out as both LAPACK_GLOBAL and LAPACK_NAME are undefined
#endif
#endif

static char* check_fn1 = (char*)LAPACK_GLOBAL(sgesv,SGESV);
static char* check_fn2 = (char*)LAPACK_GLOBAL(sposv,SPOSV);
static char* check_fn3 = (char*)LAPACK_GLOBAL(spotrf,SPOTRF);
static char* check_fn4 = (char*)LAPACK_GLOBAL(sgesdd,SGESDD);

int main(int argc, char* argv[])
{
    (void)argv;
    if(argc > 1000)
        return check_fn1[0] + check_fn2[0] + check_fn3[0] + check_fn4[0];
    return 0;
}
