#include "opencv_lapack.h"

static char* check_fn1 = (char*)sgesv_;
static char* check_fn2 = (char*)sposv_;
static char* check_fn3 = (char*)spotrf_;
static char* check_fn4 = (char*)sgesdd_;

int main(int argc, char* argv[])
{
    (void)argv;
    if(argc > 1000)
        return check_fn1[0] + check_fn2[0] + check_fn3[0] + check_fn4[0];
    return 0;
}
