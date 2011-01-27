// Disable some warnings which are caused with CUDA headers
#pragma warning(disable: 4201 4408 4100)

#include <iostream>
#include <cvconfig.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;
using namespace cv;

int main()
{
    bool can_run = true;

#if !defined(HAVE_CUDA)
    cout << "CUDA support is required (CMake key 'WITH_CUDA' must be true).\n";
    can_run = false;
#endif

#if !defined(HAVE_TBB)
    cout << "TBB support is required (CMake key 'WITH_TBB' must be true).\n";
    can_run = false;
#endif

    if (!can_run)
        return -1;

    return 0;
}