// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#include "cvutils.hpp"

namespace cv {

namespace ndsrvp {

// fastMalloc

// [0][1][2][3][4][5][6][7][8][9]
//     ^udata
//                          ^adata
//              ^adata[-1] == udata

void* fastMalloc(size_t size)
{
    uchar* udata = (uchar*)malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
    if(!udata)
        ndsrvp_error(Error::StsNoMem, "fastMalloc(): Not enough memory");
    uchar** adata = (uchar**)align((size_t)((uchar**)udata + 1), CV_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void fastFree(void* ptr)
{
    if(ptr)
    {
        uchar* udata = ((uchar**)ptr)[-1];
        if(!(udata < (uchar*)ptr && ((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*) + CV_MALLOC_ALIGN)))
            ndsrvp_error(Error::StsBadArg, "fastFree(): Invalid memory block");
        free(udata);
    }
}

// borderInterpolate

int borderInterpolate(int p, int len, int borderType)
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == CV_HAL_BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == CV_HAL_BORDER_REFLECT || borderType == CV_HAL_BORDER_REFLECT_101 )
    {
        int delta = borderType == CV_HAL_BORDER_REFLECT_101;
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == CV_HAL_BORDER_WRAP )
    {
        ndsrvp_assert(len > 0);
        if( p < 0 )
            p -= ((p - len + 1) / len) * len;
        if( p >= len )
            p %= len;
    }
    else if( borderType == CV_HAL_BORDER_CONSTANT )
        p = -1;
    else
        ndsrvp_error(Error::StsBadArg, "borderInterpolate(): Unknown/unsupported border type");
    return p;
}

} // namespace ndsrvp

} // namespace cv
