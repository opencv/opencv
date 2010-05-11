/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __CVCOMMON_H_
#define __CVCOMMON_H_

#include <cxcore.h>
#include <cv.h>
#include <cxmisc.h>

#define __BEGIN__ __CV_BEGIN__
#define __END__  __CV_END__
#define EXIT __CV_EXIT__

#define CV_DECLARE_QSORT( func_name, T, less_than )                     \
void func_name( T* array, size_t length, int aux );

#define less_than( a, b ) ((a) < (b))

CV_DECLARE_QSORT( icvSort_32f, float, less_than )

CV_DECLARE_QSORT( icvSort_32s, int, less_than )

#ifndef PATH_MAX
#define PATH_MAX 512
#endif /* PATH_MAX */

int icvMkDir( const char* filename );

/* returns index at specified position from index matrix of any type.
   if matrix is NULL, then specified position is returned */
CV_INLINE
int icvGetIdxAt( CvMat* idx, int pos );

CV_INLINE
int icvGetIdxAt( CvMat* idx, int pos )
{
    if( idx == NULL )
    {
        return pos;
    }
    else
    {
        CvScalar sc;
        int type;

        type = CV_MAT_TYPE( idx->type );
        cvRawDataToScalar( idx->data.ptr + pos *
            ( (idx->rows == 1) ? CV_ELEM_SIZE( type ) : idx->step ), type, &sc );

        return (int) sc.val[0];
    }
}

/* debug functions */

#define CV_DEBUG_SAVE( ptr ) icvSave( ptr, __FILE__, __LINE__ );

void icvSave( const CvArr* ptr, const char* filename, int line );

#endif /* __CVCOMMON_H_ */

