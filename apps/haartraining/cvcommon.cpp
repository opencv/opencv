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

#include "_cvcommon.h"

#include <cstring>
#include <ctime>

#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#endif /* _WIN32 */


CV_IMPLEMENT_QSORT( icvSort_32f, float, less_than )

CV_IMPLEMENT_QSORT( icvSort_32s, int, less_than )

int icvMkDir( const char* filename )
{
    char path[PATH_MAX];
    char* p;
    int pos;

#ifdef _WIN32
    struct _stat st;
#else /* _WIN32 */
    struct stat st;
    mode_t mode;

    mode = 0755;
#endif /* _WIN32 */

    strcpy( path, filename );

    p = path;
    for( ; ; )
    {
        pos = (int)strcspn( p, "/\\" );

        if( pos == (int) strlen( p ) ) break;
        if( pos != 0 )
        {
            p[pos] = '\0';

#ifdef _WIN32
            if( p[pos-1] != ':' )
            {
                if( _stat( path, &st ) != 0 )
                {
                    if( _mkdir( path ) != 0 ) return 0;
                }
            }
#else /* _WIN32 */
            if( stat( path, &st ) != 0 )
            {
                if( mkdir( path, mode ) != 0 ) return 0;
            }
#endif /* _WIN32 */
        }
        
        p[pos] = '/';

        p += pos + 1;
    }

    return 1;
}

#if 0
/* debug functions */
void icvSave( const CvArr* ptr, const char* filename, int line )
{
    CvFileStorage* fs;
    char buf[PATH_MAX];
    const char* name;
    
    name = strrchr( filename, '\\' );
    if( !name ) name = strrchr( filename, '/' );
    if( !name ) name = filename;
    else name++; /* skip '/' or '\\' */

    sprintf( buf, "%s-%d-%d", name, line, time( NULL ) );
    fs = cvOpenFileStorage( buf, NULL, CV_STORAGE_WRITE_TEXT );
    if( !fs ) return;
    cvWrite( fs, "debug", ptr );
    cvReleaseFileStorage( &fs );
}
#endif // #if 0

/* End of file. */
