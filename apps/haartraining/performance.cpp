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

/*
 * performance.cpp
 *
 * Measure performance of classifier
 */
#include "opencv2/core/core.hpp"
#include "opencv2/core/internal.hpp"

#include "cv.h"
#include "highgui.h"

#include <cstdio>
#include <cmath>
#include <ctime>

#ifdef _WIN32
/* use clock() function insted of time() */
#define time( arg ) (((double) clock()) / CLOCKS_PER_SEC)
#endif /* _WIN32 */

#ifndef PATH_MAX
#define PATH_MAX 512
#endif /* PATH_MAX */

typedef struct HidCascade
{
    int size;
    int count;
} HidCascade;

typedef struct ObjectPos
{
    float x;
    float y;
    float width;
    int found;    /* for reference */
    int neghbors;
} ObjectPos;

int main( int argc, char* argv[] )
{
    int i, j;
    char* classifierdir = NULL;
    //char* samplesdir    = NULL;

    int saveDetected = 1;
    double scale_factor = 1.2;
    float maxSizeDiff = 1.5F;
    float maxPosDiff  = 0.3F;

    /* number of stages. if <=0 all stages are used */
    int nos = -1, nos0;

    int width  = 24;
    int height = 24;

    int rocsize;

    FILE* info;
    char* infoname;
    char fullname[PATH_MAX];
    char detfilename[PATH_MAX];
    char* filename;
    char detname[] = "det-";

    CvHaarClassifierCascade* cascade;
    CvMemStorage* storage;
    CvSeq* objects;

    double totaltime;

    infoname = (char*)"";
    rocsize = 40;
    if( argc == 1 )
    {
        printf( "Usage: %s\n  -data <classifier_directory_name>\n"
                "  -info <collection_file_name>\n"
                "  [-maxSizeDiff <max_size_difference = %f>]\n"
                "  [-maxPosDiff <max_position_difference = %f>]\n"
                "  [-sf <scale_factor = %f>]\n"
                "  [-ni]\n"
                "  [-nos <number_of_stages = %d>]\n"
                "  [-rs <roc_size = %d>]\n"
                "  [-w <sample_width = %d>]\n"
                "  [-h <sample_height = %d>]\n",
                argv[0], maxSizeDiff, maxPosDiff, scale_factor, nos, rocsize,
                width, height );

        return 0;
    }

    for( i = 1; i < argc; i++ )
    {
        if( !strcmp( argv[i], "-data" ) )
        {
            classifierdir = argv[++i];
        }
        else if( !strcmp( argv[i], "-info" ) )
        {
            infoname = argv[++i];
        }
        else if( !strcmp( argv[i], "-maxSizeDiff" ) )
        {
            maxSizeDiff = (float) atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-maxPosDiff" ) )
        {
            maxPosDiff = (float) atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-sf" ) )
        {
            scale_factor = atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-ni" ) )
        {
            saveDetected = 0;
        }
        else if( !strcmp( argv[i], "-nos" ) )
        {
            nos = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-rs" ) )
        {
            rocsize = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-w" ) )
        {
            width = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-h" ) )
        {
            height = atoi( argv[++i] );
        }
    }

    cascade = cvLoadHaarClassifierCascade( classifierdir, cvSize( width, height ) );
    if( cascade == NULL )
    {
        printf( "Unable to load classifier from %s\n", classifierdir );

        return 1;
    }

    int* numclassifiers = new int[cascade->count];
    numclassifiers[0] = cascade->stage_classifier[0].count;
    for( i = 1; i < cascade->count; i++ )
    {
        numclassifiers[i] = numclassifiers[i-1] + cascade->stage_classifier[i].count;
    }

    storage = cvCreateMemStorage();

    nos0 = cascade->count;
    if( nos <= 0 )
        nos = nos0;

    strcpy( fullname, infoname );
    filename = strrchr( fullname, '\\' );
    if( filename == NULL )
    {
        filename = strrchr( fullname, '/' );
    }
    if( filename == NULL )
    {
        filename = fullname;
    }
    else
    {
        filename++;
    }

    info = fopen( infoname, "r" );
    totaltime = 0.0;
    if( info != NULL )
    {
        int x, y, width, height;
        IplImage* img;
        int hits, missed, falseAlarms;
        int totalHits, totalMissed, totalFalseAlarms;
        int found;
        float distance;

        int refcount;
        ObjectPos* ref;
        int detcount;
        ObjectPos* det;
        int error=0;

        int* pos;
        int* neg;

        pos = (int*) cvAlloc( rocsize * sizeof( *pos ) );
        neg = (int*) cvAlloc( rocsize * sizeof( *neg ) );
        for( i = 0; i < rocsize; i++ ) { pos[i] = neg[i] = 0; }

        printf( "+================================+======+======+======+\n" );
        printf( "|            File Name           | Hits |Missed| False|\n" );
        printf( "+================================+======+======+======+\n" );

        totalHits = totalMissed = totalFalseAlarms = 0;
        while( !feof( info ) )
        {
            if( fscanf( info, "%s %d", filename, &refcount ) != 2 || refcount <= 0 ) break;

            img = cvLoadImage( fullname );
            if( !img ) continue;

            ref = (ObjectPos*) cvAlloc( refcount * sizeof( *ref ) );
            for( i = 0; i < refcount; i++ )
            {
                error = (fscanf( info, "%d %d %d %d", &x, &y, &width, &height ) != 4);
                if( error ) break;
                ref[i].x = 0.5F * width  + x;
                ref[i].y = 0.5F * height + y;
                ref[i].width = sqrtf( 0.5F * (width * width + height * height) );
                ref[i].found = 0;
                ref[i].neghbors = 0;
            }
            if( !error )
            {
                cvClearMemStorage( storage );

                cascade->count = nos;
                totaltime -= time( 0 );
                objects = cvHaarDetectObjects( img, cascade, storage, scale_factor, 1 );
                totaltime += time( 0 );
                cascade->count = nos0;

                detcount = ( objects ? objects->total : 0);
                det = (detcount > 0) ?
                    ( (ObjectPos*)cvAlloc( detcount * sizeof( *det )) ) : NULL;
                hits = missed = falseAlarms = 0;
                for( i = 0; i < detcount; i++ )
                {
                    CvAvgComp r = *((CvAvgComp*) cvGetSeqElem( objects, i ));
                    det[i].x = 0.5F * r.rect.width  + r.rect.x;
                    det[i].y = 0.5F * r.rect.height + r.rect.y;
                    det[i].width = sqrtf( 0.5F * (r.rect.width * r.rect.width +
                                                  r.rect.height * r.rect.height) );
                    det[i].neghbors = r.neighbors;

                    if( saveDetected )
                    {
                        cvRectangle( img, cvPoint( r.rect.x, r.rect.y ),
                            cvPoint( r.rect.x + r.rect.width, r.rect.y + r.rect.height ),
                            CV_RGB( 255, 0, 0 ), 3 );
                    }

                    found = 0;
                    for( j = 0; j < refcount; j++ )
                    {
                        distance = sqrtf( (det[i].x - ref[j].x) * (det[i].x - ref[j].x) +
                                          (det[i].y - ref[j].y) * (det[i].y - ref[j].y) );
                        if( (distance < ref[j].width * maxPosDiff) &&
                            (det[i].width > ref[j].width / maxSizeDiff) &&
                            (det[i].width < ref[j].width * maxSizeDiff) )
                        {
                            ref[j].found = 1;
                            ref[j].neghbors = MAX( ref[j].neghbors, det[i].neghbors );
                            found = 1;
                        }
                    }
                    if( !found )
                    {
                        falseAlarms++;
                        neg[MIN(det[i].neghbors, rocsize - 1)]++;
                    }
                }
                for( j = 0; j < refcount; j++ )
                {
                    if( ref[j].found )
                    {
                        hits++;
                        pos[MIN(ref[j].neghbors, rocsize - 1)]++;
                    }
                    else
                    {
                        missed++;
                    }
                }

                totalHits += hits;
                totalMissed += missed;
                totalFalseAlarms += falseAlarms;
                printf( "|%32.32s|%6d|%6d|%6d|\n", filename, hits, missed, falseAlarms );
                printf( "+--------------------------------+------+------+------+\n" );
                fflush( stdout );

                if( saveDetected )
                {
                    strcpy( detfilename, detname );
                    strcat( detfilename, filename );
                    strcpy( filename, detfilename );
                    cvvSaveImage( fullname, img );
                }

                if( det ) { cvFree( &det ); det = NULL; }
            } /* if( !error ) */

            cvReleaseImage( &img );
            cvFree( &ref );
        }
        fclose( info );

        printf( "|%32.32s|%6d|%6d|%6d|\n", "Total",
                totalHits, totalMissed, totalFalseAlarms );
        printf( "+================================+======+======+======+\n" );
        printf( "Number of stages: %d\n", nos );
        printf( "Number of weak classifiers: %d\n", numclassifiers[nos - 1] );
        printf( "Total time: %f\n", totaltime );

        /* print ROC to stdout */
        for( i = rocsize - 1; i > 0; i-- )
        {
            pos[i-1] += pos[i];
            neg[i-1] += neg[i];
        }
        fprintf( stderr, "%d\n", nos );
        for( i = 0; i < rocsize; i++ )
        {
            fprintf( stderr, "\t%d\t%d\t%f\t%f\n", pos[i], neg[i],
                ((float)pos[i]) / (totalHits + totalMissed),
                ((float)neg[i]) / (totalHits + totalMissed) );
        }

        cvFree( &pos );
        cvFree( &neg );
    }

    delete[] numclassifiers;

    cvReleaseHaarClassifierCascade( &cascade );
    cvReleaseMemStorage( &storage );

    return 0;
}

