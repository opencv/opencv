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
 * haartraining.cpp
 *
 * Train cascade classifier
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "cvhaartraining.h"

int main( int argc, char* argv[] )
{
    int i = 0;
    char* nullname = (char*)"(NULL)";

    char* vecname = NULL;
    char* dirname = NULL;
    char* bgname  = NULL;

    bool bg_vecfile = false;
    int npos    = 2000;
    int nneg    = 2000;
    int nstages = 14;
    int mem     = 200;
    int nsplits = 1;
    float minhitrate     = 0.995F;
    float maxfalsealarm  = 0.5F;
    float weightfraction = 0.95F;
    int mode         = 0;
    int symmetric    = 1;
    int equalweights = 0;
    int width  = 24;
    int height = 24;
    const char* boosttypes[] = { "DAB", "RAB", "LB", "GAB" };
    int boosttype = 3;
    const char* stumperrors[] = { "misclass", "gini", "entropy" };
    int stumperror = 0;
    int maxtreesplits = 0;
    int minpos = 500;

    if( argc == 1 )
    {
        printf( "Usage: %s\n  -data <dir_name>\n"
                "  -vec <vec_file_name>\n"
                "  -bg <background_file_name>\n"
                "  [-bg-vecfile]\n"
                "  [-npos <number_of_positive_samples = %d>]\n"
                "  [-nneg <number_of_negative_samples = %d>]\n"
                "  [-nstages <number_of_stages = %d>]\n"
                "  [-nsplits <number_of_splits = %d>]\n"
                "  [-mem <memory_in_MB = %d>]\n"
                "  [-sym (default)] [-nonsym]\n"
                "  [-minhitrate <min_hit_rate = %f>]\n"
                "  [-maxfalsealarm <max_false_alarm_rate = %f>]\n"
                "  [-weighttrimming <weight_trimming = %f>]\n"
                "  [-eqw]\n"
                "  [-mode <BASIC (default) | CORE | ALL>]\n"
                "  [-w <sample_width = %d>]\n"
                "  [-h <sample_height = %d>]\n"
                "  [-bt <DAB | RAB | LB | GAB (default)>]\n"
                "  [-err <misclass (default) | gini | entropy>]\n"
                "  [-maxtreesplits <max_number_of_splits_in_tree_cascade = %d>]\n"
                "  [-minpos <min_number_of_positive_samples_per_cluster = %d>]\n",
                argv[0], npos, nneg, nstages, nsplits, mem,
                minhitrate, maxfalsealarm, weightfraction, width, height,
                maxtreesplits, minpos );

        return 0;
    }

    for( i = 1; i < argc; i++ )
    {
        if( !strcmp( argv[i], "-data" ) )
        {
            dirname = argv[++i];
        }
        else if( !strcmp( argv[i], "-vec" ) )
        {
            vecname = argv[++i];
        }
        else if( !strcmp( argv[i], "-bg" ) )
        {
            bgname = argv[++i];
        }
        else if( !strcmp( argv[i], "-bg-vecfile" ) )
        {
            bg_vecfile = true;
        }
        else if( !strcmp( argv[i], "-npos" ) )
        {
            npos = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-nneg" ) )
        {
            nneg = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-nstages" ) )
        {
            nstages = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-nsplits" ) )
        {
            nsplits = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-mem" ) )
        {
            mem = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-sym" ) )
        {
            symmetric = 1;
        }
        else if( !strcmp( argv[i], "-nonsym" ) )
        {
            symmetric = 0;
        }
        else if( !strcmp( argv[i], "-minhitrate" ) )
        {
            minhitrate = (float) atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-maxfalsealarm" ) )
        {
            maxfalsealarm = (float) atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-weighttrimming" ) )
        {
            weightfraction = (float) atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-eqw" ) )
        {
            equalweights = 1;
        }
        else if( !strcmp( argv[i], "-mode" ) )
        {
            char* tmp = argv[++i];

            if( !strcmp( tmp, "CORE" ) )
            {
                mode = 1;
            }
            else if( !strcmp( tmp, "ALL" ) )
            {
                mode = 2;
            }
            else
            {
                mode = 0;
            }
        }
        else if( !strcmp( argv[i], "-w" ) )
        {
            width = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-h" ) )
        {
            height = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-bt" ) )
        {
            i++;
            if( !strcmp( argv[i], boosttypes[0] ) )
            {
                boosttype = 0;
            }
            else if( !strcmp( argv[i], boosttypes[1] ) )
            {
                boosttype = 1;
            }
            else if( !strcmp( argv[i], boosttypes[2] ) )
            {
                boosttype = 2;
            }
            else
            {
                boosttype = 3;
            }
        }
        else if( !strcmp( argv[i], "-err" ) )
        {
            i++;
            if( !strcmp( argv[i], stumperrors[0] ) )
            {
                stumperror = 0;
            }
            else if( !strcmp( argv[i], stumperrors[1] ) )
            {
                stumperror = 1;
            }
            else
            {
                stumperror = 2;
            }
        }
        else if( !strcmp( argv[i], "-maxtreesplits" ) )
        {
            maxtreesplits = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-minpos" ) )
        {
            minpos = atoi( argv[++i] );
        }
    }

    printf( "Data dir name: %s\n", ((dirname == NULL) ? nullname : dirname ) );
    printf( "Vec file name: %s\n", ((vecname == NULL) ? nullname : vecname ) );
    printf( "BG  file name: %s, is a vecfile: %s\n", ((bgname == NULL) ? nullname : bgname ), bg_vecfile ? "yes" : "no" );
    printf( "Num pos: %d\n", npos );
    printf( "Num neg: %d\n", nneg );
    printf( "Num stages: %d\n", nstages );
    printf( "Num splits: %d (%s as weak classifier)\n", nsplits,
        (nsplits == 1) ? "stump" : "tree" );
    printf( "Mem: %d MB\n", mem );
    printf( "Symmetric: %s\n", (symmetric) ? "TRUE" : "FALSE" );
    printf( "Min hit rate: %f\n", minhitrate );
    printf( "Max false alarm rate: %f\n", maxfalsealarm );
    printf( "Weight trimming: %f\n", weightfraction );
    printf( "Equal weights: %s\n", (equalweights) ? "TRUE" : "FALSE" );
    printf( "Mode: %s\n", ( (mode == 0) ? "BASIC" : ( (mode == 1) ? "CORE" : "ALL") ) );
    printf( "Width: %d\n", width );
    printf( "Height: %d\n", height );
    //printf( "Max num of precalculated features: %d\n", numprecalculated );
    printf( "Applied boosting algorithm: %s\n", boosttypes[boosttype] );
    printf( "Error (valid only for Discrete and Real AdaBoost): %s\n",
            stumperrors[stumperror] );

    printf( "Max number of splits in tree cascade: %d\n", maxtreesplits );
    printf( "Min number of positive samples per cluster: %d\n", minpos );

    cvCreateTreeCascadeClassifier( dirname, vecname, bgname,
                               npos, nneg, nstages, mem,
                               nsplits,
                               minhitrate, maxfalsealarm, weightfraction,
                               mode, symmetric,
                               equalweights, width, height,
                               boosttype, stumperror,
                               maxtreesplits, minpos, bg_vecfile );

    return 0;
}
