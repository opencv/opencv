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

#include "precomp.hpp"

#if 0
//#ifdef WIN32 /* make sure it builds under Linux whenever it is included into Makefile.am or not. */

//void icvCutContour( CvSeq* current, IplImage* image );
CvSeq* icvCutContourRaster( CvSeq* current, CvMemStorage* storage, IplImage* image );


//create lists of segments of all contours from image
CvSeq* cvExtractSingleEdges( IplImage* image, //bw image - it's content will be destroyed by cvFindContours
                             CvMemStorage* storage )
{
    CvMemStorage* tmp_storage = cvCreateChildMemStorage( storage );
    CvSeq* contours = 0;
    cvFindContours( image, tmp_storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE );
    cvZero( image );

    //iterate through contours
      //iterate through tree
    CvSeq* current = contours;
    int number = 0;
    int level = 1;

    CvSeq* output = 0;
    CvSeq* tail_seq = 0;

    //actually this loop can iterates through tree,
    //but still we use CV_RETR_LIST it is not useful
    while( current )
    {
        number++;

        //get vertical list of segments for one contour
        CvSeq* new_seq = icvCutContourRaster( current, storage,  image );

        //add this vertical list to horisontal list
        if( new_seq )
        {
            if( tail_seq )
            {
                tail_seq->h_next = new_seq;
                new_seq->h_prev = tail_seq;
                tail_seq = new_seq;
            }
            else
            {
                output = tail_seq = new_seq;
            }
        }

        //iteration through tree
        if( current->v_next )
        {
            //goto child
            current = current->v_next;
            level++;
        }
        else
        {
            //go parent
            while( !current->h_next )
            {
                current = current->v_prev;
                level--;
                if( !level ) break;
            }

            if( current ) //go brother
                current = current->h_next;
        }
    }

    //free temporary memstorage with initial contours
    cvReleaseMemStorage( &tmp_storage );

    return output;
}

//makes vertical list of segments for 1 contour
CvSeq* icvCutContourRaster( CvSeq* current, CvMemStorage* storage, IplImage* image /*tmp image*/)
{
    //iplSet(image, 0 ); // this can cause double edges if two contours have common edge
                       // for example if object is circle with 1 pixel width
                       // to remove such problem - remove this iplSet

    //approx contour by single edges
    CvSeqReader reader;
    CvSeqWriter writer;

    int writing = 0;
    cvStartReadSeq( current, &reader, 0 );
    //below line just to avoid warning
    cvStartWriteSeq( current->flags, sizeof(CvContour), sizeof(CvPoint), storage, &writer );

    CvSeq* output = 0;
    CvSeq* tail = 0;

    //first pass through contour - compute number of branches at every point
    int i;
    for( i = 0; i < current->total; i++ )
    {
        CvPoint cur;

        CV_READ_SEQ_ELEM( cur, reader );

        //mark point
        ((uchar*)image->imageData)[image->widthStep * cur.y + cur.x]++;
        assert( ((uchar*)image->imageData)[image->widthStep * cur.y + cur.x] != 255 );

    }

    //second pass - create separate edges
    for( i = 0; i < current->total; i++ )
    {
        CvPoint cur;

        CV_READ_SEQ_ELEM( cur, reader );

        //get pixel at this point
        uchar flag = image->imageData[image->widthStep * cur.y + cur.x];
        if( flag != 255 && flag < 3) //
        {
            if(!writing)
            {
                cvStartWriteSeq( current->flags, sizeof(CvContour), sizeof(CvPoint), storage, &writer );
                writing = 1 ;
            }

            //mark point
            if( flag < 3 ) ((uchar*)image->imageData)[image->widthStep * cur.y + cur.x] = 255;
            //add it to another seq
            CV_WRITE_SEQ_ELEM( cur, writer );

        }
        else
        {
            //exclude this point from contour
           if( writing )
           {
               CvSeq* newseq = cvEndWriteSeq( &writer );
               writing = 0;

               if( tail )
               {
                   tail->v_next = newseq;
                   newseq->v_prev = tail;
                   tail = newseq;
               }
               else
               {
                   output = tail = newseq;
               }
           }
        }
    }


   if( writing ) //if were not self intersections
   {
       CvSeq* newseq = cvEndWriteSeq( &writer );
       writing = 0;

       if( tail )
       {
           tail->v_next = newseq;
           newseq->v_prev = tail;
           tail = newseq;
       }
       else
       {
           output = tail = newseq;
       }
   }


    return output;

}


/*void icvCutContour( CvSeq* current, IplImage* image )
{
    //approx contour by single edges
    CvSeqReader reader;
    CvSeqReader rev_reader;

    cvStartReadSeq( current, &reader, 0 );

    int64* cur_pt = (int64*)reader.ptr;
    int64* prev_pt = (int64*)reader.prev_elem;

    //search for point a in aba position
    for( int i = 0; i < current->total; i++ )
    {
        CV_NEXT_SEQ_ELEM( sizeof(int64), reader );

        //compare current reader pos element with old previous
        if( prev_pt[0] == ((int64*)reader.ptr)[0] )
        {
            //return to prev pos
            CV_PREV_SEQ_ELEM( sizeof(int64), reader );


            //this point is end of edge
            //start going both directions and collect edge
            cvStartReadSeq( current, &rev_reader, 1 );

            int pos = cvGetSeqReaderPos( &reader );
            cvSetSeqReaderPos( &rev_reader, pos );

            //walk in both directions
            while(1);


        }
        int64* cur_pt = (int64*)reader.ptr;
        int64* prev_pt = (int64*)reader.prev_elem;

    }
}

*/
#endif /* WIN32 */
