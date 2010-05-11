#!/usr/bin/python 

from opencv.cv import *
from opencv.highgui import *
from random import randint

def minarea_array(img, count):
    pointMat = cvCreateMat( count, 1, CV_32SC2 )
    for i in range(count):
        pointMat[i] = cvPoint( randint(img.width/4, img.width*3/4),
                               randint(img.height/4, img.height*3/4) )

    box = cvMinAreaRect2( pointMat )
    box_vtx = cvBoxPoints( box )
    success, center, radius = cvMinEnclosingCircle( pointMat )
    cvZero( img )
    for i in range(count):
        cvCircle( img, cvGet1D(pointMat,i), 2, CV_RGB( 255, 0, 0 ), CV_FILLED, CV_AA, 0 )

    box_vtx = [cvPointFrom32f(box_vtx[0]),
               cvPointFrom32f(box_vtx[1]),
               cvPointFrom32f(box_vtx[2]),
               cvPointFrom32f(box_vtx[3])]
    cvCircle( img, cvPointFrom32f(center), cvRound(radius), CV_RGB(255, 255, 0), 1, CV_AA, 0 )
    cvPolyLine( img, [box_vtx], 1, CV_RGB(0,255,255), 1, CV_AA ) 
    

    
def minarea_seq(img, count, storage):
    ptseq = cvCreateSeq( CV_SEQ_KIND_GENERIC | CV_32SC2, sizeof_CvContour, sizeof_CvPoint, storage )
    ptseq = CvSeq_CvPoint.cast( ptseq )
    for i in range(count):
        pt0 = cvPoint( randint(img.width/4, img.width*3/4),
                       randint(img.height/4, img.height*3/4) )
        cvSeqPush( ptseq, pt0 )
    box = cvMinAreaRect2( ptseq )
    box_vtx = cvBoxPoints( box )
    success, center, radius = cvMinEnclosingCircle( ptseq )
    cvZero( img )
    for pt in ptseq: 
        cvCircle( img, pt, 2, CV_RGB( 255, 0, 0 ), CV_FILLED, CV_AA, 0 )

    box_vtx = [cvPointFrom32f(box_vtx[0]),
               cvPointFrom32f(box_vtx[1]),
               cvPointFrom32f(box_vtx[2]),
               cvPointFrom32f(box_vtx[3])]
    cvCircle( img, cvPointFrom32f(center), cvRound(radius), CV_RGB(255, 255, 0), 1, CV_AA, 0 )
    cvPolyLine( img, [box_vtx], 1, CV_RGB(0,255,255), 1, CV_AA ) 
    cvClearMemStorage( storage )

if __name__ == "__main__":
    img = cvCreateImage( cvSize( 500, 500 ), 8, 3 );
    storage = cvCreateMemStorage(0);

    cvNamedWindow( "rect & circle", 1 );
        
    use_seq=True

    while True: 
        count = randint(1,100)
        if use_seq:
            minarea_seq(img, count, storage)
        else:
            minarea_array(img, count)

        cvShowImage("rect & circle", img)
        key = cvWaitKey()
        if( key == '\x1b' ):
            break;

        use_seq = not use_seq
