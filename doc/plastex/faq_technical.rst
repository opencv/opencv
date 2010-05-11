
FAQ Technical Questions on Library Use
--------------------------------------

How to access image pixels
^^^^^^^^^^^^^^^^^^^^^^^^^^

(The coordinates are 0-based and counted from image origin, either top-left (img->origin=IPL_ORIGIN_TL) or bottom-left (img->origin=IPL_ORIGIN_BL)

    * Suppose, we have 8-bit 1-channel image I (IplImage* img)::

        I(x,y) ~ ((uchar*)(img->imageData + img->widthStep*y))[x]

    * Suppose, we have 8-bit 3-channel image I (IplImage* img)::

        I(x,y)blue ~ ((uchar*)(img->imageData + img->widthStep*y))[x*3]
        I(x,y)green ~ ((uchar*)(img->imageData + img->widthStep*y))[x*3+1]
        I(x,y)red ~ ((uchar*)(img->imageData + img->widthStep*y))[x*3+2]

      e.g. increasing brightness of point (100,100) by 30 can be done this way::

        CvPoint pt = {100,100};
        ((uchar*)(img->imageData + img->widthStep*pt.y))[pt.x*3] += 30;
        ((uchar*)(img->imageData + img->widthStep*pt.y))[pt.x*3+1] += 30;
        ((uchar*)(img->imageData + img->widthStep*pt.y))[pt.x*3+2] += 30;

      or more efficiently::

        CvPoint pt = {100,100};
        uchar* temp_ptr = &((uchar*)(img->imageData + img->widthStep*pt.y))[pt.x*3];
        temp_ptr[0] += 30;
        temp_ptr[1] += 30;
        temp_ptr[2] += 30;

    * Suppose, we have 32-bit floating point, 1-channel image I (IplImage* img)::

        I(x,y) ~ ((float*)(img->imageData + img->widthStep*y))[x]

    * Now, the general case: suppose, we have N-channel image of type T::

        I(x,y)c ~ ((T*)(img->imageData + img->widthStep*y))[x*N + c]

      or you may use macro CV_IMAGE_ELEM( image_header, elemtype, y, x_Nc )::

        I(x,y)c ~ CV_IMAGE_ELEM( img, T, y, x*N + c )

There are functions that work with arbitrary (up to 4-channel) images and matrices (cvGet2D, cvSet2D), but they are pretty slow. 

