Basic Structures
================

.. index:: CvPoint

CvPoint
-------

.. ctype:: CvPoint


2D point with integer coordinates (usually zero-based).

::


    
    typedef struct CvPoint
    {
        int x; 
        int y; 
    }
    CvPoint;
    

..



    
    
    .. attribute:: x
    
    
    
        x-coordinate 
    
    
    
    .. attribute:: y
    
    
    
        y-coordinate 
    
    
    



::


    
    /* Constructor */
    inline CvPoint cvPoint( int x, int y );
    
    /* Conversion from CvPoint2D32f */
    inline CvPoint cvPointFrom32f( CvPoint2D32f point );
    

..


.. index:: CvPoint2D32f

.. _CvPoint2D32f:

CvPoint2D32f
------------

`id=0.245532424939 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvPoint2D32f>`__

.. ctype:: CvPoint2D32f



2D point with floating-point coordinates




::


    
    typedef struct CvPoint2D32f
    {
        float x;
        float y; 
    }
    CvPoint2D32f;
    

..



    
    
    .. attribute:: x
    
    
    
        x-coordinate 
    
    
    
    .. attribute:: y
    
    
    
        y-coordinate 
    
    
    



::


    
    /* Constructor */
    inline CvPoint2D32f cvPoint2D32f( double x, double y );
    
    /* Conversion from CvPoint */
    inline CvPoint2D32f cvPointTo32f( CvPoint point );
    

..


.. index:: CvPoint3D32f

.. _CvPoint3D32f:

CvPoint3D32f
------------

`id=0.0440394368915 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvPoint3D32f>`__

.. ctype:: CvPoint3D32f



3D point with floating-point coordinates




::


    
    typedef struct CvPoint3D32f
    {
        float x; 
        float y; 
        float z; 
    }
    CvPoint3D32f;
    

..



    
    
    .. attribute:: x
    
    
    
        x-coordinate 
    
    
    
    .. attribute:: y
    
    
    
        y-coordinate 
    
    
    
    .. attribute:: z
    
    
    
        z-coordinate 
    
    
    



::


    
    /* Constructor */
    inline CvPoint3D32f cvPoint3D32f( double x, double y, double z );
    

..


.. index:: CvPoint2D64f

.. _CvPoint2D64f:

CvPoint2D64f
------------

`id=0.709504732939 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvPoint2D64f>`__

.. ctype:: CvPoint2D64f



2D point with double precision floating-point coordinates




::


    
    typedef struct CvPoint2D64f
    {
        double x; 
        double y; 
    }
    CvPoint2D64f;
    

..



    
    
    .. attribute:: x
    
    
    
        x-coordinate 
    
    
    
    .. attribute:: y
    
    
    
        y-coordinate 
    
    
    



::


    
    /* Constructor */
    inline CvPoint2D64f cvPoint2D64f( double x, double y );
    
    /* Conversion from CvPoint */
    inline CvPoint2D64f cvPointTo64f( CvPoint point );
    

..


.. index:: CvPoint3D64f

.. _CvPoint3D64f:

CvPoint3D64f
------------

`id=0.0506448340848 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvPoint3D64f>`__

.. ctype:: CvPoint3D64f



3D point with double precision floating-point coordinates




::


    
    typedef struct CvPoint3D64f
    {
        double x; 
        double y; 
        double z; 
    }
    CvPoint3D64f;
    

..



    
    
    .. attribute:: x
    
    
    
        x-coordinate 
    
    
    
    .. attribute:: y
    
    
    
        y-coordinate 
    
    
    
    .. attribute:: z
    
    
    
        z-coordinate 
    
    
    



::


    
    /* Constructor */
    inline CvPoint3D64f cvPoint3D64f( double x, double y, double z );
    

..


.. index:: CvSize

.. _CvSize:

CvSize
------

`id=0.554248071465 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvSize>`__

.. ctype:: CvSize



Pixel-accurate size of a rectangle.




::


    
    typedef struct CvSize
    {
        int width; 
        int height; 
    }
    CvSize;
    

..



    
    
    .. attribute:: width
    
    
    
        Width of the rectangle 
    
    
    
    .. attribute:: height
    
    
    
        Height of the rectangle 
    
    
    



::


    
    /* Constructor */
    inline CvSize cvSize( int width, int height );
    

..


.. index:: CvSize2D32f

.. _CvSize2D32f:

CvSize2D32f
-----------

`id=0.905432526523 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvSize2D32f>`__

.. ctype:: CvSize2D32f



Sub-pixel accurate size of a rectangle.




::


    
    typedef struct CvSize2D32f
    {
        float width; 
        float height; 
    }
    CvSize2D32f;
    

..



    
    
    .. attribute:: width
    
    
    
        Width of the rectangle 
    
    
    
    .. attribute:: height
    
    
    
        Height of the rectangle 
    
    
    



::


    
    /* Constructor */
    inline CvSize2D32f cvSize2D32f( double width, double height );
    

..


.. index:: CvRect

.. _CvRect:

CvRect
------

`id=0.213953446247 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvRect>`__

.. ctype:: CvRect



Offset (usually the top-left corner) and size of a rectangle.




::


    
    typedef struct CvRect
    {
        int x; 
        int y; 
        int width; 
        int height; 
    }
    CvRect;
    

..



    
    
    .. attribute:: x
    
    
    
        x-coordinate of the top-left corner 
    
    
    
    .. attribute:: y
    
    
    
        y-coordinate of the top-left corner (bottom-left for Windows bitmaps) 
    
    
    
    .. attribute:: width
    
    
    
        Width of the rectangle 
    
    
    
    .. attribute:: height
    
    
    
        Height of the rectangle 
    
    
    



::


    
    /* Constructor */
    inline CvRect cvRect( int x, int y, int width, int height );
    

..


.. index:: CvScalar

.. _CvScalar:

CvScalar
--------

`id=0.760314360939 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvScalar>`__

.. ctype:: CvScalar



A container for 1-,2-,3- or 4-tuples of doubles.




::


    
    typedef struct CvScalar
    {
        double val[4];
    }
    CvScalar;
    

..




::


    
    /* Constructor: 
    initializes val[0] with val0, val[1] with val1, etc. 
    */
    inline CvScalar cvScalar( double val0, double val1=0,
                              double val2=0, double val3=0 );
    /* Constructor: 
    initializes all of val[0]...val[3] with val0123 
    */
    inline CvScalar cvScalarAll( double val0123 );
    
    /* Constructor: 
    initializes val[0] with val0, and all of val[1]...val[3] with zeros 
    */
    inline CvScalar cvRealScalar( double val0 );
    

..


.. index:: CvTermCriteria

.. _CvTermCriteria:

CvTermCriteria
--------------

`id=0.267162264997 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvTermCriteria>`__

.. ctype:: CvTermCriteria



Termination criteria for iterative algorithms.




::


    
    #define CV_TERMCRIT_ITER    1
    #define CV_TERMCRIT_NUMBER  CV_TERMCRIT_ITER
    #define CV_TERMCRIT_EPS     2
    
    typedef struct CvTermCriteria
    {
        int    type;
        int    max_iter; 
        double epsilon; 
    }
    CvTermCriteria;
    

..



    
    
    .. attribute:: type
    
    
    
        A combination of CV _ TERMCRIT _ ITER and CV _ TERMCRIT _ EPS 
    
    
    
    .. attribute:: max_iter
    
    
    
        Maximum number of iterations 
    
    
    
    .. attribute:: epsilon
    
    
    
        Required accuracy 
    
    
    



::


    
    /* Constructor */
    inline CvTermCriteria cvTermCriteria( int type, int max_iter, double epsilon );
    
    /* Check and transform a CvTermCriteria so that 
       type=CV_TERMCRIT_ITER+CV_TERMCRIT_EPS
       and both max_iter and epsilon are valid */
    CvTermCriteria cvCheckTermCriteria( CvTermCriteria criteria,
                                        double default_eps,
                                        int default_max_iters );
    

..


.. index:: CvMat

.. _CvMat:

CvMat
-----

`id=0.465191243774 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvMat>`__

.. ctype:: CvMat



A multi-channel matrix.




::


    
    typedef struct CvMat
    {
        int type; 
        int step; 
    
        int* refcount; 
    
        union
        {
            uchar* ptr;
            short* s;
            int* i;
            float* fl;
            double* db;
        } data; 
    
    #ifdef __cplusplus
        union
        {
            int rows;
            int height;
        };
    
        union
        {
            int cols;
            int width;
        };
    #else
        int rows; 
        int cols; 
    #endif
    
    } CvMat;
    

..



    
    
    .. attribute:: type
    
    
    
        A CvMat signature (CV _ MAT _ MAGIC _ VAL) containing the type of elements and flags 
    
    
    
    .. attribute:: step
    
    
    
        Full row length in bytes 
    
    
    
    .. attribute:: refcount
    
    
    
        Underlying data reference counter 
    
    
    
    .. attribute:: data
    
    
    
        Pointers to the actual matrix data 
    
    
    
    .. attribute:: rows
    
    
    
        Number of rows 
    
    
    
    .. attribute:: cols
    
    
    
        Number of columns 
    
    
    
Matrices are stored row by row. All of the rows are aligned by 4 bytes.

.. index:: CvMatND

.. _CvMatND:

CvMatND
-------

`id=0.322223772253 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvMatND>`__

.. ctype:: CvMatND



Multi-dimensional dense multi-channel array.




::


    
    typedef struct CvMatND
    {
        int type; 
        int dims;
    
        int* refcount; 
    
        union
        {
            uchar* ptr;
            short* s;
            int* i;
            float* fl;
            double* db;
        } data; 
    
        struct
        {
            int size;
            int step;
        }
        dim[CV_MAX_DIM];
    
    } CvMatND;
    

..



    
    
    .. attribute:: type
    
    
    
        A CvMatND signature (CV _ MATND _ MAGIC _ VAL), combining the type of elements and flags 
    
    
    
    .. attribute:: dims
    
    
    
        The number of array dimensions 
    
    
    
    .. attribute:: refcount
    
    
    
        Underlying data reference counter 
    
    
    
    .. attribute:: data
    
    
    
        Pointers to the actual matrix data 
    
    
    
    .. attribute:: dim
    
    
    
        For each dimension, the pair (number of elements, distance between elements in bytes) 
    
    
    

.. index:: CvSparseMat

.. _CvSparseMat:

CvSparseMat
-----------

`id=0.451492537542 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvSparseMat>`__

.. ctype:: CvSparseMat



Multi-dimensional sparse multi-channel array.




::


    
    typedef struct CvSparseMat
    {
        int type;
        int dims; 
        int* refcount; 
        struct CvSet* heap; 
        void** hashtable; 
        int hashsize;
        int total; 
        int valoffset; 
        int idxoffset; 
        int size[CV_MAX_DIM]; 
    
    } CvSparseMat;
    

..



    
    
    .. attribute:: type
    
    
    
        A CvSparseMat signature (CV _ SPARSE _ MAT _ MAGIC _ VAL), combining the type of elements and flags. 
    
    
    
    .. attribute:: dims
    
    
    
        Number of dimensions 
    
    
    
    .. attribute:: refcount
    
    
    
        Underlying reference counter. Not used. 
    
    
    
    .. attribute:: heap
    
    
    
        A pool of hash table nodes 
    
    
    
    .. attribute:: hashtable
    
    
    
        The hash table. Each entry is a list of nodes. 
    
    
    
    .. attribute:: hashsize
    
    
    
        Size of the hash table 
    
    
    
    .. attribute:: total
    
    
    
        Total number of sparse array nodes 
    
    
    
    .. attribute:: valoffset
    
    
    
        The value offset of the array nodes, in bytes 
    
    
    
    .. attribute:: idxoffset
    
    
    
        The index offset of the array nodes, in bytes 
    
    
    
    .. attribute:: size
    
    
    
        Array of dimension sizes 
    
    
    

.. index:: IplImage

.. _IplImage:

IplImage
--------

`id=0.99460273838 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/IplImage>`__

.. ctype:: IplImage



IPL image header




::


    
    typedef struct _IplImage
    {
        int  nSize;         
        int  ID;            
        int  nChannels;     
        int  alphaChannel;  
        int  depth;         
        char colorModel[4]; 
        char channelSeq[4]; 
        int  dataOrder;     
        int  origin;        
        int  align;         
        int  width;         
        int  height;        
        struct _IplROI *roi; 
        struct _IplImage *maskROI; 
        void  *imageId;     
        struct _IplTileInfo *tileInfo; 
        int  imageSize;                             
        char *imageData;  
        int  widthStep;   
        int  BorderMode[4]; 
        int  BorderConst[4]; 
        char *imageDataOrigin; 
    }
    IplImage;
    

..



    
    
    .. attribute:: nSize
    
    
    
        ``sizeof(IplImage)`` 
    
    
    
    .. attribute:: ID
    
    
    
        Version, always equals 0 
    
    
    
    .. attribute:: nChannels
    
    
    
        Number of channels. Most OpenCV functions support 1-4 channels. 
    
    
    
    .. attribute:: alphaChannel
    
    
    
        Ignored by OpenCV 
    
    
    
    .. attribute:: depth
    
    
    
        Channel depth in bits + the optional sign bit ( ``IPL_DEPTH_SIGN`` ). The supported depths are: 
        
            
            .. attribute:: IPL_DEPTH_8U
            
            
            
                Unsigned 8-bit integer 
            
            
            .. attribute:: IPL_DEPTH_8S
            
            
            
                Signed 8-bit integer 
            
            
            .. attribute:: IPL_DEPTH_16U
            
            
            
                Unsigned 16-bit integer 
            
            
            .. attribute:: IPL_DEPTH_16S
            
            
            
                Signed 16-bit integer 
            
            
            .. attribute:: IPL_DEPTH_32S
            
            
            
                Signed 32-bit integer 
            
            
            .. attribute:: IPL_DEPTH_32F
            
            
            
                Single-precision floating point 
            
            
            .. attribute:: IPL_DEPTH_64F
            
            
            
                Double-precision floating point 
            
            
    
    
    
    .. attribute:: colorModel
    
    
    
        Ignored by OpenCV. The OpenCV function  :ref:`CvtColor`  requires the source and destination color spaces as parameters. 
    
    
    
    .. attribute:: channelSeq
    
    
    
        Ignored by OpenCV 
    
    
    
    .. attribute:: dataOrder
    
    
    
        0 =  ``IPL_DATA_ORDER_PIXEL``  - interleaved color channels, 1 - separate color channels.  :ref:`CreateImage`  only creates images with interleaved channels. For example, the usual layout of a color image is:  :math:`b_{00} g_{00} r_{00} b_{10} g_{10} r_{10} ...` 
    
    
    
    .. attribute:: origin
    
    
    
        0 - top-left origin, 1 - bottom-left origin (Windows bitmap style) 
    
    
    
    .. attribute:: align
    
    
    
        Alignment of image rows (4 or 8). OpenCV ignores this and uses widthStep instead. 
    
    
    
    .. attribute:: width
    
    
    
        Image width in pixels 
    
    
    
    .. attribute:: height
    
    
    
        Image height in pixels 
    
    
    
    .. attribute:: roi
    
    
    
        Region Of Interest (ROI). If not NULL, only this image region will be processed. 
    
    
    
    .. attribute:: maskROI
    
    
    
        Must be NULL in OpenCV 
    
    
    
    .. attribute:: imageId
    
    
    
        Must be NULL in OpenCV 
    
    
    
    .. attribute:: tileInfo
    
    
    
        Must be NULL in OpenCV 
    
    
    
    .. attribute:: imageSize
    
    
    
        Image data size in bytes. For interleaved data, this equals  :math:`\texttt{image->height} \cdot \texttt{image->widthStep}`   
    
    
    
    .. attribute:: imageData
    
    
    
        A pointer to the aligned image data 
    
    
    
    .. attribute:: widthStep
    
    
    
        The size of an aligned image row, in bytes 
    
    
    
    .. attribute:: BorderMode
    
    
    
        Border completion mode, ignored by OpenCV 
    
    
    
    .. attribute:: BorderConst
    
    
    
        Border completion mode, ignored by OpenCV 
    
    
    
    .. attribute:: imageDataOrigin
    
    
    
        A pointer to the origin of the image data (not necessarily aligned). This is used for image deallocation. 
    
    
    
The 
:ref:`IplImage`
structure was inherited from the Intel Image Processing Library, in which the format is native. OpenCV only supports a subset of possible 
:ref:`IplImage`
formats, as outlined in the parameter list above.

In addition to the above restrictions, OpenCV handles ROIs differently. OpenCV functions require that the image size or ROI size of all source and destination images match exactly. On the other hand, the Intel Image Processing Library processes the area of intersection between the source and destination images (or ROIs), allowing them to vary independently. 

.. index:: CvArr

.. _CvArr:

CvArr
-----

`id=0.322048506688 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/c/core/CvArr>`__

.. ctype:: CvArr



Arbitrary array




::


    
    typedef void CvArr;
    

..

The metatype 
``CvArr``
is used 
*only*
as a function parameter to specify that the function accepts arrays of multiple types, such as IplImage*, CvMat* or even CvSeq* sometimes. The particular array type is determined at runtime by analyzing the first 4 bytes of the header.
