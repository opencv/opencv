Basic C Structures and Operations
=================================

.. highlight:: c

The section describes the main data structures, used by the OpenCV 1.x API, and the basic functions to create and process the data structures.

CvPoint
-------

.. ocv:struct:: CvPoint

  2D point with integer coordinates (usually zero-based).

  .. ocv:member:: int x

     x-coordinate

  .. ocv:member:: int y

     y-coordinate

.. ocv:cfunction:: CvPoint cvPoint( int x, int y )

    constructs ``CvPoint`` structure.

.. ocv:cfunction:: CvPoint cvPointFrom32f( CvPoint2D32f point )

    converts ``CvPoint2D32f`` to ``CvPoint``.

.. seealso:: :ocv:class:`Point\_`

CvPoint2D32f
------------

.. ocv:struct:: CvPoint2D32f

  2D point with floating-point coordinates.

  .. ocv:member:: float x

     x-coordinate

  .. ocv:member:: float y

     y-coordinate

.. ocv:cfunction:: CvPoint2D32f cvPoint2D32f( double x, double y )

    constructs ``CvPoint2D32f`` structure.

.. ocv:cfunction:: CvPoint2D32f cvPointTo32f( CvPoint point )

    converts ``CvPoint`` to ``CvPoint2D32f``.

.. seealso:: :ocv:class:`Point\_`

CvPoint3D32f
------------

.. ocv:struct:: CvPoint3D32f

  3D point with floating-point coordinates

  .. ocv:member:: float x

     x-coordinate

  .. ocv:member:: float y

     y-coordinate

  .. ocv:member:: float z

     z-coordinate

.. ocv:cfunction:: CvPoint3D32f cvPoint3D32f( double x, double y, double z )

    constructs ``CvPoint3D32f`` structure.

.. seealso:: :ocv:class:`Point3\_`

CvPoint2D64f
------------

.. ocv:struct:: CvPoint2D64f

  2D point with double-precision floating-point coordinates.

  .. ocv:member:: double x

     x-coordinate

  .. ocv:member:: double y

     y-coordinate

.. ocv:cfunction:: CvPoint2D64f cvPoint2D64f( double x, double y )

    constructs ``CvPoint2D64f`` structure.

.. seealso:: :ocv:class:`Point\_`

CvPoint3D64f
------------

.. ocv:struct:: CvPoint3D64f

  3D point with double-precision floating-point coordinates.

  .. ocv:member:: double x

     x-coordinate

  .. ocv:member:: double y

     y-coordinate

  .. ocv:member:: double z

.. ocv:cfunction:: CvPoint3D64f cvPoint3D64f( double x, double y, double z )

    constructs ``CvPoint3D64f`` structure.

.. seealso:: :ocv:class:`Point3\_`

CvSize
------

.. ocv:struct:: CvSize

  Size of a rectangle or an image.

  .. ocv:member:: int width

     Width of the rectangle

  .. ocv:member:: int height

     Height of the rectangle

.. ocv:cfunction:: CvSize cvSize( int width, int height )

    constructs ``CvSize`` structure.

.. seealso:: :ocv:class:`Size\_`

CvSize2D32f
-----------

.. ocv:struct:: CvSize2D32f

  Sub-pixel accurate size of a rectangle.

  .. ocv:member:: float width

     Width of the rectangle

  .. ocv:member:: float height

     Height of the rectangle

.. ocv:cfunction:: CvSize2D32f cvSize2D32f( double width, double height )

    constructs ``CvSize2D32f`` structure.

.. seealso:: :ocv:class:`Size\_`

CvRect
------

.. ocv:struct:: CvRect

  Stores coordinates of a rectangle.

  .. ocv:member:: int x

     x-coordinate of the top-left corner

  .. ocv:member:: int y

     y-coordinate of the top-left corner (sometimes bottom-left corner)

  .. ocv:member:: int width

     Width of the rectangle

  .. ocv:member:: int height

     Height of the rectangle

.. ocv:cfunction:: CvRect cvRect( int x, int y, int width, int height )

    constructs ``CvRect`` structure.

.. seealso:: :ocv:class:`Rect\_`


CvBox2D
-------

.. ocv:struct:: CvBox2D

  Stores coordinates of a rotated rectangle.

  .. ocv:member:: CvPoint2D32f center

     Center of the box

  .. ocv:member:: CvSize2D32f  size

     Box width and height

  .. ocv:member:: float angle

     Angle between the horizontal axis and the first side (i.e. length) in degrees

.. seealso:: :ocv:class:`RotatedRect`


CvScalar
--------

.. ocv:struct:: CvScalar

  A container for 1-,2-,3- or 4-tuples of doubles.

  .. ocv:member:: double[4] val

.. ocv::cfunction:: CvScalar cvScalar( double val0, double val1=0, double val2=0, double val3=0 )

    initializes val[0] with val0, val[1] with val1, val[2] with val2 and val[3] with val3.

.. ocv::cfunction:: CvScalar cvScalarAll( double val0123 )

    initializes all of val[0]...val[3] with val0123

.. ocv::cfunction:: CvScalar cvRealScalar( double val0 )

    initializes val[0] with val0, val[1], val[2] and val[3] with 0.

.. seealso:: :ocv:class:`Scalar\_`

CvTermCriteria
--------------

.. ocv:struct:: CvTermCriteria

  Termination criteria for iterative algorithms.

  .. ocv:member:: int type

     type of the termination criteria, one of:

         * ``CV_TERMCRIT_ITER`` - stop the algorithm after ``max_iter`` iterations at maximum.

         * ``CV_TERMCRIT_EPS`` - stop the algorithm after the achieved algorithm-dependent accuracy becomes lower than ``epsilon``.

         * ``CV_TERMCRIT_ITER+CV_TERMCRIT_EPS`` - stop the algorithm after ``max_iter`` iterations or when the achieved accuracy is lower than ``epsilon``, whichever comes the earliest.

  .. ocv:member:: int max_iter

     Maximum number of iterations

  .. ocv:member:: double epsilon

     Required accuracy

.. seealso:: :ocv:class:`TermCriteria`

CvMat
-----

.. ocv:struct:: CvMat

  A multi-channel dense matrix.

  .. ocv:member:: int type

     ``CvMat`` signature (``CV_MAT_MAGIC_VAL``) plus type of the elements. Type of the matrix elements can be retrieved using ``CV_MAT_TYPE`` macro: ::

         int type = CV_MAT_TYPE(matrix->type);

     For description of possible matrix elements, see :ocv:class:`Mat`.

  .. ocv:member:: int step

     Full row length in bytes

  .. ocv:member:: int* refcount

     Underlying data reference counter

  .. ocv:member:: union data

     Pointers to the actual matrix data:

         * ptr - pointer to 8-bit unsigned elements
         * s - pointer to 16-bit signed elements
         * i - pointer to 32-bit signed elements
         * fl - pointer to 32-bit floating-point elements
         * db - pointer to 64-bit floating-point elements

  .. ocv:member:: int rows

     Number of rows

  .. ocv:member:: int cols

     Number of columns

Matrix elements are stored row by row. Element (i, j) (i - 0-based row index, j - 0-based column index) of a matrix can be retrieved or modified using ``CV_MAT_ELEM`` macro: ::

    uchar pixval = CV_MAT_ELEM(grayimg, uchar, i, j)
    CV_MAT_ELEM(cameraMatrix, float, 0, 2) = image.width*0.5f;

To access multiple-channel matrices, you can use ``CV_MAT_ELEM(matrix, type, i, j*nchannels + channel_idx)``.

``CvMat`` is now obsolete; consider using :ocv:class:`Mat` instead.

CvMatND
-------

.. ocv:struct:: CvMatND

  Multi-dimensional dense multi-channel array.

  .. ocv:member:: int type

     A ``CvMatND`` signature (``CV_MATND_MAGIC_VAL``) plus the type of elements. Type of the matrix elements can be retrieved using ``CV_MAT_TYPE`` macro: ::

          int type = CV_MAT_TYPE(ndmatrix->type);

  .. ocv:member:: int dims

     The number of array dimensions

  .. ocv:member:: int* refcount

     Underlying data reference counter

  .. ocv:member:: union data

     Pointers to the actual matrix data

         * ptr - pointer to 8-bit unsigned elements
         * s - pointer to 16-bit signed elements
         * i - pointer to 32-bit signed elements
         * fl - pointer to 32-bit floating-point elements
         * db - pointer to 64-bit floating-point elements

  .. ocv:member:: array dim

     Arrays of pairs (array size along the i-th dimension, distance between neighbor elements along i-th dimension): ::

         for(int i = 0; i < ndmatrix->dims; i++)
             printf("size[i] = %d, step[i] = %d\n", ndmatrix->dim[i].size, ndmatrix->dim[i].step);

``CvMatND`` is now obsolete; consider using :ocv:class:`Mat` instead.

CvSparseMat
-----------

.. ocv:struct:: CvSparseMat

  Multi-dimensional sparse multi-channel array.

  .. ocv:member:: int type

     A ``CvSparseMat`` signature (CV_SPARSE_MAT_MAGIC_VAL) plus the type of sparse matrix elements. Similarly to ``CvMat`` and ``CvMatND``, use ``CV_MAT_TYPE()`` to retrieve type of the elements.

  .. ocv:member:: int dims

     Number of dimensions

  .. ocv:member:: int* refcount

     Underlying reference counter. Not used.

  .. ocv:member:: CvSet* heap

     A pool of hash table nodes

  .. ocv:member:: void** hashtable

     The hash table. Each entry is a list of nodes.

  .. ocv:member:: int hashsize

     Size of the hash table

  .. ocv:member:: int[] size

     Array of dimension sizes

IplImage
--------

.. ocv:struct:: IplImage

  IPL image header

  .. ocv:member:: int nSize

     ``sizeof(IplImage)``

  .. ocv:member:: int ID

     Version, always equals 0

  .. ocv:member:: int nChannels

     Number of channels. Most OpenCV functions support 1-4 channels.

  .. ocv:member:: int alphaChannel

     Ignored by OpenCV

  .. ocv:member:: int depth

     Channel depth in bits + the optional sign bit ( ``IPL_DEPTH_SIGN`` ). The supported depths are:

         * ``IPL_DEPTH_8U`` - unsigned 8-bit integer. Equivalent to ``CV_8U`` in matrix types.
         * ``IPL_DEPTH_8S`` - signed 8-bit integer. Equivalent to ``CV_8S`` in matrix types.
         * ``IPL_DEPTH_16U`` - unsigned 16-bit integer. Equivalent to ``CV_16U`` in matrix types.
         * ``IPL_DEPTH_16S`` - signed 8-bit integer. Equivalent to ``CV_16S`` in matrix types.
         * ``IPL_DEPTH_32S`` - signed 32-bit integer. Equivalent to ``CV_32S`` in matrix types.
         * ``IPL_DEPTH_32F`` - single-precision floating-point number. Equivalent to ``CV_32F`` in matrix types.
         * ``IPL_DEPTH_64F`` - double-precision floating-point number. Equivalent to ``CV_64F`` in matrix types.

  .. ocv:member:: char[] colorModel

     Ignored by OpenCV.

  .. ocv:member:: char[] channelSeq

     Ignored by OpenCV

  .. ocv:member:: int dataOrder

     0 =  ``IPL_DATA_ORDER_PIXEL``  - interleaved color channels, 1 - separate color channels.  :ocv:cfunc:`CreateImage`  only creates images with interleaved channels. For example, the usual layout of a color image is:  :math:`b_{00} g_{00} r_{00} b_{10} g_{10} r_{10} ...`

  .. ocv:member:: int origin

     0 - top-left origin, 1 - bottom-left origin (Windows bitmap style)

  .. ocv:member:: int align

     Alignment of image rows (4 or 8). OpenCV ignores this and uses widthStep instead.

  .. ocv:member:: int width

     Image width in pixels

  .. ocv:member:: int height

     Image height in pixels

  .. ocv:member:: IplROI* roi

     Region Of Interest (ROI). If not NULL, only this image region will be processed.

  .. ocv:member:: IplImage* maskROI

     Must be NULL in OpenCV

  .. ocv:member:: void* imageId

     Must be NULL in OpenCV

  .. ocv:member:: void* tileInfo

     Must be NULL in OpenCV

  .. ocv:member:: int imageSize

     Image data size in bytes. For interleaved data, this equals  :math:`\texttt{image->height} \cdot \texttt{image->widthStep}`

  .. ocv:member:: char* imageData

     A pointer to the aligned image data. Do not assign imageData directly. Use :ocv:cfunc:`SetData`.

  .. ocv:member:: int widthStep

     The size of an aligned image row, in bytes.

  .. ocv:member:: int[] BorderMode

     Border completion mode, ignored by OpenCV

  .. ocv:member:: int[] BorderConst

     Constant border value, ignored by OpenCV

  .. ocv:member:: char* imageDataOrigin

     A pointer to the origin of the image data (not necessarily aligned). This is used for image deallocation.

The ``IplImage`` is taken from the Intel Image Processing Library, in which the format is native. OpenCV only supports a subset of possible ``IplImage`` formats, as outlined in the parameter list above.

In addition to the above restrictions, OpenCV handles ROIs differently. OpenCV functions require that the image size or ROI size of all source and destination images match exactly. On the other hand, the Intel Image Processing Library processes the area of intersection between the source and destination images (or ROIs), allowing them to vary independently.

CvArr
-----

.. ocv:struct:: CvArr

This is the "metatype" used *only* as a function parameter. It denotes that the function accepts arrays of multiple types, such as IplImage*, CvMat* or even CvSeq* sometimes. The particular array type is determined at runtime by analyzing the first 4 bytes of the header. In C++ interface the role of ``CvArr`` is played by ``InputArray`` and ``OutputArray``.

ClearND
-------
Clears a specific array element.

.. ocv:cfunction:: void cvClearND( CvArr* arr, const int* idx )

.. ocv:pyoldfunction:: cv.ClearND(arr, idx)-> None

    :param arr: Input array
    :param idx: Array of the element indices

The function clears (sets to zero) a specific element of a dense array or deletes the element of a sparse array. If the sparse array element does not exists, the function does nothing.

CloneImage
----------
Makes a full copy of an image, including the header, data, and ROI.

.. ocv:cfunction:: IplImage* cvCloneImage(const IplImage* image)
.. ocv:pyoldfunction:: cv.CloneImage(image) -> image

    :param image: The original image

CloneMat
--------
Creates a full matrix copy.

.. ocv:cfunction:: CvMat* cvCloneMat(const CvMat* mat)
.. ocv:pyoldfunction:: cv.CloneMat(mat) -> mat

    :param mat: Matrix to be copied

Creates a full copy of a matrix and returns a pointer to the copy. Note that the matrix copy is compacted, that is, it will not have gaps between rows.

CloneMatND
----------
Creates full copy of a multi-dimensional array and returns a pointer to the copy.

.. ocv:cfunction:: CvMatND* cvCloneMatND(const CvMatND* mat)
.. ocv:pyoldfunction:: cv.CloneMatND(mat) -> matND

    :param mat: Input array

CloneSparseMat
--------------
Creates full copy of sparse array.

.. ocv:cfunction:: CvSparseMat* cvCloneSparseMat(const CvSparseMat* mat)

    :param mat: Input array

The function creates a copy of the input array and returns pointer to the copy.


ConvertScale
------------
Converts one array to another with optional linear transformation.

.. ocv:cfunction:: void cvConvertScale(const CvArr* src, CvArr* dst, double scale=1, double shift=0)
.. ocv:pyoldfunction:: cv.ConvertScale(src, dst, scale=1.0, shift=0.0)-> None
.. ocv:pyoldfunction:: cv.Convert(src, dst)-> None

    ::

        #define cvCvtScale cvConvertScale
        #define cvScale  cvConvertScale
        #define cvConvert(src, dst )  cvConvertScale((src), (dst), 1, 0 )

    ..

    :param src: Source array

    :param dst: Destination array

    :param scale: Scale factor

    :param shift: Value added to the scaled source array elements

The function has several different purposes, and thus has several different names. It copies one array to another with optional scaling, which is performed first, and/or optional type conversion, performed after:

.. math::

    \texttt{dst} (I) =  \texttt{scale} \texttt{src} (I) + ( \texttt{shift} _0, \texttt{shift} _1,...)


All the channels of multi-channel arrays are processed independently.

The type of conversion is done with rounding and saturation, that is if the
result of scaling + conversion can not be represented exactly by a value
of the destination array element type, it is set to the nearest representable
value on the real axis.


Copy
----
Copies one array to another.

.. ocv:cfunction:: void cvCopy(const CvArr* src, CvArr* dst, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.Copy(src, dst, mask=None)-> None

    :param src: The source array

    :param dst: The destination array

    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed

The function copies selected elements from an input array to an output array:

.. math::

    \texttt{dst} (I)= \texttt{src} (I)  \quad \text{if} \quad \texttt{mask} (I)  \ne 0.

If any of the passed arrays is of ``IplImage`` type, then its ROI and COI fields are used. Both arrays must have the same type, the same number of dimensions, and the same size. The function can also copy sparse arrays (mask is not supported in this case).


CreateData
----------
Allocates array data

.. ocv:cfunction:: void cvCreateData(CvArr* arr)
.. ocv:pyoldfunction:: cv.CreateData(arr) -> None

    :param arr: Array header

The function allocates image, matrix or multi-dimensional dense array data. Note that in the case of matrix types OpenCV allocation functions are used. In the case of IplImage they are used
unless ``CV_TURN_ON_IPL_COMPATIBILITY()`` has been called before. In the latter case IPL functions are used to allocate the data.

CreateImage
-----------
Creates an image header and allocates the image data.

.. ocv:cfunction:: IplImage* cvCreateImage(CvSize size, int depth, int channels)
.. ocv:pyoldfunction:: cv.CreateImage(size, depth, channels)->image

    :param size: Image width and height

    :param depth: Bit depth of image elements. See  :ocv:struct:`IplImage`  for valid depths.

    :param channels: Number of channels per pixel. See  :ocv:struct:`IplImage`  for details. This function only creates images with interleaved channels.

This function call is equivalent to the following code: ::

    header = cvCreateImageHeader(size, depth, channels);
    cvCreateData(header);

CreateImageHeader
-----------------
Creates an image header but does not allocate the image data.

.. ocv:cfunction:: IplImage* cvCreateImageHeader(CvSize size, int depth, int channels)
.. ocv:pyoldfunction:: cv.CreateImageHeader(size, depth, channels) -> image

    :param size: Image width and height

    :param depth: Image depth (see  :ocv:cfunc:`CreateImage` )

    :param channels: Number of channels (see  :ocv:cfunc:`CreateImage` )

CreateMat
---------
Creates a matrix header and allocates the matrix data.

.. ocv:cfunction:: CvMat* cvCreateMat( int rows, int cols, int type)
.. ocv:pyoldfunction:: cv.CreateMat(rows, cols, type) -> mat

    :param rows: Number of rows in the matrix

    :param cols: Number of columns in the matrix

    :param type: The type of the matrix elements in the form  ``CV_<bit depth><S|U|F>C<number of channels>`` , where S=signed, U=unsigned, F=float. For example, CV _ 8UC1 means the elements are 8-bit unsigned and the there is 1 channel, and CV _ 32SC2 means the elements are 32-bit signed and there are 2 channels.

The function call is equivalent to the following code: ::

    CvMat* mat = cvCreateMatHeader(rows, cols, type);
    cvCreateData(mat);

CreateMatHeader
---------------
Creates a matrix header but does not allocate the matrix data.

.. ocv:cfunction:: CvMat* cvCreateMatHeader( int rows, int cols, int type)
.. ocv:pyoldfunction:: cv.CreateMatHeader(rows, cols, type) -> mat

    :param rows: Number of rows in the matrix

    :param cols: Number of columns in the matrix

    :param type: Type of the matrix elements, see  :ocv:cfunc:`CreateMat`

The function allocates a new matrix header and returns a pointer to it. The matrix data can then be allocated using :ocv:cfunc:`CreateData` or set explicitly to user-allocated data via :ocv:cfunc:`SetData`.

CreateMatND
-----------
Creates the header and allocates the data for a multi-dimensional dense array.

.. ocv:cfunction:: CvMatND* cvCreateMatND( int dims, const int* sizes, int type)
.. ocv:pyoldfunction:: cv.CreateMatND(dims, type) -> matND

    :param dims: Number of array dimensions. This must not exceed CV_MAX_DIM (32 by default, but can be changed at build time).

    :param sizes: Array of dimension sizes.

    :param type: Type of array elements, see  :ocv:cfunc:`CreateMat` .

This function call is equivalent to the following code: ::

    CvMatND* mat = cvCreateMatNDHeader(dims, sizes, type);
    cvCreateData(mat);

CreateMatNDHeader
-----------------
Creates a new matrix header but does not allocate the matrix data.

.. ocv:cfunction:: CvMatND* cvCreateMatNDHeader( int dims, const int* sizes, int type)
.. ocv:pyoldfunction:: cv.CreateMatNDHeader(dims, type) -> matND

    :param dims: Number of array dimensions

    :param sizes: Array of dimension sizes

    :param type: Type of array elements, see  :ocv:cfunc:`CreateMat`

The function allocates a header for a multi-dimensional dense array. The array data can further be allocated using  :ocv:cfunc:`CreateData` or set explicitly to user-allocated data via  :ocv:cfunc:`SetData`.

CreateSparseMat
---------------
Creates sparse array.

.. ocv:cfunction:: CvSparseMat* cvCreateSparseMat(int dims, const int* sizes, int type)

    :param dims: Number of array dimensions. In contrast to the dense matrix, the number of dimensions is practically unlimited (up to  :math:`2^{16}` ).

    :param sizes: Array of dimension sizes

    :param type: Type of array elements. The same as for CvMat

The function allocates a multi-dimensional sparse array. Initially the array contain no elements, that is
:ocv:cfunc:`PtrND` and other related functions will return 0 for every index.


CrossProduct
------------
Calculates the cross product of two 3D vectors.

.. ocv:cfunction:: void cvCrossProduct(const CvArr* src1, const CvArr* src2, CvArr* dst)
.. ocv:pyoldfunction:: cv.CrossProduct(src1, src2, dst)-> None

    :param src1: The first source vector

    :param src2: The second source vector

    :param dst: The destination vector

The function calculates the cross product of two 3D vectors:

.. math::

    \texttt{dst} =  \texttt{src1} \times \texttt{src2}

or:

.. math::

    \begin{array}{l} \texttt{dst} _1 =  \texttt{src1} _2  \texttt{src2} _3 -  \texttt{src1} _3  \texttt{src2} _2 \\ \texttt{dst} _2 =  \texttt{src1} _3  \texttt{src2} _1 -  \texttt{src1} _1  \texttt{src2} _3 \\ \texttt{dst} _3 =  \texttt{src1} _1  \texttt{src2} _2 -  \texttt{src1} _2  \texttt{src2} _1 \end{array}


DotProduct
----------
Calculates the dot product of two arrays in Euclidean metrics.

.. ocv:cfunction:: double cvDotProduct(const CvArr* src1, const CvArr* src2)
.. ocv:pyoldfunction:: cv.DotProduct(src1, src2) -> float

    :param src1: The first source array

    :param src2: The second source array

The function calculates and returns the Euclidean dot product of two arrays.

.. math::

    src1  \bullet src2 =  \sum _I ( \texttt{src1} (I)  \texttt{src2} (I))

In the case of multiple channel arrays, the results for all channels are accumulated. In particular,
``cvDotProduct(a,a)`` where  ``a`` is a complex vector, will return  :math:`||\texttt{a}||^2`.
The function can process multi-dimensional arrays, row by row, layer by layer, and so on.


Get?D
-----

.. ocv:cfunction:: CvScalar cvGet1D(const CvArr* arr, int idx0)
.. ocv:cfunction:: CvScalar cvGet2D(const CvArr* arr, int idx0, int idx1)
.. ocv:cfunction:: CvScalar cvGet3D(const CvArr* arr, int idx0, int idx1, int idx2)
.. ocv:cfunction:: CvScalar cvGetND( const CvArr* arr, const int* idx )

.. ocv:pyoldfunction:: cv.Get1D(arr, idx) -> scalar
.. ocv:pyoldfunction:: cv.Get2D(arr, idx0, idx1) -> scalar
.. ocv:pyoldfunction:: cv.Get3D(arr, idx0, idx1, idx2) -> scalar
.. ocv:pyoldfunction:: cv.GetND(arr, indices) -> scalar

    Return a specific array element.

    :param arr: Input array

    :param idx0: The first zero-based component of the element index

    :param idx1: The second zero-based component of the element index

    :param idx2: The third zero-based component of the element index

    :param idx: Array of the element indices

The functions return a specific array element. In the case of a sparse array the functions return 0 if the requested node does not exist (no new node is created by the functions).

GetCol(s)
---------
Returns one of more array columns.

.. ocv:cfunction:: CvMat* cvGetCol(const CvArr* arr, CvMat* submat, int col)

.. ocv:cfunction:: CvMat* cvGetCols( const CvArr* arr, CvMat* submat, int start_col, int end_col )

.. ocv:pyoldfunction:: cv.GetCol(arr, col)-> submat

.. ocv:pyoldfunction:: cv.GetCols(arr, startCol, endCol)-> submat

    :param arr: Input array

    :param submat: Pointer to the resulting sub-array header

    :param col: Zero-based index of the selected column

    :param start_col: Zero-based index of the starting column (inclusive) of the span

    :param end_col: Zero-based index of the ending column (exclusive) of the span

The functions return the header, corresponding to a specified column span of the input array. That is, no data is copied. Therefore, any modifications of the submatrix will affect the original array. If you need to copy the columns, use :ocv:cfunc:`CloneMat`. ``cvGetCol(arr, submat, col)`` is a shortcut for ``cvGetCols(arr, submat, col, col+1)``.

GetDiag
-------
Returns one of array diagonals.

.. ocv:cfunction:: CvMat* cvGetDiag(const CvArr* arr, CvMat* submat, int diag=0)
.. ocv:pyoldfunction:: cv.GetDiag(arr, diag=0)-> submat

    :param arr: Input array

    :param submat: Pointer to the resulting sub-array header

    :param diag: Index of the array diagonal. Zero value corresponds to the main diagonal, -1 corresponds to the diagonal above the main, 1 corresponds to the diagonal below the main, and so forth.

The function returns the header, corresponding to a specified diagonal of the input array.

GetDims
---------
Return number of array dimensions

.. ocv:cfunction:: int cvGetDims(const CvArr* arr, int* sizes=NULL)
.. ocv:pyoldfunction:: cv.GetDims(arr) -> (dim1, dim2, ...)

    :param arr: Input array

    :param sizes: Optional output vector of the array dimension sizes. For
        2d arrays the number of rows (height) goes first, number of columns
        (width) next.

The function returns the array dimensionality and the array of dimension sizes. In the case of  ``IplImage`` or `CvMat` it always returns 2 regardless of number of image/matrix rows. For example, the following code calculates total number of array elements: ::

    int sizes[CV_MAX_DIM];
    int i, total = 1;
    int dims = cvGetDims(arr, size);
    for(i = 0; i < dims; i++ )
        total *= sizes[i];

GetDimSize
------------
Returns array size along the specified dimension.

.. ocv:cfunction:: int cvGetDimSize(const CvArr* arr, int index)

    :param arr: Input array

    :param index: Zero-based dimension index (for matrices 0 means number of rows, 1 means number of columns; for images 0 means height, 1 means width)

GetElemType
-----------
Returns type of array elements.

.. ocv:cfunction:: int cvGetElemType(const CvArr* arr)
.. ocv:pyoldfunction:: cv.GetElemType(arr)-> int

    :param arr: Input array

The function returns type of the array elements. In the case of ``IplImage`` the type is converted to ``CvMat``-like representation. For example, if the image has been created as: ::

    IplImage* img = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);

The code ``cvGetElemType(img)`` will return ``CV_8UC3``.

GetImage
--------
Returns image header for arbitrary array.

.. ocv:cfunction:: IplImage* cvGetImage( const CvArr* arr, IplImage* image_header )

.. ocv:pyoldfunction:: cv.GetImage(arr) -> iplimage

    :param arr: Input array

    :param image_header: Pointer to  ``IplImage``  structure used as a temporary buffer

The function returns the image header for the input array that can be a matrix (:ocv:struct:`CvMat`) or image (:ocv:struct:`IplImage`). In the case of an image the function simply returns the input pointer. In the case of ``CvMat`` it initializes an ``image_header`` structure with the parameters of the input matrix. Note that if we transform ``IplImage`` to ``CvMat`` using :ocv:cfunc:`GetMat` and then transform ``CvMat`` back to IplImage using this function, we will get different headers if the ROI is set in the original image.

GetImageCOI
-----------
Returns the index of the channel of interest.

.. ocv:cfunction:: int cvGetImageCOI(const IplImage* image)
.. ocv:pyoldfunction:: cv.GetImageCOI(image) -> int

    :param image: A pointer to the image header

Returns the channel of interest of in an IplImage. Returned values correspond to the ``coi`` in
:ocv:cfunc:`SetImageCOI`.

GetImageROI
-----------
Returns the image ROI.

.. ocv:cfunction:: CvRect cvGetImageROI(const IplImage* image)
.. ocv:pyoldfunction:: cv.GetImageROI(image)-> CvRect

    :param image: A pointer to the image header

If there is no ROI set, ``cvRect(0,0,image->width,image->height)`` is returned.

GetMat
------
Returns matrix header for arbitrary array.

.. ocv:cfunction:: CvMat* cvGetMat(const CvArr* arr, CvMat* header, int* coi=NULL, int allowND=0)
.. ocv:pyoldfunction:: cv.GetMat(arr, allowND=0) -> mat

    :param arr: Input array

    :param header: Pointer to  :ocv:struct:`CvMat`  structure used as a temporary buffer

    :param coi: Optional output parameter for storing COI

    :param allowND: If non-zero, the function accepts multi-dimensional dense arrays (CvMatND*) and returns 2D matrix (if CvMatND has two dimensions) or 1D matrix (when CvMatND has 1 dimension or more than 2 dimensions). The ``CvMatND`` array must be continuous.

The function returns a matrix header for the input array that can be a matrix - :ocv:struct:`CvMat`, an image - :ocv:struct:`IplImage`, or a multi-dimensional dense array - :ocv:struct:`CvMatND` (the third option is allowed only if ``allowND != 0``) . In the case of matrix the function simply returns the input pointer. In the case of ``IplImage*`` or ``CvMatND`` it initializes the ``header`` structure with parameters of the current image ROI and returns ``&header``. Because COI is not supported by ``CvMat``, it is returned separately.

The function provides an easy way to handle both types of arrays - ``IplImage`` and  ``CvMat`` using the same code. Input array must have non-zero data pointer, otherwise the function will report an error.

.. seealso:: :ocv:cfunc:`GetImage`, :ocv:func:`cvarrToMat`.

.. note:: If the input array is ``IplImage`` with planar data layout and COI set, the function returns the pointer to the selected plane and ``COI == 0``. This feature allows user to process ``IplImage`` structures with planar data layout, even though OpenCV does not support such images.

GetNextSparseNode
-----------------
Returns the next sparse matrix element

.. ocv:cfunction:: CvSparseNode* cvGetNextSparseNode( CvSparseMatIterator* mat_iterator )

    :param mat_iterator: Sparse array iterator

The function moves iterator to the next sparse matrix element and returns pointer to it. In the current version there is no any particular order of the elements, because they are stored in the hash table. The sample below demonstrates how to iterate through the sparse matrix: ::

    // print all the non-zero sparse matrix elements and compute their sum
    double sum = 0;
    int i, dims = cvGetDims(sparsemat);
    CvSparseMatIterator it;
    CvSparseNode* node = cvInitSparseMatIterator(sparsemat, &it);

    for(; node != 0; node = cvGetNextSparseNode(&it))
    {
        /* get pointer to the element indices */
        int* idx = CV_NODE_IDX(array, node);
        /* get value of the element (assume that the type is CV_32FC1) */
        float val = *(float*)CV_NODE_VAL(array, node);
        printf("M");
        for(i = 0; i < dims; i++ )
            printf("[%d]", idx[i]);
        printf("=%g\n", val);

        sum += val;
    }

    printf("nTotal sum = %g\n", sum);


GetRawData
----------
Retrieves low-level information about the array.

.. ocv:cfunction:: void cvGetRawData( const CvArr* arr, uchar** data, int* step=NULL, CvSize* roi_size=NULL )

    :param arr: Array header

    :param data: Output pointer to the whole image origin or ROI origin if ROI is set

    :param step: Output full row length in bytes

    :param roi_size: Output ROI size

The function fills output variables with low-level information about the array data. All output parameters are optional, so some of the pointers may be set to ``NULL``. If the array is ``IplImage`` with ROI set, the parameters of ROI are returned.

The following example shows how to get access to array elements. It computes absolute values of the array elements ::

    float* data;
    int step;
    CvSize size;

    cvGetRawData(array, (uchar**)&data, &step, &size);
    step /= sizeof(data[0]);

    for(int y = 0; y < size.height; y++, data += step )
        for(int x = 0; x < size.width; x++ )
            data[x] = (float)fabs(data[x]);

GetReal?D
---------
Return a specific element of single-channel 1D, 2D, 3D or nD array.

.. ocv:cfunction:: double cvGetReal1D(const CvArr* arr, int idx0)
.. ocv:cfunction:: double cvGetReal2D(const CvArr* arr, int idx0, int idx1)
.. ocv:cfunction:: double cvGetReal3D(const CvArr* arr, int idx0, int idx1, int idx2)
.. ocv:cfunction:: double cvGetRealND( const CvArr* arr, const int* idx )

.. ocv:pyoldfunction:: cv.GetReal1D(arr, idx0)->float
.. ocv:pyoldfunction:: cv.GetReal2D(arr, idx0, idx1)->float
.. ocv:pyoldfunction:: cv.GetReal3D(arr, idx0, idx1, idx2)->float
.. ocv:pyoldfunction:: cv.GetRealND(arr, idx)->float

    :param arr: Input array. Must have a single channel.

    :param idx0: The first zero-based component of the element index

    :param idx1: The second zero-based component of the element index

    :param idx2: The third zero-based component of the element index

    :param idx: Array of the element indices

Returns a specific element of a single-channel array. If the array has multiple channels, a runtime error is raised. Note that ``Get?D`` functions can be used safely for both single-channel and multiple-channel arrays though they are a bit slower.

In the case of a sparse array the functions return 0 if the requested node does not exist (no new node is created by the functions).


GetRow(s)
---------
Returns array row or row span.

.. ocv:cfunction:: CvMat* cvGetRow(const CvArr* arr, CvMat* submat, int row)

.. ocv:cfunction:: CvMat* cvGetRows( const CvArr* arr, CvMat* submat, int start_row, int end_row, int delta_row=1 )

.. ocv:pyoldfunction:: cv.GetRow(arr, row)-> submat
.. ocv:pyoldfunction:: cv.GetRows(arr, startRow, endRow, deltaRow=1)-> submat

    :param arr: Input array

    :param submat: Pointer to the resulting sub-array header

    :param row: Zero-based index of the selected row

    :param start_row: Zero-based index of the starting row (inclusive) of the span

    :param end_row: Zero-based index of the ending row (exclusive) of the span

    :param delta_row: Index step in the row span. That is, the function extracts every  ``delta_row`` -th row from  ``start_row``  and up to (but not including)  ``end_row`` .

The functions return the header, corresponding to a specified row/row span of the input array. ``cvGetRow(arr, submat, row)`` is a shortcut for ``cvGetRows(arr, submat, row, row+1)``.


GetSize
-------
Returns size of matrix or image ROI.

.. ocv:cfunction:: CvSize cvGetSize(const CvArr* arr)
.. ocv:pyoldfunction:: cv.GetSize(arr)-> (width, height)

    :param arr: array header

The function returns number of rows (CvSize::height) and number of columns (CvSize::width) of the input matrix or image. In the case of image the size of ROI is returned.

GetSubRect
----------
Returns matrix header corresponding to the rectangular sub-array of input image or matrix.

.. ocv:cfunction:: CvMat* cvGetSubRect(const CvArr* arr, CvMat* submat, CvRect rect)
.. ocv:pyoldfunction:: cv.GetSubRect(arr, rect) -> submat

    :param arr: Input array

    :param submat: Pointer to the resultant sub-array header

    :param rect: Zero-based coordinates of the rectangle of interest

The function returns header, corresponding to a specified rectangle of the input array. In other words, it allows the user to treat a rectangular part of input array as a stand-alone array. ROI is taken into account by the function so the sub-array of ROI is actually extracted.

DecRefData
----------
Decrements an array data reference counter.

.. ocv:cfunction:: void cvDecRefData(CvArr* arr)

    :param arr: Pointer to an array header

The function decrements the data reference counter in a :ocv:struct:`CvMat` or :ocv:struct:`CvMatND` if the reference counter pointer is not NULL. If the counter reaches zero, the data is deallocated. In the current implementation the reference counter is not NULL only if the data was allocated using the  :ocv:cfunc:`CreateData` function. The counter will be NULL in other cases such as: external data was assigned to the header using :ocv:cfunc:`SetData`, header is part of a larger matrix or image, or the header was converted from an image or n-dimensional matrix header.


IncRefData
----------
Increments array data reference counter.

.. ocv:cfunction:: int cvIncRefData(CvArr* arr)

    :param arr: Array header

The function increments :ocv:struct:`CvMat` or :ocv:struct:`CvMatND` data reference counter and returns the new counter value if the reference counter pointer is not NULL, otherwise it returns zero.


InitImageHeader
---------------
Initializes an image header that was previously allocated.

.. ocv:cfunction:: IplImage* cvInitImageHeader( IplImage* image, CvSize size, int depth, int channels, int origin=0, int align=4)

    :param image: Image header to initialize

    :param size: Image width and height

    :param depth: Image depth (see  :ocv:cfunc:`CreateImage` )

    :param channels: Number of channels (see  :ocv:cfunc:`CreateImage` )

    :param origin: Top-left  ``IPL_ORIGIN_TL``  or bottom-left  ``IPL_ORIGIN_BL``

    :param align: Alignment for image rows, typically 4 or 8 bytes

The returned ``IplImage*`` points to the initialized header.


InitMatHeader
-------------
Initializes a pre-allocated matrix header.

.. ocv:cfunction:: CvMat* cvInitMatHeader( CvMat* mat, int rows, int cols, int type,  void* data=NULL, int step=CV_AUTOSTEP)

    :param mat: A pointer to the matrix header to be initialized

    :param rows: Number of rows in the matrix

    :param cols: Number of columns in the matrix

    :param type: Type of the matrix elements, see  :ocv:cfunc:`CreateMat` .

    :param data: Optional: data pointer assigned to the matrix header

    :param step: Optional: full row width in bytes of the assigned data. By default, the minimal possible step is used which assumes there are no gaps between subsequent rows of the matrix.

This function is often used to process raw data with OpenCV matrix functions. For example, the following code computes the matrix product of two matrices, stored as ordinary arrays: ::

    double a[] = { 1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 10, 11, 12 };

    double b[] = { 1, 5, 9,
                   2, 6, 10,
                   3, 7, 11,
                   4, 8, 12 };

    double c[9];
    CvMat Ma, Mb, Mc ;

    cvInitMatHeader(&Ma, 3, 4, CV_64FC1, a);
    cvInitMatHeader(&Mb, 4, 3, CV_64FC1, b);
    cvInitMatHeader(&Mc, 3, 3, CV_64FC1, c);

    cvMatMulAdd(&Ma, &Mb, 0, &Mc);
    // the c array now contains the product of a (3x4) and b (4x3)


InitMatNDHeader
---------------
Initializes a pre-allocated multi-dimensional array header.

.. ocv:cfunction:: CvMatND* cvInitMatNDHeader( CvMatND* mat, int dims, const int* sizes, int type, void* data=NULL)

    :param mat: A pointer to the array header to be initialized

    :param dims: The number of array dimensions

    :param sizes: An array of dimension sizes

    :param type: Type of array elements, see  :ocv:cfunc:`CreateMat`

    :param data: Optional data pointer assigned to the matrix header


InitSparseMatIterator
---------------------
Initializes sparse array elements iterator.

.. ocv:cfunction:: CvSparseNode* cvInitSparseMatIterator( const CvSparseMat* mat, CvSparseMatIterator* mat_iterator )

    :param mat: Input array

    :param mat_iterator: Initialized iterator

The function initializes iterator of sparse array elements and returns pointer to the first element, or NULL if the array is empty.


Mat
---
Initializes matrix header (lightweight variant).

.. ocv:cfunction:: CvMat cvMat( int rows, int cols, int type, void* data=NULL)

    :param rows: Number of rows in the matrix

    :param cols: Number of columns in the matrix

    :param type: Type of the matrix elements - see  :ocv:cfunc:`CreateMat`

    :param data: Optional data pointer assigned to the matrix header

Initializes a matrix header and assigns data to it. The matrix is filled *row*-wise (the first ``cols`` elements of data form the first row of the matrix, etc.)

This function is a fast inline substitution for :ocv:cfunc:`InitMatHeader`. Namely, it is equivalent to: ::

    CvMat mat;
    cvInitMatHeader(&mat, rows, cols, type, data, CV_AUTOSTEP);


Ptr?D
-----
Return pointer to a particular array element.

.. ocv:cfunction:: uchar* cvPtr1D(const CvArr* arr, int idx0, int* type=NULL)

.. ocv:cfunction:: uchar* cvPtr2D(const CvArr* arr, int idx0, int idx1, int* type=NULL)

.. ocv:cfunction:: uchar* cvPtr3D(const CvArr* arr, int idx0, int idx1, int idx2, int* type=NULL)

.. ocv:cfunction:: uchar* cvPtrND( const CvArr* arr, const int* idx, int* type=NULL, int create_node=1, unsigned* precalc_hashval=NULL )

    :param arr: Input array

    :param idx0: The first zero-based component of the element index

    :param idx1: The second zero-based component of the element index

    :param idx2: The third zero-based component of the element index

    :param idx: Array of the element indices

    :param type: Optional output parameter: type of matrix elements

    :param create_node: Optional input parameter for sparse matrices. Non-zero value of the parameter means that the requested element is created if it does not exist already.

    :param precalc_hashval: Optional input parameter for sparse matrices. If the pointer is not NULL, the function does not recalculate the node hash value, but takes it from the specified location. It is useful for speeding up pair-wise operations (TODO: provide an example)

The functions return a pointer to a specific array element. Number of array dimension should match to the number of indices passed to the function except for ``cvPtr1D`` function that can be used for sequential access to 1D, 2D or nD dense arrays.

The functions can be used for sparse arrays as well - if the requested node does not exist they create it and set it to zero.

All these as well as other functions accessing array elements (
:ocv:cfunc:`GetND`
,
:ocv:cfunc:`GetRealND`
,
:ocv:cfunc:`Set`
,
:ocv:cfunc:`SetND`
,
:ocv:cfunc:`SetRealND`
) raise an error in case if the element index is out of range.


ReleaseData
-----------
Releases array data.

.. ocv:cfunction:: void cvReleaseData(CvArr* arr)

    :param arr: Array header

The function releases the array data. In the case of
:ocv:struct:`CvMat`
or
:ocv:struct:`CvMatND`
it simply calls cvDecRefData(), that is the function can not deallocate external data. See also the note to
:ocv:cfunc:`CreateData`
.


ReleaseImage
------------
Deallocates the image header and the image data.

.. ocv:cfunction:: void cvReleaseImage(IplImage** image)

    :param image: Double pointer to the image header

This call is a shortened form of ::

    if(*image )
    {
        cvReleaseData(*image);
        cvReleaseImageHeader(image);
    }

..

ReleaseImageHeader
------------------
Deallocates an image header.

.. ocv:cfunction:: void cvReleaseImageHeader(IplImage** image)

    :param image: Double pointer to the image header

This call is an analogue of ::

    if(image )
    {
        iplDeallocate(*image, IPL_IMAGE_HEADER | IPL_IMAGE_ROI);
        *image = 0;
    }

..

but it does not use IPL functions by default (see the ``CV_TURN_ON_IPL_COMPATIBILITY`` macro).


ReleaseMat
----------
Deallocates a matrix.

.. ocv:cfunction:: void cvReleaseMat(CvMat** mat)

    :param mat: Double pointer to the matrix

The function decrements the matrix data reference counter and deallocates matrix header. If the data reference counter is 0, it also deallocates the data. ::

    if(*mat )
        cvDecRefData(*mat);
    cvFree((void**)mat);

..

ReleaseMatND
------------
Deallocates a multi-dimensional array.

.. ocv:cfunction:: void cvReleaseMatND(CvMatND** mat)

    :param mat: Double pointer to the array

The function decrements the array data reference counter and releases the array header. If the reference counter reaches 0, it also deallocates the data. ::

    if(*mat )
        cvDecRefData(*mat);
    cvFree((void**)mat);

..

ReleaseSparseMat
----------------
Deallocates sparse array.

.. ocv:cfunction:: void cvReleaseSparseMat(CvSparseMat** mat)

    :param mat: Double pointer to the array

The function releases the sparse array and clears the array pointer upon exit.

ResetImageROI
-------------
Resets the image ROI to include the entire image and releases the ROI structure.

.. ocv:cfunction:: void cvResetImageROI(IplImage* image)
.. ocv:pyoldfunction:: cv.ResetImageROI(image)-> None

    :param image: A pointer to the image header

This produces a similar result to the following, but in addition it releases the ROI structure. ::

    cvSetImageROI(image, cvRect(0, 0, image->width, image->height ));
    cvSetImageCOI(image, 0);

..

Reshape
-------
Changes shape of matrix/image without copying data.

.. ocv:cfunction:: CvMat* cvReshape( const CvArr* arr, CvMat* header, int new_cn, int new_rows=0 )

.. ocv:pyoldfunction:: cv.Reshape(arr, newCn, newRows=0) -> mat

    :param arr: Input array

    :param header: Output header to be filled

    :param new_cn: New number of channels. 'new_cn = 0' means that the number of channels remains unchanged.

    :param new_rows: New number of rows. 'new_rows = 0' means that the number of rows remains unchanged unless it needs to be changed according to  ``new_cn``  value.

The function initializes the CvMat header so that it points to the same data as the original array but has a different shape - different number of channels, different number of rows, or both.

The following example code creates one image buffer and two image headers, the first is for a 320x240x3 image and the second is for a 960x240x1 image: ::

    IplImage* color_img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    CvMat gray_mat_hdr;
    IplImage gray_img_hdr, *gray_img;
    cvReshape(color_img, &gray_mat_hdr, 1);
    gray_img = cvGetImage(&gray_mat_hdr, &gray_img_hdr);

..

And the next example converts a 3x3 matrix to a single 1x9 vector:

::

    CvMat* mat = cvCreateMat(3, 3, CV_32F);
    CvMat row_header, *row;
    row = cvReshape(mat, &row_header, 0, 1);

..

ReshapeMatND
------------
Changes the shape of a multi-dimensional array without copying the data.

.. ocv:cfunction:: CvArr* cvReshapeMatND( const CvArr* arr, int sizeof_header, CvArr* header, int new_cn, int new_dims, int* new_sizes )

.. ocv:pyoldfunction:: cv.ReshapeMatND(arr, newCn, newDims) -> mat

    :param arr: Input array

    :param sizeof_header: Size of output header to distinguish between IplImage, CvMat and CvMatND output headers

    :param header: Output header to be filled

    :param new_cn: New number of channels. ``new_cn = 0``  means that the number of channels remains unchanged.

    :param new_dims: New number of dimensions. ``new_dims = 0`` means that the number of dimensions remains the same.

    :param new_sizes: Array of new dimension sizes. Only  ``new_dims-1``  values are used, because the total number of elements must remain the same. Thus, if  ``new_dims = 1``,  ``new_sizes``  array is not used.

The function is an advanced version of :ocv:cfunc:`Reshape` that can work with multi-dimensional arrays as well (though it can work with ordinary images and matrices) and change the number of dimensions.

Below are the two samples from the
:ocv:cfunc:`Reshape`
description rewritten using
:ocv:cfunc:`ReshapeMatND`
: ::

    IplImage* color_img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    IplImage gray_img_hdr, *gray_img;
    gray_img = (IplImage*)cvReshapeND(color_img, &gray_img_hdr, 1, 0, 0);

    ...

    /* second example is modified to convert 2x2x2 array to 8x1 vector */
    int size[] = { 2, 2, 2 };
    CvMatND* mat = cvCreateMatND(3, size, CV_32F);
    CvMat row_header, *row;
    row = (CvMat*)cvReshapeND(mat, &row_header, 0, 1, 0);

..

Set
---
Sets every element of an array to a given value.

.. ocv:cfunction:: void cvSet(CvArr* arr, CvScalar value, const CvArr* mask=NULL)
.. ocv:pyoldfunction:: cv.Set(arr, value, mask=None)-> None

    :param arr: The destination array

    :param value: Fill value

    :param mask: Operation mask, 8-bit single channel array; specifies elements of the destination array to be changed

The function copies the scalar ``value`` to every selected element of the destination array:

.. math::

    \texttt{arr} (I)= \texttt{value} \quad \text{if} \quad \texttt{mask} (I)  \ne 0

If array ``arr`` is of ``IplImage`` type, then is ROI used, but COI must not be set.

Set?D
-----
Change the particular array element.

.. ocv:cfunction:: void cvSet1D(CvArr* arr, int idx0, CvScalar value)

.. ocv:cfunction:: void cvSet2D(CvArr* arr, int idx0, int idx1, CvScalar value)

.. ocv:cfunction:: void cvSet3D(CvArr* arr, int idx0, int idx1, int idx2, CvScalar value)

.. ocv:cfunction:: void cvSetND( CvArr* arr, const int* idx, CvScalar value )

.. ocv:pyoldfunction:: cv.Set1D(arr, idx, value) -> None
.. ocv:pyoldfunction:: cv.Set2D(arr, idx0, idx1, value) -> None
.. ocv:pyoldfunction:: cv.Set3D(arr, idx0, idx1, idx2, value) -> None
.. ocv:pyoldfunction:: cv.SetND(arr, indices, value) -> None


    :param arr: Input array

    :param idx0: The first zero-based component of the element index

    :param idx1: The second zero-based component of the element index

    :param idx2: The third zero-based component of the element index

    :param idx: Array of the element indices

    :param value: The assigned value

The functions assign the new value to a particular array element. In the case of a sparse array the functions create the node if it does not exist yet.

SetData
-------
Assigns user data to the array header.

.. ocv:cfunction:: void cvSetData(CvArr* arr, void* data, int step)
.. ocv:pyoldfunction:: cv.SetData(arr, data, step)-> None

    :param arr: Array header

    :param data: User data

    :param step: Full row length in bytes

The function assigns user data to the array header. Header should be initialized before using
:ocv:cfunc:`cvCreateMatHeader`, :ocv:cfunc:`cvCreateImageHeader`, :ocv:cfunc:`cvCreateMatNDHeader`,
:ocv:cfunc:`cvInitMatHeader`, :ocv:cfunc:`cvInitImageHeader` or :ocv:cfunc:`cvInitMatNDHeader`.



SetImageCOI
-----------
Sets the channel of interest in an IplImage.

.. ocv:cfunction:: void cvSetImageCOI( IplImage* image, int coi)
.. ocv:pyoldfunction:: cv.SetImageCOI(image, coi)-> None

    :param image: A pointer to the image header

    :param coi: The channel of interest. 0 - all channels are selected, 1 - first channel is selected, etc. Note that the channel indices become 1-based.

If the ROI is set to ``NULL`` and the coi is *not* 0, the ROI is allocated. Most OpenCV functions do  *not* support the COI setting, so to process an individual image/matrix channel one may copy (via :ocv:cfunc:`Copy` or :ocv:cfunc:`Split`) the channel to a separate image/matrix, process it and then copy the result back (via :ocv:cfunc:`Copy` or :ocv:cfunc:`Merge`) if needed.


SetImageROI
-----------
Sets an image Region Of Interest (ROI) for a given rectangle.

.. ocv:cfunction:: void cvSetImageROI( IplImage* image, CvRect rect)
.. ocv:pyoldfunction:: cv.SetImageROI(image, rect)-> None

    :param image: A pointer to the image header

    :param rect: The ROI rectangle

If the original image ROI was ``NULL`` and the ``rect`` is not the whole image, the ROI structure is allocated.

Most OpenCV functions support the use of ROI and treat the image rectangle as a separate image. For example, all of the pixel coordinates are counted from the top-left (or bottom-left) corner of the ROI, not the original image.


SetReal?D
---------
Change a specific array element.

.. ocv:cfunction:: void cvSetReal1D(CvArr* arr, int idx0, double value)

.. ocv:cfunction:: void cvSetReal2D(CvArr* arr, int idx0, int idx1, double value)

.. ocv:cfunction:: void cvSetReal3D(CvArr* arr, int idx0, int idx1, int idx2, double value)

.. ocv:cfunction:: void cvSetRealND( CvArr* arr, const int* idx, double value )

.. ocv:pyoldfunction:: cv.SetReal1D(arr, idx, value) -> None
.. ocv:pyoldfunction:: cv.SetReal2D(arr, idx0, idx1, value) -> None
.. ocv:pyoldfunction:: cv.SetReal3D(arr, idx0, idx1, idx2, value) -> None
.. ocv:pyoldfunction:: cv.SetRealND(arr, indices, value) -> None

    :param arr: Input array

    :param idx0: The first zero-based component of the element index

    :param idx1: The second zero-based component of the element index

    :param idx2: The third zero-based component of the element index

    :param idx: Array of the element indices

    :param value: The assigned value

The functions assign a new value to a specific element of a single-channel array. If the array has multiple channels, a runtime error is raised. Note that the ``Set*D`` function can be used safely for both single-channel and multiple-channel arrays, though they are a bit slower.

In the case of a sparse array the functions create the node if it does not yet exist.

SetZero
-------
Clears the array.

.. ocv:cfunction:: void cvSetZero(CvArr* arr)
.. ocv:pyoldfunction:: cv.SetZero(arr) -> None

    :param arr: Array to be cleared

The function clears the array. In the case of dense arrays (CvMat, CvMatND or IplImage), cvZero(array) is equivalent to cvSet(array,cvScalarAll(0),0). In the case of sparse arrays all the elements are removed.

mGet
----
Returns the particular element of single-channel floating-point matrix.

.. ocv:cfunction:: double cvmGet(const CvMat* mat, int row, int col)
.. ocv:pyoldfunction:: cv.mGet(mat, row, col) -> float

    :param mat: Input matrix

    :param row: The zero-based index of row

    :param col: The zero-based index of column

The function is a fast replacement for :ocv:cfunc:`GetReal2D` in the case of single-channel floating-point matrices. It is faster because it is inline, it does fewer checks for array type and array element type, and it checks for the row and column ranges only in debug mode.

mSet
----
Sets a specific element of a single-channel floating-point matrix.

.. ocv:cfunction:: void cvmSet(CvMat* mat, int row, int col, double value)
.. ocv:pyoldfunction:: cv.mSet(mat, row, col, value)-> None

    :param mat: The matrix

    :param row: The zero-based index of row

    :param col: The zero-based index of column

    :param value: The new value of the matrix element

The function is a fast replacement for :ocv:cfunc:`SetReal2D` in the case of single-channel floating-point matrices. It is faster because it is inline, it does fewer checks for array type and array element type,  and it checks for the row and column ranges only in debug mode.


SetIPLAllocators
----------------
Makes OpenCV use IPL functions for allocating IplImage and IplROI structures.

.. ocv:cfunction:: void cvSetIPLAllocators( Cv_iplCreateImageHeader create_header,                          Cv_iplAllocateImageData allocate_data, Cv_iplDeallocate deallocate,                          Cv_iplCreateROI create_roi, Cv_iplCloneImage clone_image )

Normally, the function is not called directly. Instead, a simple macro ``CV_TURN_ON_IPL_COMPATIBILITY()`` is used that calls ``cvSetIPLAllocators`` and passes there pointers to IPL allocation functions. ::

    ...
    CV_TURN_ON_IPL_COMPATIBILITY()
    ...


RNG
---
Initializes a random number generator state.

.. ocv:cfunction:: CvRNG cvRNG(int64 seed=-1)
.. ocv:pyoldfunction:: cv.RNG(seed=-1LL)-> CvRNG

    :param seed: 64-bit value used to initiate a random sequence

The function initializes a random number generator and returns the state. The pointer to the state can be then passed to the :ocv:cfunc:`RandInt`, :ocv:cfunc:`RandReal` and :ocv:cfunc:`RandArr` functions. In the current implementation a multiply-with-carry generator is used.

.. seealso:: the C++ class :ocv:class:`RNG` replaced ``CvRNG``.


RandArr
-------
Fills an array with random numbers and updates the RNG state.

.. ocv:cfunction:: void cvRandArr( CvRNG* rng, CvArr* arr, int dist_type, CvScalar param1, CvScalar param2 )

.. ocv:pyoldfunction:: cv.RandArr(rng, arr, distType, param1, param2)-> None

    :param rng: CvRNG state initialized by :ocv:cfunc:`RNG`

    :param arr: The destination array

    :param dist_type: Distribution type

            * **CV_RAND_UNI** uniform distribution

            * **CV_RAND_NORMAL** normal or Gaussian distribution

    :param param1: The first parameter of the distribution. In the case of a uniform distribution it is the inclusive lower boundary of the random numbers range. In the case of a normal distribution it is the mean value of the random numbers.

    :param param2: The second parameter of the distribution. In the case of a uniform distribution it is the exclusive upper boundary of the random numbers range. In the case of a normal distribution it is the standard deviation of the random numbers.

The function fills the destination array with uniformly or normally distributed random numbers.

.. seealso:: :ocv:func:`randu`, :ocv:func:`randn`, :ocv:func:`RNG::fill`.

RandInt
-------
Returns a 32-bit unsigned integer and updates RNG.

.. ocv:cfunction:: unsigned cvRandInt(CvRNG* rng)
.. ocv:pyoldfunction:: cv.RandInt(rng)-> unsigned

    :param rng: CvRNG state initialized by  :ocv:cfunc:`RNG`.

The function returns a uniformly-distributed random 32-bit unsigned integer and updates the RNG state. It is similar to the rand() function from the C runtime library, except that OpenCV functions always generates a 32-bit random number, regardless of the platform.


RandReal
--------
Returns a floating-point random number and updates RNG.

.. ocv:cfunction:: double cvRandReal(CvRNG* rng)
.. ocv:pyoldfunction:: cv.RandReal(rng) -> float

    :param rng: RNG state initialized by  :ocv:cfunc:`RNG`

The function returns a uniformly-distributed random floating-point number between 0 and 1 (1 is not included).


fromarray
---------
Create a CvMat from an object that supports the array interface.

.. ocv:pyoldfunction:: cv.fromarray(array, allowND=False) -> mat

    :param object: Any object that supports the array interface

    :param allowND: If true, will return a CvMatND

If the object supports the `array interface <http://docs.scipy.org/doc/numpy/reference/arrays.interface.html>`_
,
return a :ocv:struct:`CvMat` or :ocv:struct:`CvMatND`, depending on ``allowND`` flag:

  * If ``allowND = False``, then the object's array must be either 2D or 3D. If it is 2D, then the returned CvMat has a single channel.  If it is 3D, then the returned CvMat will have N channels, where N is the last dimension of the array. In this case, N cannot be greater than OpenCV's channel limit, ``CV_CN_MAX``.

  * If``allowND = True``, then ``fromarray`` returns a single-channel :ocv:struct:`CvMatND` with the same shape as the original array.

For example, `NumPy <http://numpy.scipy.org/>`_ arrays support the array interface, so can be converted to OpenCV objects:

.. code-block::python

    >>> import cv2.cv as cv, numpy
    >>> a = numpy.ones((480, 640))
    >>> mat = cv.fromarray(a)
    >>> print cv.GetDims(mat), cv.CV_MAT_CN(cv.GetElemType(mat))
    (480, 640) 1
    >>> a = numpy.ones((480, 640, 3))
    >>> mat = cv.fromarray(a)
    >>> print cv.GetDims(mat), cv.CV_MAT_CN(cv.GetElemType(mat))
    (480, 640) 3
    >>> a = numpy.ones((480, 640, 3))
    >>> mat = cv.fromarray(a, allowND = True)
    >>> print cv.GetDims(mat), cv.CV_MAT_CN(cv.GetElemType(mat))
    (480, 640, 3) 1

.. note:: In the new Python wrappers (**cv2** module) the function is not needed, since cv2 can process  Numpy arrays (and this is the only supported array type).

