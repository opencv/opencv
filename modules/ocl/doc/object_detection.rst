Object Detection
=============================

.. highlight:: cpp

ocl::OclCascadeClassifier
-------------------------
.. ocv:class:: ocl::OclCascadeClassifier : public CascadeClassifier

Cascade classifier class used for object detection. Supports HAAR cascade classifier  in the form of cross link ::

    class CV_EXPORTS OclCascadeClassifier : public CascadeClassifier
    {
    public:
            void detectMultiScale(oclMat &image, CV_OUT std::vector<cv::Rect>& faces,
                                              double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0,
                                              Size minSize = Size(), Size maxSize = Size());
    };

.. note::

   (Ocl) A face detection example using cascade classifiers can be found at opencv_source_code/samples/ocl/facedetect.cpp

ocl::OclCascadeClassifier::oclHaarDetectObjects
------------------------------------------------------
Detects objects of different sizes in the input image.

.. ocv:function:: void ocl::OclCascadeClassifier::detectMultiScale(oclMat &image, std::vector<cv::Rect>& faces, double scaleFactor = 1.1, int minNeighbors = 3, int flags = 0, Size minSize = Size(), Size maxSize = Size())

    :param image:  Matrix of type CV_8U containing an image where objects should be detected.

    :param faces: Vector of rectangles where each rectangle contains the detected object.

    :param scaleFactor: Parameter specifying how much the image size is reduced at each image scale.

    :param minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.

    :param minSize: Minimum possible object size. Objects smaller than that are ignored.

    :param maxSize: Maximum possible object size. Objects larger than that are ignored.

The function provides a very similar interface with that in CascadeClassifier class, except using oclMat as input image.

ocl::MatchTemplateBuf
---------------------
.. ocv:struct:: ocl::MatchTemplateBuf

Class providing memory buffers for :ocv:func:`ocl::matchTemplate` function, plus it allows to adjust some specific parameters. ::

    struct CV_EXPORTS MatchTemplateBuf
    {
        Size user_block_size;
        oclMat imagef, templf;
        std::vector<oclMat> images;
        std::vector<oclMat> image_sums;
        std::vector<oclMat> image_sqsums;
    };

You can use field `user_block_size` to set specific block size for :ocv:func:`ocl::matchTemplate` function. If you leave its default value `Size(0,0)` then automatic estimation of block size will be used (which is optimized for speed). By varying `user_block_size` you can reduce memory requirements at the cost of speed.

ocl::matchTemplate
------------------
Computes a proximity map for a raster template and an image where the template is searched for.

.. ocv:function:: void ocl::matchTemplate(const oclMat& image, const oclMat& templ, oclMat& result, int method)

.. ocv:function:: void ocl::matchTemplate(const oclMat& image, const oclMat& templ, oclMat& result, int method, MatchTemplateBuf &buf)

    :param image: Source image.  ``CV_32F`` and  ``CV_8U`` depth images (1..4 channels) are supported for now.

    :param templ: Template image with the size and type the same as  ``image`` .

    :param result: Map containing comparison results ( ``CV_32FC1`` ). If  ``image`` is  *W x H*  and ``templ`` is  *w x h*, then  ``result`` must be *W-w+1 x H-h+1*.

    :param method: Specifies the way to compare the template with the image.

    :param buf: Optional buffer to avoid extra memory allocations and to adjust some specific parameters. See :ocv:struct:`ocl::MatchTemplateBuf`.

    The following methods are supported for the ``CV_8U`` depth images for now:

    * ``CV_TM_SQDIFF``
    * ``CV_TM_SQDIFF_NORMED``
    * ``CV_TM_CCORR``
    * ``CV_TM_CCORR_NORMED``
    * ``CV_TM_CCOEFF``
    * ``CV_TM_CCOEFF_NORMED``

    The following methods are supported for the ``CV_32F`` images for now:

    * ``CV_TM_SQDIFF``
    * ``CV_TM_CCORR``

.. seealso:: :ocv:func:`matchTemplate`
