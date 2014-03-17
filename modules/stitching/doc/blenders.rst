Image Blenders
==============

.. highlight:: cpp

detail::Blender
---------------
.. ocv:class:: detail::Blender

Base class for all blenders. ::

    class CV_EXPORTS Blender
    {
    public:
        virtual ~Blender() {}

        enum { NO, FEATHER, MULTI_BAND };
        static Ptr<Blender> createDefault(int type, bool try_gpu = false);

        void prepare(const std::vector<Point> &corners, const std::vector<Size> &sizes);
        virtual void prepare(Rect dst_roi);
        virtual void feed(const Mat &img, const Mat &mask, Point tl);
        virtual void blend(Mat &dst, Mat &dst_mask);

    protected:
        Mat dst_, dst_mask_;
        Rect dst_roi_;
    };

detail::Blender::prepare
------------------------

Prepares the blender for blending.

.. ocv:function:: void detail::Blender::prepare(const std::vector<Point> &corners, const std::vector<Size> &sizes)

    :param corners: Source images top-left corners

    :param sizes: Source image sizes

detail::Blender::feed
---------------------

Processes the image.

.. ocv:function:: void detail::Blender::feed(InputArray img, InputArray mask, Point tl)

    :param img: Source image

    :param mask: Source image mask

    :param tl: Source image top-left corners

detail::Blender::blend
----------------------

Blends and returns the final pano.

.. ocv:function:: void detail::Blender::blend(InputOutputArray dst, InputOutputArray dst_mask)

    :param dst: Final pano

    :param dst_mask: Final pano mask

detail::FeatherBlender
----------------------
.. ocv:class:: detail::FeatherBlender : public detail::Blender

Simple blender which mixes images at its borders. ::

    class CV_EXPORTS FeatherBlender : public Blender
    {
    public:
        FeatherBlender(float sharpness = 0.02f) { setSharpness(sharpness); }

        float sharpness() const { return sharpness_; }
        void setSharpness(float val) { sharpness_ = val; }

        void prepare(Rect dst_roi);
        void feed(const Mat &img, const Mat &mask, Point tl);
        void blend(Mat &dst, Mat &dst_mask);

        // Creates weight maps for fixed set of source images by their masks and top-left corners.
        // Final image can be obtained by simple weighting of the source images.
        Rect createWeightMaps(const std::vector<Mat> &masks, const std::vector<Point> &corners,
                              std::vector<Mat> &weight_maps);

    private:
        /* hidden */
    };

.. seealso:: :ocv:class:`detail::Blender`

detail::MultiBandBlender
------------------------
.. ocv:class:: detail::MultiBandBlender : public detail::Blender

Blender which uses multi-band blending algorithm (see [BA83]_). ::

    class CV_EXPORTS MultiBandBlender : public Blender
    {
    public:
        MultiBandBlender(int try_gpu = false, int num_bands = 5);
        int numBands() const { return actual_num_bands_; }
        void setNumBands(int val) { actual_num_bands_ = val; }

        void prepare(Rect dst_roi);
        void feed(const Mat &img, const Mat &mask, Point tl);
        void blend(Mat &dst, Mat &dst_mask);

    private:
        /* hidden */
    };

.. seealso:: :ocv:class:`detail::Blender`
