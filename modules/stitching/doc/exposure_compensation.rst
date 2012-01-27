Exposure Compensation
=====================

.. highlight:: cpp

detail::ExposureCompensation
----------------------------
.. ocv:class:: detail::ExposureCompensation

Base class for all exposure compensators. ::

    class CV_EXPORTS ExposureCompensator
    {
    public:
        virtual ~ExposureCompensator() {}

        enum { NO, GAIN, GAIN_BLOCKS };
        static Ptr<ExposureCompensator> createDefault(int type);

        void feed(const std::vector<Point> &corners, const std::vector<Mat> &images,
                  const std::vector<Mat> &masks);
        virtual void feed(const std::vector<Point> &corners, const std::vector<Mat> &images,
                          const std::vector<std::pair<Mat,uchar> > &masks) = 0;
        virtual void apply(int index, Point corner, Mat &image, const Mat &mask) = 0;
    };

detail::ExposureCompensation::feed
----------------------------------

.. ocv:function:: void detail::ExposureCompensation::feed(const std::vector<Point> &corners, const std::vector<Mat> &images, const std::vector<Mat> &masks)

    :param corners: Source image top-left corners

    :param images: Source images

    :param masks: Image masks to update

detail::ExposureCompensation::feed
----------------------------------

.. ocv:function:: void detail::ExposureCompensation::feed(const std::vector<Point> &corners, const std::vector<Mat> &images, const std::vector<std::pair<Mat,uchar> > &masks)

    :param corners: Source image top-left corners

    :param images: Source images

    :param masks: Image masks to update (second value in pair specifies the value which should be used to detect where image is)

detil::ExposureCompensation::apply
----------------------------------

Compensate exposure in the specified image.

.. ocv:function:: void detail::ExposureCompensation::apply(int index, Point corner, Mat &image, const Mat &mask)

    :param index: Image index

    :param corner: Image top-left corner

    :param image: Image to process

    :param mask: Image mask

detail::NoExposureCompensator
-----------------------------
.. ocv:class:: detail::NoExposureCompensator

Stub exposure compensator which does nothing. ::

    class CV_EXPORTS NoExposureCompensator : public ExposureCompensator
    {
    public:
        void feed(const std::vector<Point> &/*corners*/, const std::vector<Mat> &/*images*/,
                  const std::vector<std::pair<Mat,uchar> > &/*masks*/) {};
        void apply(int /*index*/, Point /*corner*/, Mat &/*image*/, const Mat &/*mask*/) {};
    };

.. seealso:: :ocv:class:`detail::ExposureCompensation`

detail::GainCompensator
-----------------------
.. ocv:class:: detail::GainCompensator

Exposure compensator which tries to remove exposure related artifacts by adjusting image intensities, see [BL07]_ and [WJ10]_ for details. ::

    class CV_EXPORTS GainCompensator : public ExposureCompensator
    {
    public:
        void feed(const std::vector<Point> &corners, const std::vector<Mat> &images,
                  const std::vector<std::pair<Mat,uchar> > &masks);
        void apply(int index, Point corner, Mat &image, const Mat &mask);
        std::vector<double> gains() const;

    private:
        /* hidden */
    };

.. seealso:: :ocv:class:`detail::ExposureCompensation`

detail::BlocksGainCompensator
-----------------------------
.. ocv:class:: detail::BlocksGainCompensator

Exposure compensator which tries to remove exposure related artifacts by adjusting image block intensities, see [UES01]_ for details. ::

    class CV_EXPORTS BlocksGainCompensator : public ExposureCompensator
    {
    public:
        BlocksGainCompensator(int bl_width = 32, int bl_height = 32) 
                : bl_width_(bl_width), bl_height_(bl_height) {}
        void feed(const std::vector<Point> &corners, const std::vector<Mat> &images,
                  const std::vector<std::pair<Mat,uchar> > &masks);
        void apply(int index, Point corner, Mat &image, const Mat &mask);

    private:
        /* hidden */
    };

.. seealso:: :ocv:class:`detail::ExposureCompensation`

