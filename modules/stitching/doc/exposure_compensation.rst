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

Exposure compensator which tries to remove exposure related artifacts by adjusting image intensities. ::

    class CV_EXPORTS GainCompensator : public ExposureCompensator
    {
    public:
        void feed(const std::vector<Point> &corners, const std::vector<Mat> &images,
                  const std::vector<std::pair<Mat,uchar> > &masks);
        void apply(int index, Point corner, Mat &image, const Mat &mask);
        std::vector<double> gains() const;

    private:
        Mat_<double> gains_;
    };

.. seealso:: :ocv:class:`detail::ExposureCompensation`

detail::BlocksGainCompensator
-----------------------------
.. ocv:class:: detail::BlocksGainCompensator

Exposure compensator which tries to remove exposure related artifacts by adjusting image block intensities. ::

    class CV_EXPORTS BlocksGainCompensator : public ExposureCompensator
    {
    public:
        BlocksGainCompensator(int bl_width = 32, int bl_height = 32) 
                : bl_width_(bl_width), bl_height_(bl_height) {}
        void feed(const std::vector<Point> &corners, const std::vector<Mat> &images,
                  const std::vector<std::pair<Mat,uchar> > &masks);
        void apply(int index, Point corner, Mat &image, const Mat &mask);

    private:
        int bl_width_, bl_height_;
        std::vector<Mat_<float> > gain_maps_;
    };

.. seealso:: :ocv:class:`detail::ExposureCompensation`

