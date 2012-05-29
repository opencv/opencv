Seam Estimation
===============

.. highlight:: cpp

detail::SeamFinder
------------------
.. ocv:class:: detail::SeamFinder

Base class for a seam estimator. ::

    class CV_EXPORTS SeamFinder
    {
    public:
        virtual ~SeamFinder() {}
        virtual void find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                          std::vector<Mat> &masks) = 0;
    };

detail::SeamFinder::find
------------------------

Estimates seams.

.. ocv:function:: void detail::SeamFinder::find(const std::vector<Mat> &src, const std::vector<Point> &corners, std::vector<Mat> &masks)

    :param src: Source images

    :param corners: Source image top-left corners

    :param masks: Source image masks to update


detail::NoSeamFinder
--------------------
.. ocv:class:: detail::NoSeamFinder : public detail::SeamFinder

Stub seam estimator which does nothing. ::

    class CV_EXPORTS NoSeamFinder : public SeamFinder
    {
    public:
        void find(const std::vector<Mat>&, const std::vector<Point>&, std::vector<Mat>&) {}
    };

.. seealso:: :ocv:class:`detail::SeamFinder`

detail::PairwiseSeamFinder
--------------------------
.. ocv:class:: detail::PairwiseSeamFinder : public detail::SeamFinder

Base class for all pairwise seam estimators. ::

    class CV_EXPORTS PairwiseSeamFinder : public SeamFinder
    {
    public:
        virtual void find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                          std::vector<Mat> &masks);

    protected:
        void run();
        virtual void findInPair(size_t first, size_t second, Rect roi) = 0;

        std::vector<Mat> images_;
        std::vector<Size> sizes_;
        std::vector<Point> corners_;
        std::vector<Mat> masks_;
    };

.. seealso:: :ocv:class:`detail::SeamFinder`

detail::PairwiseSeamFinder::findInPair
--------------------------------------

Resolves masks intersection of two specified images in the given ROI.

.. ocv:function:: void detail::PairwiseSeamFinder::findInPair(size_t first, size_t second, Rect roi)

    :param first: First image index

    :param second: Second image index

    :param roi: Region of interest

detail::VoronoiSeamFinder
-------------------------
.. ocv:class:: detail::VoronoiSeamFinder : public detail::PairwiseSeamFinder

Voronoi diagram-based seam estimator. ::

    class CV_EXPORTS VoronoiSeamFinder : public PairwiseSeamFinder
    {
    public:
        virtual void find(const std::vector<Size> &size, const std::vector<Point> &corners,
                          std::vector<Mat> &masks);
    private:
        void findInPair(size_t first, size_t second, Rect roi);
    };

.. seealso:: :ocv:class:`detail::PairwiseSeamFinder`

detail::GraphCutSeamFinderBase
------------------------------
.. ocv:class:: detail::GraphCutSeamFinderBase

Base class for all minimum graph-cut-based seam estimators. ::

    class CV_EXPORTS GraphCutSeamFinderBase
    {
    public:
        enum { COST_COLOR, COST_COLOR_GRAD };
    };

detail::GraphCutSeamFinder
--------------------------
.. ocv:class:: detail::GraphCutSeamFinder : public detail::GraphCutSeamFinderBase, public detail::SeamFinder

Minimum graph cut-based seam estimator. See details in [V03]_. ::

    class CV_EXPORTS GraphCutSeamFinder : public GraphCutSeamFinderBase, public SeamFinder
    {
    public:
        GraphCutSeamFinder(int cost_type = COST_COLOR_GRAD, float terminal_cost = 10000.f,
                           float bad_region_penalty = 1000.f);

        void find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                  std::vector<Mat> &masks);

    private:
        /* hidden */
    };

.. seealso:: 
    :ocv:class:`detail::GraphCutSeamFinderBase`,
    :ocv:class:`detail::SeamFinder`
