Seam Estimation
===============

.. highlight:: cpp

detail::SeamFinder
------------------
.. ocv:class:: detail::SeamFinder

Base class for seam estimators. ::

    class CV_EXPORTS SeamFinder
    {
    public:
        virtual ~SeamFinder() {}
        virtual void find(const std::vector<Mat> &src, const std::vector<Point> &corners,
                          std::vector<Mat> &masks) = 0;
    };

detail::NoSeamFinder
--------------------
.. ocv:class:: detail::NoSeamFinder

Stub seam estimator which does nothing. ::

    class CV_EXPORTS NoSeamFinder : public SeamFinder
    {
    public:
        void find(const std::vector<Mat>&, const std::vector<Point>&, std::vector<Mat>&) {}
    };

.. seealso:: :ocv:class:`detail::SeamFinder`

detail::PairwiseSeamFinder
--------------------------
.. ocv:class:: detail::PairwiseSeamFinder

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

detail::VoronoiSeamFinder
-------------------------
.. ocv:class:: detail::VoronoiSeamFinder

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

Base class for all minimum graph cut-based seam estimators. ::

    class CV_EXPORTS GraphCutSeamFinderBase
    {
    public:
        enum { COST_COLOR, COST_COLOR_GRAD };
    };

detail::GraphCutSeamFinder
--------------------------
.. ocv:class:: detail::GraphCutSeamFinder

Minimum graph cut-based seam estimator. ::

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
