Images Warping
==============

.. highlight:: cpp

detail::RotationWarper
----------------------
.. ocv:class:: detail::RotationWarper

Rotation-only model image warper interface. ::

    class CV_EXPORTS RotationWarper
    {
    public:
        virtual ~RotationWarper() {}

        virtual Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R) = 0;

        virtual Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap) = 0;

        virtual Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                           Mat &dst) = 0;
       
        virtual void warpBackward(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                                  Size dst_size, Mat &dst) = 0;

        virtual Rect warpRoi(Size src_size, const Mat &K, const Mat &R) = 0;
    };

detail::RotationWarper::warpPoint
---------------------------------

Projects the image point.

.. ocv:function:: Point2f detail::RotationWarper::warpPoint(const Point2f &pt, const Mat &K, const Mat &R)

    :param pt: Source point
    
    :param K: Camera intrinsic parameters

    :param R: Camera rotation matrix

    :return: Projected point

detail::RotationWarper::buildMaps
---------------------------------

Builds the projection maps according to the given camera data.

.. ocv:function:: Rect detail::RotationWarper::buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)

    :param src_size: Source image size
    
    :param K: Camera intrinsic parameters

    :param R: Camera rotation matrix

    :param xmap: Projection map for the x axis

    :param ymap: Projection map for the y axis

    :return: Projected image minimum bounding box

detail::RotationWarper::warp
----------------------------

Projects the image.

.. ocv:function:: Point detal::RotationWarper::warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode, Mat &dst)

    :param src: Source image
    
    :param K: Camera intrinsic parameters

    :param R: Camera rotation matrix

    :param interp_mode: Interpolation mode

    :param border_mode: Border extrapolation mode

    :param dst: Projected image

    :return: Project image top-left corner

detail::RotationWarper::warpBackward
------------------------------------

Projects the image backward.

.. ocv:function:: void detail::RotationWarper::warpBackward(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode, Size dst_size, Mat &dst)

    :param src: Projected image

    :param K: Camera intrinsic parameters

    :param R: Camera rotation matrix

    :param interp_mode: Interpolation mode

    :param border_mode: Border extrapolation mode

    :param dst_size: Backward-projected image size

    :param dst_size: Backward-projected image

detail::RotationWarper::warpRoi
-------------------------------

.. ocv:function:: Rect detail::RotationWarper::warpRoi(Size src_size, const Mat &K, const Mat &R)

    :param src_size: Source image bounding box
    
    :param K: Camera intrinsic parameters

    :param R: Camera rotation matrix

    :return: Projected image minimum bounding box

detail::ProjectorBase
---------------------
.. ocv:struct:: detail::ProjectorBase

Base class for warping logic implementation. ::

    struct CV_EXPORTS ProjectorBase
    {
        void setCameraParams(const Mat &K = Mat::eye(3, 3, CV_32F), 
                            const Mat &R = Mat::eye(3, 3, CV_32F), 
                            const Mat &T = Mat::zeros(3, 1, CV_32F));

        float scale;
        float k[9];
        float rinv[9];
        float r_kinv[9];
        float k_rinv[9];
        float t[3];
    };

detail::RotationWarperBase
--------------------------
.. ocv:class:: detail::RotationWarperBase

Base class for rotation-based warper using a `detail::ProjectorBase`_ derived class. ::

    template <class P>
    class CV_EXPORTS RotationWarperBase : public RotationWarper
    {   
    public:
        Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R);
        
        Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap);

        Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                Mat &dst);

        void warpBackward(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                        Size dst_size, Mat &dst);

        Rect warpRoi(Size src_size, const Mat &K, const Mat &R);

    protected:

        // Detects ROI of the destination image. It's correct for any projection.
        virtual void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);

        // Detects ROI of the destination image by walking over image border.
        // Correctness for any projection isn't guaranteed.
        void detectResultRoiByBorder(Size src_size, Point &dst_tl, Point &dst_br);

        P projector_;
    };

detail::PlaneWarper
-------------------
.. ocv:class:: detail::PlaneWarper

Warper that maps an image onto the z = 1 plane. ::

    class CV_EXPORTS PlaneWarper : public RotationWarperBase<PlaneProjector>
    {
    public:
        PlaneWarper(float scale = 1.f) { projector_.scale = scale; }

        void setScale(float scale) { projector_.scale = scale; }

        Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R, const Mat &T);

        Rect buildMaps(Size src_size, const Mat &K, const Mat &R, const Mat &T, Mat &xmap, Mat &ymap);

        Point warp(const Mat &src, const Mat &K, const Mat &R, const Mat &T, int interp_mode, int border_mode,
                   Mat &dst);

        Rect warpRoi(Size src_size, const Mat &K, const Mat &R, const Mat &T);

    protected:
        void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);
    };

.. seealso:: :ocv:class:`detail::RotationWarper`

detail::SphericalWarper
-----------------------
.. ocv:class:: detail::SphericalWarper

Warper that maps an image onto the unit sphere located at the origin. ::

    class CV_EXPORTS SphericalWarper : public RotationWarperBase<SphericalProjector>
    {
    public:
        SphericalWarper(float scale) { projector_.scale = scale; }

    protected:
        void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br);
    };

.. seealso:: :ocv:class:`detail::RotationWarper`
   
detail::CylindricalWarper
-------------------------
.. ocv:class:: detail::CylindricalWarper

Warper that maps an image onto the x*x + z*z = 1 cylinder. ::
    
    class CV_EXPORTS CylindricalWarper : public RotationWarperBase<CylindricalProjector>
    {
    public:
        CylindricalWarper(float scale) { projector_.scale = scale; }

    protected:
        void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
        { 
            RotationWarperBase<CylindricalProjector>::detectResultRoiByBorder(src_size, dst_tl, dst_br); 
        }
    };

.. seealso:: :ocv:class:`detail::RotationWarper`

