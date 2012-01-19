Images Warping
==============

.. highlight:: cpp

detail::RotationWarper
----------------------
.. ocv:class:: detail::RotationWarper

Rotation-only model image warpers interface. ::

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

