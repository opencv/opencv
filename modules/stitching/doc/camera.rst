Camera
======

.. highlight:: cpp

detail::CameraParams
--------------------
.. ocv:struct:: detail::CameraParams

Describes camera parameters.

.. note:: Translation is assumed to be zero during the whole stitching pipeline. 

::

    struct CV_EXPORTS CameraParams
    {
        CameraParams();
        CameraParams(const CameraParams& other);
        const CameraParams& operator =(const CameraParams& other);
        Mat K() const;

        double focal; // Focal length
        double aspect; // Aspect ratio
        double ppx; // Principal point X
        double ppy; // Principal point Y
        Mat R; // Rotation
        Mat t; // Translation
    };
