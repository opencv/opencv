Fast Marching Method
====================

.. highlight:: cpp

The Fast Marching Method [T04]_ is used in of the video stabilization routines to do motion and color inpainting. The method is implemented is a flexible way and it's made public for other users.

videostab::FastMarchingMethod
-----------------------------

.. ocv:class:: videostab::FastMarchingMethod

Describes the Fast Marching Method implementation.

::

    class CV_EXPORTS FastMarchingMethod
    {
    public:
        FastMarchingMethod();

        template <typename Inpaint>
        Inpaint run(const Mat &mask, Inpaint inpaint);

        Mat distanceMap() const;
    };


videostab::FastMarchingMethod::FastMarchingMethod
-------------------------------------------------

Constructor.

.. ocv:function:: videostab::FastMarchingMethod::FastMarchingMethod()


videostab::FastMarchingMethod::run
----------------------------------

Template method that runs the Fast Marching Method.

.. ocv:function:: template<typename Inpaint> Inpaint videostab::FastMarchingMethod::run(const Mat &mask, Inpaint inpaint)

    :param mask: Image mask. ``0`` value indicates that the pixel value must be inpainted, ``255`` indicates that the pixel value is known, other values aren't acceptable.

    :param inpaint: Inpainting functor that overloads ``void operator ()(int x, int y)``.

    :return: Inpainting functor.


videostab::FastMarchingMethod::distanceMap
------------------------------------------

.. ocv:function:: Mat videostab::FastMarchingMethod::distanceMap() const

    :return: Distance map that's created during working of the method.
