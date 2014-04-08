Intel® IPP Asynchronous C/C++ Converters
========================================

.. highlight:: cpp

General Information
-------------------

This section describes conversion between OpenCV and `Intel® IPP Asynchronous C/C++ <http://software.intel.com/en-us/intel-ipp-preview>`_ library.
`Getting Started Guide <http://registrationcenter.intel.com/irc_nas/3727/ipp_async_get_started.htm>`_ help you to install the library, configure header and library build paths.

hpp::getHpp
-----------
Create ``hppiMatrix`` from ``Mat``.

.. ocv:function:: hppiMatrix* hpp::getHpp(const Mat& src, hppAccel accel)

    :param src: input matrix.
    :param accel: accelerator instance. Supports type:

            * **HPP_ACCEL_TYPE_CPU** - accelerated by optimized CPU instructions.

            * **HPP_ACCEL_TYPE_GPU** - accelerated by GPU programmable units or fixed-function accelerators.

            * **HPP_ACCEL_TYPE_ANY** - any acceleration or no acceleration available.

This function allocates and initializes the ``hppiMatrix`` that has the same size and type as input matrix, returns the ``hppiMatrix*``.

If you want to use zero-copy for GPU you should to have 4KB aligned matrix data. See details `hppiCreateSharedMatrix <http://software.intel.com/ru-ru/node/501697>`_.

Supports ``CV_8U``, ``CV_16U``, ``CV_16S``, ``CV_32S``, ``CV_32F``, ``CV_64F``.

.. note:: The ``hppiMatrix`` pointer to the image buffer in system memory refers to the ``src.data``. Control the lifetime of the matrix and don't change its data, if there is no special need.
.. seealso:: :ref:`howToUseIPPAconversion`, :ocv:func:`hpp::getMat`


hpp::getMat
-----------
Create ``Mat`` from ``hppiMatrix``.

.. ocv:function:: Mat hpp::getMat(hppiMatrix* src, hppAccel accel, int cn)

    :param src: input hppiMatrix.

    :param accel: accelerator instance (see :ocv:func:`hpp::getHpp` for the list of acceleration framework types).

    :param cn: number of channels.

This function allocates and initializes the ``Mat`` that has the same size and type as input matrix.
Supports ``CV_8U``, ``CV_16U``, ``CV_16S``, ``CV_32S``, ``CV_32F``, ``CV_64F``.

.. seealso:: :ref:`howToUseIPPAconversion`, :ocv:func:`hpp::copyHppToMat`, :ocv:func:`hpp::getHpp`.


hpp::copyHppToMat
-----------------
Convert ``hppiMatrix`` to ``Mat``.

.. ocv:function:: void hpp::copyHppToMat(hppiMatrix* src, Mat& dst, hppAccel accel, int cn)

    :param src: input hppiMatrix.

    :param dst: output matrix.

    :param accel: accelerator instance (see :ocv:func:`hpp::getHpp` for the list of acceleration framework types).

    :param cn: number of channels.

This function allocates and initializes new matrix (if needed) that has the same size and type as input matrix.
Supports ``CV_8U``, ``CV_16U``, ``CV_16S``, ``CV_32S``, ``CV_32F``, ``CV_64F``.

.. seealso:: :ref:`howToUseIPPAconversion`, :ocv:func:`hpp::getMat`, :ocv:func:`hpp::getHpp`.