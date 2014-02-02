.. _howToUseIPPAconversion:

Intel® IPP Asynchronous C/C++ library in OpenCV
***********************************************

Goal
====

.. _hppiSobel: http://software.intel.com/en-us/node/474701
.. _hppiMatrix: http://software.intel.com/en-us/node/501660

The tutorial demonstrates the `Intel® IPP Asynchronous C/C++ <http://software.intel.com/en-us/intel-ipp-preview>`_ library usage with OpenCV.
The code example below illustrates implementation of the Sobel operation, accelerated with Intel® IPP Asynchronous C/C++ functions.
In this code example, :ippa_convert:`hpp::getMat <>` and :ippa_convert:`hpp::getHpp <>` functions are used for data conversion between hppiMatrix_ and ``Mat`` matrices.

Code
====

You may also find the source code in the :file:`samples/cpp/tutorial_code/core/ippasync/ippasync_sample.cpp`
file of the OpenCV source library or :download:`download it from here
<../../../../samples/cpp/tutorial_code/core/ippasync/ippasync_sample.cpp>`.

.. literalinclude:: ../../../../samples/cpp/tutorial_code/core/ippasync/ippasync_sample.cpp
   :language: cpp
   :linenos:
   :tab-width: 4

Explanation
===========

#. Create parameters for OpenCV:

   .. code-block:: cpp

      VideoCapture cap;
      Mat image, gray, result;

   and IPP Async:

   .. code-block:: cpp

      hppiMatrix* src,* dst;
      hppAccel accel = 0;
      hppAccelType accelType;
      hppStatus sts;
      hppiVirtualMatrix * virtMatrix;

#. Load input image or video. How to open and read video stream you can see in the :ref:`videoInputPSNRMSSIM` tutorial.

   .. code-block:: cpp

      if( useCamera )
      {
         printf("used camera\n");
         cap.open(0);
      }
      else
      {
         printf("used image %s\n", file.c_str());
         cap.open(file.c_str());
      }

      if( !cap.isOpened() )
      {
         printf("can not open camera or video file\n");
         return -1;
      }

#. Create accelerator instance using `hppCreateInstance <http://software.intel.com/en-us/node/501686>`_:

   .. code-block:: cpp

      accelType = sAccel == "cpu" ? HPP_ACCEL_TYPE_CPU:
                  sAccel == "gpu" ? HPP_ACCEL_TYPE_GPU:
                                    HPP_ACCEL_TYPE_ANY;

      //Create accelerator instance
      sts = hppCreateInstance(accelType, 0, &accel);
      CHECK_STATUS(sts, "hppCreateInstance");

#. Create an array of virtual matrices using `hppiCreateVirtualMatrices <http://software.intel.com/en-us/node/501700>`_ function.

   .. code-block:: cpp

      virtMatrix = hppiCreateVirtualMatrices(accel, 1);

#. Prepare a matrix for input and output data:

   .. code-block:: cpp

      cap >> image;
      if(image.empty())
         break;

      cvtColor( image, gray, COLOR_BGR2GRAY );

      result.create( image.rows, image.cols, CV_8U);

#. Convert ``Mat`` to hppiMatrix_ using :ippa_convert:`getHpp <>` and call hppiSobel_ function.

   .. code-block:: cpp

      //convert Mat to hppiMatrix
      src = getHpp(gray, accel);
      dst = getHpp(result, accel);

      sts = hppiSobel(accel,src, HPP_MASK_SIZE_3X3,HPP_NORM_L1,virtMatrix[0]);
      CHECK_STATUS(sts,"hppiSobel");

      sts = hppiConvert(accel, virtMatrix[0], 0, HPP_RND_MODE_NEAR, dst, HPP_DATA_TYPE_8U);
      CHECK_STATUS(sts,"hppiConvert");

      // Wait for tasks to complete
      sts = hppWait(accel, HPP_TIME_OUT_INFINITE);
      CHECK_STATUS(sts, "hppWait");

   We use `hppiConvert <http://software.intel.com/en-us/node/501746>`_ because hppiSobel_ returns destination
   matrix with ``HPP_DATA_TYPE_16S`` data type for source matrix with ``HPP_DATA_TYPE_8U`` type.
   You should check ``hppStatus`` after each call IPP Async function.

#. Create windows and show the images, the usual way.

   .. code-block:: cpp

      imshow("image", image);
      imshow("rez", result);

      waitKey(15);

#. Delete hpp matrices.

   .. code-block:: cpp

      sts =  hppiFreeMatrix(src);
      CHECK_DEL_STATUS(sts,"hppiFreeMatrix");

      sts =  hppiFreeMatrix(dst);
      CHECK_DEL_STATUS(sts,"hppiFreeMatrix");

#. Delete virtual matrices and accelerator instance.

   .. code-block:: cpp

      if (virtMatrix)
      {
         sts = hppiDeleteVirtualMatrices(accel, virtMatrix);
         CHECK_DEL_STATUS(sts,"hppiDeleteVirtualMatrices");
      }

      if (accel)
      {
         sts = hppDeleteInstance(accel);
         CHECK_DEL_STATUS(sts, "hppDeleteInstance");
      }

Result
=======

After compiling the code above we can execute it giving an image or video path and accelerator type as an argument.
For this tutorial we use baboon.png image as input. The result is below.

  .. image:: images/How_To_Use_IPPA_Result.jpg
    :alt: Final Result
    :align: center