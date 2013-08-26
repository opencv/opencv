.. _hdrimaging:

High Dynamic Range Imaging
***************************************

Introduction
------------------
Today most digital images and imaging devices use three bytes per channel thus limiting the dynamic range of the device to two orders of magnitude, while human eye can adapt to lighting conditions varying by ten orders of magnitude. When we take photographs bright regions may be overexposed and dark ones may be on the other hand underexposed so we can't capture the whole scene in a single exposure.  HDR imaging works with images that use more that 8 bits per channel (usually 32-bit float values), allowing any dynamic range. 

There are different ways to obtain HDR images but the most common one is to use photographs of the scene taken with different exposure values. To combine the exposures it is useful to know your camera's response function and there are algorithms to estimate it. After the HDR image has been constructed it has to be converted back to 8-bit to view it on regular displays. This process is called tonemapping. Additional complexities arise when objects of the scene or camera move between shots.

In this tutorial we show how to make and display HDR image provided we have exposure sequence. In our case images are already aligned and there are no moving objects. We also demonstrate an alternative approach called exposure fusion that produces low dynamic range image. Each step of this pipeline can be made using different algorithms so take a look at the reference manual to find them all.

Exposure sequence
------------------

.. image:: images/memorial.png
  :height: 357pt
  :width:  242pt
  :alt: Exposure sequence
  :align: center

Source Code
===========

.. literalinclude:: ../../../../samples/cpp/tutorial_code/photo/hdr_imaging/hdr_imaging.cpp
   :language: cpp
   :linenos:
   :tab-width: 4

Explanation
===========

1. **Load images and exposure times**

  .. code-block:: cpp

    vector<Mat> images;
    vector<float> times;
    loadExposureSeq(argv[1], images, times);

  First we load input images and exposure times from user-defined destination. The folder should contain images and *list.txt* - file that contains file names and inverse exposure times.
  
  For our image sequence the list looks like this:
  
  .. code-block:: none 
  
    memorial00.png 0.03125
    memorial01.png 0.0625 
    ...
    memorial15.png 1024

2. **Estimate camera response**

  .. code-block:: cpp

    Mat response;
    Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec();
    calibrate->process(images, response, times);
    
  It is necessary to know camera response function for most HDR construction algorithms.
    
  We use one of calibration algorithms to estimate inverse CRF for all 256 pixel values.
    
3. **Make HDR image**

  .. code-block:: cpp

    Mat hdr;
    Ptr<MergeDebevec> merge_debevec = createMergeDebevec();
    merge_debevec->process(images, hdr, times, response);
    
  We use Debevec's weighting scheme to construct HDR image using response calculated in the previous item.
    
4. **Tonemap HDR image**

  .. code-block:: cpp
  
    Mat ldr;
    Ptr<TonemapDurand> tonemap = createTonemapDurand(2.2f);
    tonemap->process(hdr, ldr);
    
  Since we want to see our results on common LDR display we have to map our HDR image to 8-bit range preserving most details.
    
  That is what tonemapping algorithms are for. We use bilateral filtering tonemapper and set 2.2 as value for gamma correction.
    
5. **Perform exposure fusion**

  .. code-block:: cpp
  
    Mat fusion;
    Ptr<MergeMertens> merge_mertens = createMergeMertens();
    merge_mertens->process(images, fusion);

  There is an alternative way to merge our exposures in case we don't need HDR image.
    
  This process is called exposure fusion and produces LDR image that doesn't require gamma correction. It also doesn't use exposure values of the photographs.
    
6. **Write results**

  .. code-block:: cpp
  
    imwrite("fusion.png", fusion * 255);
    imwrite("ldr.png", ldr * 255);
    imwrite("hdr.hdr", hdr);
    
  Now it's time to view the results.
    
  Note that HDR image can't be stored in one of common image formats, so we save it as Radiance image (.hdr).
  
  Also all HDR imaging functions return results in [0, 1] range so we multiply them by 255.
  
Results
=======

Tonemapped image
------------------

.. image:: images/ldr.png
  :height: 357pt
  :width:  242pt
  :alt: Tonemapped image
  :align: center
  
Exposure fusion
------------------

.. image:: images/fusion.png
  :height: 357pt
  :width:  242pt
  :alt: Exposure fusion
  :align: center
