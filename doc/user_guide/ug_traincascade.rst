***************************
Cascade Classifier Training
***************************

.. highlight:: cpp

Introduction
============
Usage of a cascade classifier consists of two main stages: a training and a detection. 
Detection stage with some base information about a cascade classifier are described in 
a documentation of ``objdetect`` module of general OpenCV documentation. Current section 
will describe how to prepare a training data and train a cascade classifier.

Importent notes
---------------
In OpenCV there are two applications to train cascade classifier: ``opencv_haartraining`` 
and ``opencv_traincascade``. ``opencv_traincascade`` is a newer version, written on C++. But the main 
versions difference is that ``opencv_traincascade`` supports both Haar [Viola2001]_ and LBP [Liao2007]_ (Local Binary Patterns) 
features. LBP features are integer in contrast with Haar features, so both a training and 
a detection with LBP are several times faster then with Haar features. As regards a comparison of LBP 
and Haar detection quality, it depends on a cascade training: a quality of training dataset first of all 
and training parameters too. It's possible to train a LBP-based classifier that will give 
almost the same quality as Haar-based one.

``opencv_traincascade`` and ``opencv_haartraining`` stores the trained classifier in different file formats. 
Note, the newer cascade detection interfaces (see ``CascadeClassifier`` class in ``objdetect`` module) supportes 
both formats. ``opencv_traincascade`` can save a trained cascade in the older format. But ``opencv_traincascade`` 
and ``opencv_haartraining`` can not load a classifier in another format for the futher training.

Also there are some auxilary utilities related to the cascade classifier training. 

    * ``opencv_createsamples`` is used to prepare the training base of positive samples and the test samples too. ``opencv_createsamples`` produces the positive samples dataset in a format that applicable (supported) both in ``opencv_haartraining`` and ``opencv_traincascade`` applications. 
    
    * ``opencv_performance`` may be used to evaluate the quality of the classifier trained by ``opencv_haartraining`` application only. It takes a collection of marked up images, applies the classifier and outputs the performance, i.e. number of found objects, number of missed objects, number of false alarms and other information.

Since ``opencv_haartraining`` is obsolete application, only ``opencv_traincascade`` will be described futher. ``opencv_createsamples`` utility is  needed 
to prepare a training data for ``opencv_traincascade``, so it will be described too.


Training data creation
======================
For training a training samples must be collected. There are two sample types: negative samples and positive samples. Negative samples 
correspond to non-object images. Positive samples correspond to object images. Negative samples set must be prepared manually, whereas 
positive samples set are created using ``opencv_createsamples`` utility.

Negative Samples
----------------
Negative samples are taken from arbitrary images. These images must not contain object representations. Negative samples are passed through 
background description file. It is a text file in which each text line contains the filename (relative to the directory of the description file) 
of negative sample image. This file must be created manually. Note that the negative samples and sample images are also called background 
samples or background samples images, and are used interchangeably in this document. Described images may be of different sizes. But each image 
should be (but not nessesarily) larger then training window size, because these images are used to subsample negative image of training size.


Example of negative description file:
 
Directory structure:

    .. code-block:: text

        /img
          img1.jpg
          img2.jpg
        bg.txt
 
File bg.txt:

    .. code-block:: text

        img/img1.jpg
        img/img2.jpg
        
Positive Samples
----------------
Positive samples are created by createsamples utility. They may be created from single object image or from collection of previously marked up images.

The single object image may for instance contain a company logo. Then are large set of positive samples are created from the given object image by randomly rotating, changing the logo color as well as placing the logo on arbitrary background.
The amount and range of randomness can be controlled by command line arguments.

Command line arguments:

* ``-vec <vec_file_name>``

    Name of the output file containing the positive samples for training.
    
* ``-img <image_file_name>``

    Source object image (e.g., a company logo).
    
* ``-bg <background_file_name>``

    Background description file; contains a list of images into which randomly distorted versions of the object are pasted for positive sample generation.

* ``-num <number_of_samples>``
    
    Number of positive samples to generate.
    
* ``-bgcolor <background_color>``

    Background color (currently grayscale images are assumed); the background color denotes the transparent color. Since there might be compression artifacts, the amount of color tolerance can be specified by ``-bgthresh``. All pixels between ``bgcolor-bgthresh`` and ``bgcolor+bgthresh`` are regarded as transparent.
    
* ``-bgthresh <background_color_threshold>``

* ``-inv``
    
    If specified, the colors will be inverted.
    
* ``-randinv``

    If specified, the colors will be inverted randomly.
      
* ``-maxidev <max_intensity_deviation>``
 
    Maximal intensity deviation of foreground samples pixels.
    
* ``-maxxangle <max_x_rotation_angle>``

* ``-maxyangle <max_y_rotation_angle>``

* ``-maxzangle <max_z_rotation_angle>``

      Maximum rotation angles in radians.
      
* ``-show``

    If specified, each sample will be shown. Pressing ``Esc`` will continue creation process without samples showing. Useful debugging option.
    
* ``-w <sample_width>``

    Width (in pixels) of the output samples.
  
* ``-h <sample_height>``

    Height (in pixels) of the output samples.

For following procedure is used to create a sample object instance:
The source image is rotated random around all three axes. The chosen angle is limited my ``-max?angle``. Next pixels of intensities in the range of [``bg_color-bg_color_threshold``; ``bg_color+bg_color_threshold``] are regarded as transparent. White noise is added to the intensities of the foreground. If ``-inv`` key is specified then foreground pixel intensities are inverted. If ``-randinv`` key is specified then it is randomly selected whether for this sample inversion will be applied. Finally, the obtained image is placed onto arbitrary background from the background description file, resized to the pixel size specified by ``-w`` and ``-h`` and stored into the file specified by the ``-vec`` command line parameter.

Positive samples also may be obtained from a collection of previously marked up images. This collection is described by text file similar to background description file. Each line of this file corresponds to collection image. The first element of the line is image file name. It is followed by number of object instances. The following numbers are the coordinates of bounding rectangles (x, y, width, height).

Example of description file:
 
Directory structure:

    .. code-block:: text

        /img
          img1.jpg
          img2.jpg
        info.dat
 
File info.dat:

    .. code-block:: text
    
        img/img1.jpg  1  140 100 45 45
        img/img2.jpg  2  100 200 50 50   50 30 25 25
 
Image img1.jpg contains single object instance with bounding rectangle (140, 100, 45, 45). Image img2.jpg contains two object instances.
 
In order to create positive samples from such collection ``-info`` argument should be specified instead of ``-img``:

* ``-info <collection_file_name>``

    Description file of marked up images collection.
 
The scheme of sample creation in this case is as follows. The object instances are taken from images. Then they are resized to samples size and stored in output file. No distortion is applied, so the only affecting arguments are ``-w``, ``-h``, ``-show`` and ``-num``.
 
createsamples utility may be used for examining samples stored in positive samples file. In order to do this only ``-vec``, ``-w`` and ``-h`` parameters should be specified.
 
Note that for training, it does not matter how positive samples files are generated. So the createsamples utility is only one way to collect/create a vector file of positive samples.

Cascade Training
================
The next step after samples creation is training of classifier. As mentioned above ``opencv_traincascade`` or ``opencv_haartraining`` may be used to train a cascade classifier, but only the newer ``opencv_traincascade`` will be described futher.

Command line arguments of ``opencv_traincascade`` application grouped by purposes:

#.

    Common arguments:
    
    * ``-data <cascade_dir_name>``
    
        Directory name in which the trained classifier is stored.
      
    * ``-vec <vec_file_name>``
    
        File name of positive sample file (created by trainingsamples utility or by any other means).
      
    * ``-bg <background_file_name>``
    
        Background description file.
      
    * ``-numPos <number_of_positive_samples>``
    
    * ``-numNeg <number_of_negative_samples>``
    
        Number of positive/negative samples used in training of each classifier stage.
        
    * ``-numStages <number_of_stages>``
    
        Number of stages to be trained.
        
    * ``-precalcValBufSize <precalculated_vals_buffer_size_in_Mb>``
        
        Size of buffer of precalculated feature values (in Mb).
        
    * ``-precalcIdxBufSize <precalculated_idxs_buffer_size_in_Mb>``
    
        Size of buffer of precalculated feature indices (in Mb). The more memory you have the faster the training process.
        
    * ``-baseFormatSave``
        
        This argument is actual in case of Haar-like features. If it is specified, the cascade will be saved in the old format.
        
#.

    Cascade parameters:

    * ``-stageType <BOOST(default)>``
    
        Type of stages. Only boosted classifier are supported as stage type yet.
        
    * ``-featureType<{HAAR(default), LBP}>``
    
        Type of features: ``HAAR`` - Haar-like features, ``LBP`` - local binary patterns.
    
    * ``-w <sampleWidth>``
    
    * ``-h <sampleHeight>``
    
        Size of training samples (in pixels). Must have exactly the same values as used during training samples creation (utility ``opencv_createsamples``).
        
#.

    Boosted classifer parameters:
    
    * ``-bt <{DAB, RAB, LB, GAB(default)}>``
    
        Type of boosted classifiers: ``DAB`` - Discrete AdaBoost, ``RAB`` - Real AdaBoost, ``LB`` - LogitBoost, ``GAB`` - Gentle AdaBoost.
        
    * ``-minHitRate <min_hit_rate>``
        
        Minimal desired hit rate for each stage classifier. Overall hit rate may be estimated as (min_hit_rate^number_of_stages).
        
    * ``-maxFalseAlarmRate <max_false_alarm_rate>``
    
      Maximal desired false alarm rate for each stage classifier. Overall false alarm rate may be estimated as (max_false_alarm_rate^number_of_stages).
      
    * ``-weightTrimRate <weight_trim_rate>``
    
        Specifies wheter and how much weight trimming should be used. A decent choice is 0.95.
        
    * ``-maxDepth <max_depth_of_weak_tree>``
    
        Maximal depth of weak tree. A decent choice is 1 that is case of stumps.
        
    * ``-maxWeakCount <max_weak_tree_count>``
    
        Maximal count of weak trees for each cascade stage. The boosted classifier (stage) has so many weak trees (``<=maxWeakCount``), so need to achieve the given ``-maxFalseAlarmRate``.
        
#.

    Haar-like feature parameters:
    
    * ``-mode <BASIC (default) | CORE | ALL>``
    
        Selects the type of haar features set used in training. ``BASIC`` use only upright features, while ``ALL`` uses the full set of upright and 45 degree rotated feature set. See [Rainer2002]_ for more details.
    
#.    

    Local Binary Patterns parameters:
    
    Local Binary Patterns have not parameters.

After the ``opencv_traincascade`` application has finished its work, the trained cascade will be saved in file cascade.xml in the folder which was passed as ``-data`` parameter. Other files in this folder are created for possibility of discontinuous training and you may delete them after training completion.

Note ``opencv_traincascade`` application is TBB-parallelized. To use it in multicore mode OpenCV must be built with TBB.

.. [Viola2001] Paul Viola, Michael Jones. *Rapid Object Detection using a Boosted Cascade of Simple Features*. Conference on Computer Vision and Pattern Recognition (CVPR), 2001, pp. 511-518.

.. [Rainer2002] Rainer Lienhart and Jochen Maydt. *An Extended Set of Haar-like Features for Rapid Object Detection*. Submitted to ICIP2002.

.. [Liao2007] Shengcai Liao, Xiangxin Zhu, Zhen Lei, Lun Zhang and Stan Z. Li. *Learning Multi-scale Block Local Binary Patterns for Face Recognition*. International Conference on Biometrics (ICB), 2007, pp. 828-837.
