Cascade Classifier Training {#tutorial_traincascade}
===========================

Introduction
------------

The work with a cascade classifier inlcudes two major stages: training and detection. Detection
stage is described in a documentation of objdetect module of general OpenCV documentation.
Documentation gives some basic information about cascade classifier. Current guide is describing how
to train a cascade classifier: preparation of the training data and running the training application.

### Important notes

There are two applications in OpenCV to train cascade classifier: opencv_haartraining and
opencv_traincascade. opencv_traincascade is a newer version, written in C++ in accordance to
OpenCV 2.x API. But the main difference between this two applications is that opencv_traincascade
supports both Haar @cite Viola01 and @cite Liao2007 (Local Binary Patterns) features. LBP features
are integer in contrast to Haar features, so both training and detection with LBP are several times
faster then with Haar features. Regarding the LBP and Haar detection quality, it depends on
training: the quality of training dataset first of all and training parameters too. It's possible to
train a LBP-based classifier that will provide almost the same quality as Haar-based one.

opencv_traincascade and opencv_haartraining store the trained classifier in different file
formats. Note, the newer cascade detection interface (see CascadeClassifier class in objdetect
module) support both formats. opencv_traincascade can save (export) a trained cascade in the older
format. But opencv_traincascade and opencv_haartraining can not load (import) a classifier in
another format for the futher training after interruption.

Note that opencv_traincascade application can use TBB for multi-threading. To use it in multicore
mode OpenCV must be built with TBB.

Also there are some auxilary utilities related to the training.

-   opencv_createsamples is used to prepare a training dataset of positive and test samples.
    opencv_createsamples produces dataset of positive samples in a format that is supported by
    both opencv_haartraining and opencv_traincascade applications. The output is a file
    with \*.vec extension, it is a binary format which contains images.
-   opencv_performance may be used to evaluate the quality of classifiers, but for trained by
    opencv_haartraining only. It takes a collection of marked up images, runs the classifier and
    reports the performance, i.e. number of found objects, number of missed objects, number of
    false alarms and other information.

Since opencv_haartraining is an obsolete application, only opencv_traincascade will be described
futher. opencv_createsamples utility is needed to prepare a training data for opencv_traincascade,
so it will be described too.

Training data preparation
-------------------------

For training we need a set of samples. There are two types of samples: negative and positive.
Negative samples correspond to non-object images. Positive samples correspond to images with
detected objects. Set of negative samples must be prepared manually, whereas set of positive samples
is created using opencv_createsamples utility.

### Negative Samples

Negative samples are taken from arbitrary images. These images must not contain detected objects.
Negative samples are enumerated in a special file. It is a text file in which each line contains an
image filename (relative to the directory of the description file) of negative sample image. This
file must be created manually. Note that negative samples and sample images are also called
background samples or background images, and are used interchangeably in this document.
Described images may be of different sizes. But each image should be (but not nessesarily) larger
than a training window size, because these images are used to subsample negative image to the
training size.

An example of description file:

Directory structure:
@code{.text}
/img
  img1.jpg
  img2.jpg
bg.txt
@endcode
File bg.txt:
@code{.text}
img/img1.jpg
img/img2.jpg
@endcode
### Positive Samples

Positive samples are created by opencv_createsamples utility. They may be created from a single
image with object or from a collection of previously marked up images.

Please note that you need a large dataset of positive samples before you give it to the mentioned
utility, because it only applies perspective transformation. For example you may need only one
positive sample for absolutely rigid object like an OpenCV logo, but you definetely need hundreds
and even thousands of positive samples for faces. In the case of faces you should consider all the
race and age groups, emotions and perhaps beard styles.

So, a single object image may contain a company logo. Then a large set of positive samples is
created from the given object image by random rotating, changing the logo intensity as well as
placing the logo on arbitrary background. The amount and range of randomness can be controlled by
command line arguments of opencv_createsamples utility.

Command line arguments:

-   -vec \<vec_file_name\>

    Name of the output file containing the positive samples for training.

-   -img \<image_file_name\>

    Source object image (e.g., a company logo).

-   -bg \<background_file_name\>

    Background description file; contains a list of images which are used as a background for
    randomly distorted versions of the object.

-   -num \<number_of_samples\>

    Number of positive samples to generate.

-   -bgcolor \<background_color\>

    Background color (currently grayscale images are assumed); the background color denotes the
    transparent color. Since there might be compression artifacts, the amount of color tolerance
    can be specified by -bgthresh. All pixels withing bgcolor-bgthresh and bgcolor+bgthresh range
    are interpreted as transparent.

-   -bgthresh \<background_color_threshold\>
-   -inv

    If specified, colors will be inverted.

-   -randinv

    If specified, colors will be inverted randomly.

-   -maxidev \<max_intensity_deviation\>

    Maximal intensity deviation of pixels in foreground samples.

-   -maxxangle \<max_x_rotation_angle\>
-   -maxyangle \<max_y_rotation_angle\>
-   -maxzangle \<max_z_rotation_angle\>

    Maximum rotation angles must be given in radians.

-   -show

    Useful debugging option. If specified, each sample will be shown. Pressing Esc will continue
    the samples creation process without.

-   -w \<sample_width\>

    Width (in pixels) of the output samples.

-   -h \<sample_height\>

    Height (in pixels) of the output samples.

For following procedure is used to create a sample object instance: The source image is rotated
randomly around all three axes. The chosen angle is limited my -max?angle. Then pixels having the
intensity from [bg_color-bg_color_threshold; bg_color+bg_color_threshold] range are
interpreted as transparent. White noise is added to the intensities of the foreground. If the -inv
key is specified then foreground pixel intensities are inverted. If -randinv key is specified then
algorithm randomly selects whether inversion should be applied to this sample. Finally, the obtained
image is placed onto an arbitrary background from the background description file, resized to the
desired size specified by -w and -h and stored to the vec-file, specified by the -vec command line
option.

Positive samples also may be obtained from a collection of previously marked up images. This
collection is described by a text file similar to background description file. Each line of this
file corresponds to an image. The first element of the line is the filename. It is followed by the
number of object instances. The following numbers are the coordinates of objects bounding rectangles
(x, y, width, height).

An example of description file:

Directory structure:
@code{.text}
/img
  img1.jpg
  img2.jpg
info.dat
@endcode
File info.dat:
@code{.text}
img/img1.jpg  1  140 100 45 45
img/img2.jpg  2  100 200 50 50   50 30 25 25
@endcode
Image img1.jpg contains single object instance with the following coordinates of bounding rectangle:
(140, 100, 45, 45). Image img2.jpg contains two object instances.

In order to create positive samples from such collection, -info argument should be specified instead
of \`-img\`:

-   -info \<collection_file_name\>

    Description file of marked up images collection.

The scheme of samples creation in this case is as follows. The object instances are taken from
images. Then they are resized to target samples size and stored in output vec-file. No distortion is
applied, so the only affecting arguments are -w, -h, -show and -num.

opencv_createsamples utility may be used for examining samples stored in positive samples file. In
order to do this only -vec, -w and -h parameters should be specified.

Note that for training, it does not matter how vec-files with positive samples are generated. But
opencv_createsamples utility is the only one way to collect/create a vector file of positive
samples, provided by OpenCV.

Example of vec-file is available here opencv/data/vec_files/trainingfaces_24-24.vec. It can be
used to train a face detector with the following window size: -w 24 -h 24.

Cascade Training
----------------

The next step is the training of classifier. As mentioned above opencv_traincascade or
opencv_haartraining may be used to train a cascade classifier, but only the newer
opencv_traincascade will be described futher.

Command line arguments of opencv_traincascade application grouped by purposes:

-#  Common arguments:

    -   -data \<cascade_dir_name\>

        Where the trained classifier should be stored.

    -   -vec \<vec_file_name\>

        vec-file with positive samples (created by opencv_createsamples utility).

    -   -bg \<background_file_name\>

        Background description file.

    -   -numPos \<number_of_positive_samples\>
    -   -numNeg \<number_of_negative_samples\>

        Number of positive/negative samples used in training for every classifier stage.

    -   -numStages \<number_of_stages\>

        Number of cascade stages to be trained.

    -   -precalcValBufSize \<precalculated_vals_buffer_size_in_Mb\>

        Size of buffer for precalculated feature values (in Mb).

    -   -precalcIdxBufSize \<precalculated_idxs_buffer_size_in_Mb\>

        Size of buffer for precalculated feature indices (in Mb). The more memory you have the
        faster the training process.

    -   -baseFormatSave

        This argument is actual in case of Haar-like features. If it is specified, the cascade will
        be saved in the old format.

    -   -numThreads \<max_number_of_threads\>

        Maximum number of threads to use during training. Notice that the actual number of used
        threads may be lower, depending on your machine and compilation options.

    -   -acceptanceRatioBreakValue \<break_value\>

        This argument is used to determine how precise your model should keep learning and when to stop.
        A good guideline is to train not further than 10e-5, to ensure the model does not overtrain on your training data.
        By default this value is set to -1 to disable this feature.

-#  Cascade parameters:

    -   -stageType \<BOOST(default)\>

        Type of stages. Only boosted classifier are supported as a stage type at the moment.

    -   -featureType\<{HAAR(default), LBP}\>

        Type of features: HAAR - Haar-like features, LBP - local binary patterns.

    -   -w \<sampleWidth\>
    -   -h \<sampleHeight\>

        Size of training samples (in pixels). Must have exactly the same values as used during
        training samples creation (opencv_createsamples utility).

-#  Boosted classifer parameters:

    -   -bt \<{DAB, RAB, LB, GAB(default)}\>

        Type of boosted classifiers: DAB - Discrete AdaBoost, RAB - Real AdaBoost, LB - LogitBoost,
        GAB - Gentle AdaBoost.

    -   -minHitRate \<min_hit_rate\>

        Minimal desired hit rate for each stage of the classifier. Overall hit rate may be estimated
        as (min_hit_rate\^number_of_stages).

    -   -maxFalseAlarmRate \<max_false_alarm_rate\>

        Maximal desired false alarm rate for each stage of the classifier. Overall false alarm rate
        may be estimated as (max_false_alarm_rate\^number_of_stages).

    -   -weightTrimRate \<weight_trim_rate\>

        Specifies whether trimming should be used and its weight. A decent choice is 0.95.

    -   -maxDepth \<max_depth_of_weak_tree\>

        Maximal depth of a weak tree. A decent choice is 1, that is case of stumps.

    -   -maxWeakCount \<max_weak_tree_count\>

        Maximal count of weak trees for every cascade stage. The boosted classifier (stage) will
        have so many weak trees (\<=maxWeakCount), as needed to achieve the
        given -maxFalseAlarmRate.

-#  Haar-like feature parameters:

    -   -mode \<BASIC (default) | CORE | ALL\>

        Selects the type of Haar features set used in training. BASIC use only upright features,
        while ALL uses the full set of upright and 45 degree rotated feature set. See @cite Lienhart02
        for more details.

-#  Local Binary Patterns parameters:

    Local Binary Patterns don't have parameters.

After the opencv_traincascade application has finished its work, the trained cascade will be saved
in cascade.xml file in the folder, which was passed as -data parameter. Other files in this folder
are created for the case of interrupted training, so you may delete them after completion of
training.

Training is finished and you can test you cascade classifier!
