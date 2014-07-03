Common Interfaces of Descriptor Extractors
==========================================

.. highlight:: cpp

Extractors of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. This section is devoted to computing descriptors
represented as vectors in a multidimensional space. All objects that implement the ``vector``
descriptor extractors inherit the
:ocv:class:`DescriptorExtractor` interface.

.. note::

   * An example explaining keypoint extraction can be found at opencv_source_code/samples/cpp/descriptor_extractor_matcher.cpp
   * An example on descriptor evaluation can be found at opencv_source_code/samples/cpp/detector_descriptor_evaluation.cpp

DescriptorExtractor
-------------------
.. ocv:class:: DescriptorExtractor : public Algorithm

Abstract base class for computing descriptors for image keypoints. ::

    class CV_EXPORTS DescriptorExtractor
    {
    public:
        virtual ~DescriptorExtractor();

        void compute( const Mat& image, vector<KeyPoint>& keypoints,
                      Mat& descriptors ) const;
        void compute( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints,
                      vector<Mat>& descriptors ) const;

        virtual void read( const FileNode& );
        virtual void write( FileStorage& ) const;

        virtual int descriptorSize() const = 0;
        virtual int descriptorType() const = 0;

        static Ptr<DescriptorExtractor> create( const string& descriptorExtractorType );

    protected:
        ...
    };


In this interface, a keypoint descriptor can be represented as a
dense, fixed-dimension vector of a basic type. Most descriptors
follow this pattern as it simplifies computing
distances between descriptors. Therefore, a collection of
descriptors is represented as
:ocv:class:`Mat` , where each row is a keypoint descriptor.



DescriptorExtractor::compute
--------------------------------
Computes the descriptors for a set of keypoints detected in an image (first variant) or image set (second variant).

.. ocv:function:: void DescriptorExtractor::compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const

.. ocv:function:: void DescriptorExtractor::compute( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, vector<Mat>& descriptors ) const

    :param image: Image.

    :param images: Image set.

    :param keypoints: Input collection of keypoints. Keypoints for which a descriptor cannot be computed are removed and the remaining ones may be reordered. Sometimes new keypoints can be added, for example: ``SIFT`` duplicates a keypoint with several dominant orientations (for each orientation).

    :param descriptors: Computed descriptors. In the second variant of the method ``descriptors[i]`` are descriptors computed for a ``keypoints[i]``. Row ``j`` is the ``keypoints`` (or ``keypoints[i]``) is the descriptor for keypoint ``j``-th keypoint.


DescriptorExtractor::create
-------------------------------
Creates a descriptor extractor by name.

.. ocv:function:: Ptr<DescriptorExtractor>  DescriptorExtractor::create( const string& descriptorExtractorType )

    :param descriptorExtractorType: Descriptor extractor type.

The current implementation supports the following types of a descriptor extractor:

 * ``"SIFT"`` -- :ocv:class:`SIFT`
 * ``"SURF"`` -- :ocv:class:`SURF`
 * ``"BRIEF"`` -- :ocv:class:`BriefDescriptorExtractor`
 * ``"BRISK"`` -- :ocv:class:`BRISK`
 * ``"ORB"`` -- :ocv:class:`ORB`
 * ``"FREAK"`` -- :ocv:class:`FREAK`

A combined format is also supported: descriptor extractor adapter name ( ``"Opponent"`` --
:ocv:class:`OpponentColorDescriptorExtractor` ) + descriptor extractor name (see above),
for example: ``"OpponentSIFT"`` .


OpponentColorDescriptorExtractor
--------------------------------
.. ocv:class:: OpponentColorDescriptorExtractor : public DescriptorExtractor

Class adapting a descriptor extractor to compute descriptors in the Opponent Color Space
(refer to Van de Sande et al., CGIV 2008 *Color Descriptors for Object Category Recognition*).
Input RGB image is transformed in the Opponent Color Space. Then, an unadapted descriptor extractor
(set in the constructor) computes descriptors on each of three channels and concatenates
them into a single color descriptor. ::

    class OpponentColorDescriptorExtractor : public DescriptorExtractor
    {
    public:
        OpponentColorDescriptorExtractor( const Ptr<DescriptorExtractor>& dextractor );

        virtual void read( const FileNode& );
        virtual void write( FileStorage& ) const;
        virtual int descriptorSize() const;
        virtual int descriptorType() const;
    protected:
        ...
    };



BriefDescriptorExtractor
------------------------
.. ocv:class:: BriefDescriptorExtractor : public DescriptorExtractor

Class for computing BRIEF descriptors described in a paper of Calonder M., Lepetit V.,
Strecha C., Fua P. *BRIEF: Binary Robust Independent Elementary Features* ,
11th European Conference on Computer Vision (ECCV), Heraklion, Crete. LNCS Springer, September 2010. ::

    class BriefDescriptorExtractor : public DescriptorExtractor
    {
    public:
        static const int PATCH_SIZE = 48;
        static const int KERNEL_SIZE = 9;

        // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes.
        BriefDescriptorExtractor( int bytes = 32 );

        virtual void read( const FileNode& );
        virtual void write( FileStorage& ) const;
        virtual int descriptorSize() const;
        virtual int descriptorType() const;
    protected:
        ...
    };

.. note::

   * A complete BRIEF extractor sample can be found at opencv_source_code/samples/cpp/brief_match_test.cpp
