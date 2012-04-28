Common Interfaces of Descriptor Extractors
==========================================

.. highlight:: cpp

Extractors of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. This section is devoted to computing descriptors
represented as vectors in a multidimensional space. All objects that implement the ``vector``
descriptor extractors inherit the
:ocv:class:`DescriptorExtractor` interface.



DescriptorExtractor
-------------------
.. ocv:class:: DescriptorExtractor

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

    :param keypoints: Input collection of keypoints. Keypoints for which a descriptor cannot be computed are removed. Sometimes new keypoints can be added, for example: ``SIFT`` duplicates keypoint with several dominant orientations (for each orientation).

    :param descriptors: Computed descriptors. In the second variant of the method ``descriptors[i]`` are descriptors computed for a ``keypoints[i]`. Row ``j`` is the ``keypoints`` (or ``keypoints[i]``) is the descriptor for keypoint ``j``-th keypoint.

DescriptorExtractor::read
-----------------------------
Reads the object of a descriptor extractor from a file node.

.. ocv:function:: void DescriptorExtractor::read( const FileNode& fn )

    :param fn: File node from which the detector is read.



DescriptorExtractor::write
------------------------------
Writes the object of a descriptor extractor to a file storage.

.. ocv:function:: void DescriptorExtractor::write( FileStorage& fs ) const

    :param fs: File storage where the detector is written.



DescriptorExtractor::create
-------------------------------
Creates a descriptor extractor by name.

.. ocv:function:: Ptr<DescriptorExtractor>  DescriptorExtractor::create( const string& descriptorExtractorType )

    :param descriptorExtractorType: Descriptor extractor type.

The current implementation supports the following types of a descriptor extractor:

 * ``"SIFT"`` -- :ocv:class:`SiftDescriptorExtractor`
 * ``"SURF"`` -- :ocv:class:`SurfDescriptorExtractor`
 * ``"ORB"`` -- :ocv:class:`OrbDescriptorExtractor`
 * ``"BRIEF"`` -- :ocv:class:`BriefDescriptorExtractor`

A combined format is also supported: descriptor extractor adapter name ( ``"Opponent"`` --
:ocv:class:`OpponentColorDescriptorExtractor` ) + descriptor extractor name (see above),
for example: ``"OpponentSIFT"`` .



SiftDescriptorExtractor
-----------------------
.. ocv:class:: SiftDescriptorExtractor

Wrapping class for computing descriptors by using the
:ocv:class:`SIFT` class. ::

    class SiftDescriptorExtractor : public DescriptorExtractor
    {
    public:
        SiftDescriptorExtractor(
            const SIFT::DescriptorParams& descriptorParams=SIFT::DescriptorParams(),
            const SIFT::CommonParams& commonParams=SIFT::CommonParams() );
        SiftDescriptorExtractor( double magnification, bool isNormalize=true,
            bool recalculateAngles=true, int nOctaves=SIFT::CommonParams::DEFAULT_NOCTAVES,
            int nOctaveLayers=SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS,
            int firstOctave=SIFT::CommonParams::DEFAULT_FIRST_OCTAVE,
            int angleMode=SIFT::CommonParams::FIRST_ANGLE );

        virtual void read (const FileNode &fn);
        virtual void write (FileStorage &fs) const;
        virtual int descriptorSize() const;
        virtual int descriptorType() const;
    protected:
        ...
    }




SurfDescriptorExtractor
-----------------------
.. ocv:class:: SurfDescriptorExtractor

Wrapping class for computing descriptors by using the
:ocv:class:`SURF` class. ::

    class SurfDescriptorExtractor : public DescriptorExtractor
    {
    public:
        SurfDescriptorExtractor( int nOctaves=4,
                                 int nOctaveLayers=2, bool extended=false );

        virtual void read (const FileNode &fn);
        virtual void write (FileStorage &fs) const;
        virtual int descriptorSize() const;
        virtual int descriptorType() const;
    protected:
        ...
    }




OrbDescriptorExtractor
---------------------------
.. ocv:class:: OrbDescriptorExtractor

Wrapping class for computing descriptors by using the
:ocv:class:`ORB` class. ::

    template<typename T>
    class ORbDescriptorExtractor : public DescriptorExtractor
    {
    public:
        OrbDescriptorExtractor( ORB::PatchSize patch_size );

        virtual void read( const FileNode &fn );
        virtual void write( FileStorage &fs ) const;
        virtual int descriptorSize() const;
        virtual int descriptorType() const;
    protected:
        ...
    }



OpponentColorDescriptorExtractor
--------------------------------
.. ocv:class:: OpponentColorDescriptorExtractor

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
.. ocv:class:: BriefDescriptorExtractor

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


