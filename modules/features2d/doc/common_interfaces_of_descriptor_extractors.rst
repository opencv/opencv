Common Interfaces of Descriptor Extractors
==========================================

.. highlight:: cpp

Extractors of keypoint descriptors in OpenCV have wrappers with common interface that enables to switch easily
between different algorithms solving the same problem. This section is devoted to computing descriptors
that are represented as vectors in a multidimensional space. All objects that implement ''vector''
descriptor extractors inherit
:func:`DescriptorExtractor` interface.

.. index:: DescriptorExtractor

.. _DescriptorExtractor:

DescriptorExtractor
-------------------
.. c:type:: DescriptorExtractor

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
..

In this interface we assume a keypoint descriptor can be represented as a
dense, fixed-dimensional vector of some basic type. Most descriptors used
in practice follow this pattern, as it makes it very easy to compute
distances between descriptors. Therefore we represent a collection of
descriptors as a
:func:`Mat` , where each row is one keypoint descriptor.

.. index:: DescriptorExtractor::compute

DescriptorExtractor::compute
--------------------------------
.. c:function:: void DescriptorExtractor::compute( const Mat\& image,                                      vector<KeyPoint>\& keypoints,                                                                      Mat\& descriptors ) const

    Compute the descriptors for a set of keypoints detected in an image (first variant)
or image set (second variant).

    :param image: The image.

    :param keypoints: The keypoints. Keypoints for which a descriptor cannot be computed are removed.

    :param descriptors: The descriptors. Row i is the descriptor for keypoint i.

.. c:function:: void DescriptorExtractor::compute( const vector<Mat>\& images,                                                           vector<vector<KeyPoint> >\& keypoints,                                                       vector<Mat>\& descriptors ) const

    * **images** The image set.

    * **keypoints** Input keypoints collection. keypoints[i] is keypoints
                          detected in images[i]. Keypoints for which a descriptor
                          can not be computed are removed.

    * **descriptors** Descriptor collection. descriptors[i] are descriptors computed for
                            a set keypoints[i].

.. index:: DescriptorExtractor::read

DescriptorExtractor::read
-----------------------------
.. c:function:: void DescriptorExtractor::read( const FileNode\& fn )

    Read descriptor extractor object from file node.

    :param fn: File node from which detector will be read.

.. index:: DescriptorExtractor::write

DescriptorExtractor::write
------------------------------
.. c:function:: void DescriptorExtractor::write( FileStorage\& fs ) const

    Write descriptor extractor object to file storage.

    :param fs: File storage in which detector will be written.

.. index:: DescriptorExtractor::create

DescriptorExtractor::create
-------------------------------
:func:`DescriptorExtractor`
.. c:function:: Ptr<DescriptorExtractor>  DescriptorExtractor::create( const string\& descriptorExtractorType )

    Descriptor extractor factory that creates of given type with
default parameters (rather using default constructor).

    :param descriptorExtractorType: Descriptor extractor type.

Now the following descriptor extractor types are supported:
\ ``"SIFT"`` --
:func:`SiftFeatureDetector`,\ ``"SURF"`` --
:func:`SurfFeatureDetector`,\ ``"BRIEF"`` --
:func:`BriefFeatureDetector` .
\
Also combined format is supported: descriptor extractor adapter name ( ``"Opponent"`` --
:func:`OpponentColorDescriptorExtractor` ) + descriptor extractor name (see above),
e.g. ``"OpponentSIFT"`` , etc.

.. index:: SiftDescriptorExtractor

.. _SiftDescriptorExtractor:

SiftDescriptorExtractor
-----------------------
.. c:type:: SiftDescriptorExtractor

Wrapping class for descriptors computing using
:func:`SIFT` class. ::

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
..

.. index:: SurfDescriptorExtractor

.. _SurfDescriptorExtractor:

SurfDescriptorExtractor
-----------------------
.. c:type:: SurfDescriptorExtractor

Wrapping class for descriptors computing using
:func:`SURF` class. ::

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
..

.. index:: CalonderDescriptorExtractor

.. _CalonderDescriptorExtractor:

CalonderDescriptorExtractor
---------------------------
.. c:type:: CalonderDescriptorExtractor

Wrapping class for descriptors computing using
:func:`RTreeClassifier` class. ::

    template<typename T>
    class CalonderDescriptorExtractor : public DescriptorExtractor
    {
    public:
        CalonderDescriptorExtractor( const string& classifierFile );

        virtual void read( const FileNode &fn );
        virtual void write( FileStorage &fs ) const;
        virtual int descriptorSize() const;
        virtual int descriptorType() const;
    protected:
        ...
    }
..

.. index:: OpponentColorDescriptorExtractor

.. _OpponentColorDescriptorExtractor:

OpponentColorDescriptorExtractor
--------------------------------
.. c:type:: OpponentColorDescriptorExtractor

Adapts a descriptor extractor to compute descripors in Opponent Color Space
(refer to van de Sande et al., CGIV 2008 "Color Descriptors for Object Category Recognition").
Input RGB image is transformed in Opponent Color Space. Then unadapted descriptor extractor
(set in constructor) computes descriptors on each of the three channel and concatenate
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
..

.. index:: BriefDescriptorExtractor

.. _BriefDescriptorExtractor:

BriefDescriptorExtractor
------------------------
.. c:type:: BriefDescriptorExtractor

Class for computing BRIEF descriptors described in paper of Calonder M., Lepetit V.,
Strecha C., Fua P.: ''BRIEF: Binary Robust Independent Elementary Features.''
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
..

