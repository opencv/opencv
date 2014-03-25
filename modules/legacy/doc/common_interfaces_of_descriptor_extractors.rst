Common Interfaces of Descriptor Extractors
==========================================

.. highlight:: cpp

Extractors of keypoint descriptors in OpenCV have wrappers with a common interface that enables you to easily switch
between different algorithms solving the same problem. This section is devoted to computing descriptors
represented as vectors in a multidimensional space. All objects that implement the ``vector``
descriptor extractors inherit the
:ocv:class:`DescriptorExtractor` interface.



CalonderDescriptorExtractor
---------------------------
.. ocv:class:: CalonderDescriptorExtractor : public DescriptorExtractor

Wrapping class for computing descriptors by using the
:ocv:class:`RTreeClassifier` class. ::

    template<typename T>
    class CalonderDescriptorExtractor : public DescriptorExtractor
    {
    public:
        CalonderDescriptorExtractor( const String& classifierFile );

        virtual void read( const FileNode &fn );
        virtual void write( FileStorage &fs ) const;
        virtual int descriptorSize() const;
        virtual int descriptorType() const;
        virtual int defaultNorm() const;
    protected:
        ...
    }
