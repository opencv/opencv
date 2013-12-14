Data Structures
=============================

.. ocv:class:: ocl::oclMat

OpenCV C++ 1-D or 2-D dense array class ::

    class CV_EXPORTS oclMat
    {
    public:
        //! default constructor
        oclMat();
        //! constructs oclMatrix of the specified size and type (_type is CV_8UC1, CV_64FC3, CV_32SC(12) etc.)
        oclMat(int rows, int cols, int type);
        oclMat(Size size, int type);
        //! constucts oclMatrix and fills it with the specified value _s.
        oclMat(int rows, int cols, int type, const Scalar &s);
        oclMat(Size size, int type, const Scalar &s);
        //! copy constructor
        oclMat(const oclMat &m);

        //! constructor for oclMatrix headers pointing to user-allocated data
        oclMat(int rows, int cols, int type, void *data, size_t step = Mat::AUTO_STEP);
        oclMat(Size size, int type, void *data, size_t step = Mat::AUTO_STEP);

        //! creates a matrix header for a part of the bigger matrix
        oclMat(const oclMat &m, const Range &rowRange, const Range &colRange);
        oclMat(const oclMat &m, const Rect &roi);

        //! builds oclMat from Mat. Perfom blocking upload to device.
        explicit oclMat (const Mat &m);

        //! destructor - calls release()
        ~oclMat();

        //! assignment operators
        oclMat &operator = (const oclMat &m);
        //! assignment operator. Perfom blocking upload to device.
        oclMat &operator = (const Mat &m);
        oclMat &operator = (const oclMatExpr& expr);

        //! pefroms blocking upload data to oclMat.
        void upload(const cv::Mat &m);


        //! downloads data from device to host memory. Blocking calls.
        operator Mat() const;
        void download(cv::Mat &m) const;

        //! convert to _InputArray
        operator _InputArray();

        //! convert to _OutputArray
        operator _OutputArray();

        //! returns a new oclMatrix header for the specified row
        oclMat row(int y) const;
        //! returns a new oclMatrix header for the specified column
        oclMat col(int x) const;
        //! ... for the specified row span
        oclMat rowRange(int startrow, int endrow) const;
        oclMat rowRange(const Range &r) const;
        //! ... for the specified column span
        oclMat colRange(int startcol, int endcol) const;
        oclMat colRange(const Range &r) const;

        //! returns deep copy of the oclMatrix, i.e. the data is copied
        oclMat clone() const;

        //! copies those oclMatrix elements to "m" that are marked with non-zero mask elements.
        // It calls m.create(this->size(), this->type()).
        // It supports any data type
        void copyTo( oclMat &m, const oclMat &mask = oclMat()) const;

        //! converts oclMatrix to another datatype with optional scalng. See cvConvertScale.
        void convertTo( oclMat &m, int rtype, double alpha = 1, double beta = 0 ) const;

        void assignTo( oclMat &m, int type = -1 ) const;

        //! sets every oclMatrix element to s
        oclMat& operator = (const Scalar &s);
        //! sets some of the oclMatrix elements to s, according to the mask
        oclMat& setTo(const Scalar &s, const oclMat &mask = oclMat());
        //! creates alternative oclMatrix header for the same data, with different
        // number of channels and/or different number of rows. see cvReshape.
        oclMat reshape(int cn, int rows = 0) const;

        //! allocates new oclMatrix data unless the oclMatrix already has specified size and type.
        // previous data is unreferenced if needed.
        void create(int rows, int cols, int type);
        void create(Size size, int type);

        //! allocates new oclMatrix with specified device memory type.
        void createEx(int rows, int cols, int type, DevMemRW rw_type, DevMemType mem_type);
        void createEx(Size size, int type, DevMemRW rw_type, DevMemType mem_type);

        //! decreases reference counter;
        // deallocate the data when reference counter reaches 0.
        void release();

        //! swaps with other smart pointer
        void swap(oclMat &mat);

        //! locates oclMatrix header within a parent oclMatrix. See below
        void locateROI( Size &wholeSize, Point &ofs ) const;
        //! moves/resizes the current oclMatrix ROI inside the parent oclMatrix.
        oclMat& adjustROI( int dtop, int dbottom, int dleft, int dright );
        //! extracts a rectangular sub-oclMatrix
        // (this is a generalized form of row, rowRange etc.)
        oclMat operator()( Range rowRange, Range colRange ) const;
        oclMat operator()( const Rect &roi ) const;

        oclMat& operator+=( const oclMat& m );
        oclMat& operator-=( const oclMat& m );
        oclMat& operator*=( const oclMat& m );
        oclMat& operator/=( const oclMat& m );

        //! returns true if the oclMatrix data is continuous
        // (i.e. when there are no gaps between successive rows).
        // similar to CV_IS_oclMat_CONT(cvoclMat->type)
        bool isContinuous() const;
        //! returns element size in bytes,
        // similar to CV_ELEM_SIZE(cvMat->type)
        size_t elemSize() const;
        //! returns the size of element channel in bytes.
        size_t elemSize1() const;
        //! returns element type, similar to CV_MAT_TYPE(cvMat->type)
        int type() const;
        //! returns element type, i.e. 8UC3 returns 8UC4 because in ocl
        //! 3 channels element actually use 4 channel space
        int ocltype() const;
        //! returns element type, similar to CV_MAT_DEPTH(cvMat->type)
        int depth() const;
        //! returns element type, similar to CV_MAT_CN(cvMat->type)
        int channels() const;
        //! returns element type, return 4 for 3 channels element,
        //!becuase 3 channels element actually use 4 channel space
        int oclchannels() const;
        //! returns step/elemSize1()
        size_t step1() const;
        //! returns oclMatrix size:
        // width == number of columns, height == number of rows
        Size size() const;
        //! returns true if oclMatrix data is NULL
        bool empty() const;

        //! matrix transposition
        oclMat t() const;

        /*! includes several bit-fields:
          - the magic signature
          - continuity flag
          - depth
          - number of channels
          */
        int flags;
        //! the number of rows and columns
        int rows, cols;
        //! a distance between successive rows in bytes; includes the gap if any
        size_t step;
        //! pointer to the data(OCL memory object)
        uchar *data;

        //! pointer to the reference counter;
        // when oclMatrix points to user-allocated data, the pointer is NULL
        int *refcount;

        //! helper fields used in locateROI and adjustROI
        //datastart and dataend are not used in current version
        uchar *datastart;
        uchar *dataend;

        //! OpenCL context associated with the oclMat object.
        Context *clCxt;
        //add offset for handle ROI, calculated in byte
        int offset;
        //add wholerows and wholecols for the whole matrix, datastart and dataend are no longer used
        int wholerows;
        int wholecols;
    };

Basically speaking, the ``oclMat`` is the mirror of ``Mat`` with the extension of OCL feature, the members have the same meaning and useage of ``Mat`` except following:

* ``datastart`` and ``dataend`` are replaced with ``wholerows`` and ``wholecols``

* Only basic flags are supported in ``oclMat`` (i.e. depth number of channels)

* All the 3-channel matrix (i.e. RGB image) are represented by 4-channel matrix in ``oclMat``. It means 3-channel image have 4-channel space with the last channel unused. We provide a transparent interface to handle the difference between OpenCV ``Mat`` and ``oclMat``.
    For example: If a ``oclMat`` has 3 channels, ``channels()`` returns 3 and ``oclchannels()`` returns 4
