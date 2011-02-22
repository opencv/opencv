XML/YAML Persistence
====================

.. highlight:: cpp



.. index:: FileStorage

.. _FileStorage:

FileStorage
-----------

`id=0.36488878292 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/FileStorage>`__

.. ctype:: FileStorage



The XML/YAML file storage class




::


    
    class FileStorage
    {
    public:
        enum { READ=0, WRITE=1, APPEND=2 };
        enum { UNDEFINED=0, VALUE_EXPECTED=1, NAME_EXPECTED=2, INSIDE_MAP=4 };
        // the default constructor
        FileStorage();
        // the constructor that opens the file for reading
        // (flags=FileStorage::READ) or writing (flags=FileStorage::WRITE)
        FileStorage(const string& filename, int flags);
        // wraps the already opened CvFileStorage*
        FileStorage(CvFileStorage* fs);
        // the destructor; closes the file if needed
        virtual ~FileStorage();
    
        // opens the specified file for reading (flags=FileStorage::READ)
        // or writing (flags=FileStorage::WRITE)
        virtual bool open(const string& filename, int flags);
        // checks if the storage is opened
        virtual bool isOpened() const;
        // closes the file
        virtual void release();
    
        // returns the first top-level node
        FileNode getFirstTopLevelNode() const;
        // returns the root file node
        // (it's the parent of the first top-level node)
        FileNode root(int streamidx=0) const;
        // returns the top-level node by name
        FileNode operator[](const string& nodename) const;
        FileNode operator[](const char* nodename) const;
    
        // returns the underlying CvFileStorage*
        CvFileStorage* operator *() { return fs; }
        const CvFileStorage* operator *() const { return fs; }
        
        // writes the certain number of elements of the specified format
        // (see DataType) without any headers
        void writeRaw( const string& fmt, const uchar* vec, size_t len );
        
        // writes an old-style object (CvMat, CvMatND etc.)
        void writeObj( const string& name, const void* obj );
    
        // returns the default object name from the filename
        // (used by cvSave() with the default object name etc.)
        static string getDefaultObjectName(const string& filename);
    
        Ptr<CvFileStorage> fs;
        string elname;
        vector<char> structs;
        int state;
    };
    

..


.. index:: FileNode

.. _FileNode:

FileNode
--------

`id=0.228849909258 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/FileNode>`__

.. ctype:: FileNode



The XML/YAML file node class




::


    
    class CV_EXPORTS FileNode
    {
    public:
        enum { NONE=0, INT=1, REAL=2, FLOAT=REAL, STR=3,
            STRING=STR, REF=4, SEQ=5, MAP=6, TYPE_MASK=7,
            FLOW=8, USER=16, EMPTY=32, NAMED=64 };
        FileNode();
        FileNode(const CvFileStorage* fs, const CvFileNode* node);
        FileNode(const FileNode& node);
        FileNode operator[](const string& nodename) const;
        FileNode operator[](const char* nodename) const;
        FileNode operator[](int i) const;
        int type() const;
        int rawDataSize(const string& fmt) const;
        bool empty() const;
        bool isNone() const;
        bool isSeq() const;
        bool isMap() const;
        bool isInt() const;
        bool isReal() const;
        bool isString() const;
        bool isNamed() const;
        string name() const;
        size_t size() const;
        operator int() const;
        operator float() const;
        operator double() const;
        operator string() const;
    
        FileNodeIterator begin() const;
        FileNodeIterator end() const;
    
        void readRaw( const string& fmt, uchar* vec, size_t len ) const;
        void* readObj() const;
    
        // do not use wrapper pointer classes for better efficiency
        const CvFileStorage* fs;
        const CvFileNode* node;
    };
    

..


.. index:: FileNodeIterator

.. _FileNodeIterator:

FileNodeIterator
----------------

`id=0.575104633905 Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/cpp/core/FileNodeIterator>`__

.. ctype:: FileNodeIterator



The XML/YAML file node iterator class




::


    
    class CV_EXPORTS FileNodeIterator
    {
    public:
        FileNodeIterator();
        FileNodeIterator(const CvFileStorage* fs,
            const CvFileNode* node, size_t ofs=0);
        FileNodeIterator(const FileNodeIterator& it);
        FileNode operator *() const;
        FileNode operator ->() const;
    
        FileNodeIterator& operator ++();
        FileNodeIterator operator ++(int);
        FileNodeIterator& operator --();
        FileNodeIterator operator --(int);
        FileNodeIterator& operator += (int);
        FileNodeIterator& operator -= (int);
    
        FileNodeIterator& readRaw( const string& fmt, uchar* vec,
                                   size_t maxCount=(size_t)INT_MAX );
    
        const CvFileStorage* fs;
        const CvFileNode* container;
        CvSeqReader reader;
        size_t remaining;
    };
    

..

