/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_CORE_PERSISTENCE_HPP__
#define __OPENCV_CORE_PERSISTENCE_HPP__

#ifndef __cplusplus
#  error persistence.hpp header must be compiled as C++
#endif

// black-box structures used by FileStorage
typedef struct CvFileStorage CvFileStorage;
typedef struct CvFileNode CvFileNode;

#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"

namespace cv {

////////////////////////// XML & YAML I/O //////////////////////////

class CV_EXPORTS FileNode;
class CV_EXPORTS FileNodeIterator;

/*!
 XML/YAML File Storage Class.

 The class describes an object associated with XML or YAML file.
 It can be used to store data to such a file or read and decode the data.

 The storage is organized as a tree of nested sequences (or lists) and mappings.
 Sequence is a heterogenious array, which elements are accessed by indices or sequentially using an iterator.
 Mapping is analogue of std::map or C structure, which elements are accessed by names.
 The most top level structure is a mapping.
 Leaves of the file storage tree are integers, floating-point numbers and text strings.

 For example, the following code:

 \code
 // open file storage for writing. Type of the file is determined from the extension
 FileStorage fs("test.yml", FileStorage::WRITE);
 fs << "test_int" << 5 << "test_real" << 3.1 << "test_string" << "ABCDEFGH";
 fs << "test_mat" << Mat::eye(3,3,CV_32F);

 fs << "test_list" << "[" << 0.0000000000001 << 2 << CV_PI << -3435345 << "2-502 2-029 3egegeg" <<
 "{:" << "month" << 12 << "day" << 31 << "year" << 1969 << "}" << "]";
 fs << "test_map" << "{" << "x" << 1 << "y" << 2 << "width" << 100 << "height" << 200 << "lbp" << "[:";

 const uchar arr[] = {0, 1, 1, 0, 1, 1, 0, 1};
 fs.writeRaw("u", arr, (int)(sizeof(arr)/sizeof(arr[0])));

 fs << "]" << "}";
 \endcode

 will produce the following file:

 \verbatim
 %YAML:1.0
 test_int: 5
 test_real: 3.1000000000000001e+00
 test_string: ABCDEFGH
 test_mat: !!opencv-matrix
     rows: 3
     cols: 3
     dt: f
     data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]
 test_list:
     - 1.0000000000000000e-13
     - 2
     - 3.1415926535897931e+00
     - -3435345
     - "2-502 2-029 3egegeg"
     - { month:12, day:31, year:1969 }
 test_map:
     x: 1
     y: 2
     width: 100
     height: 200
     lbp: [ 0, 1, 1, 0, 1, 1, 0, 1 ]
 \endverbatim

 and to read the file above, the following code can be used:

 \code
 // open file storage for reading.
 // Type of the file is determined from the content, not the extension
 FileStorage fs("test.yml", FileStorage::READ);
 int test_int = (int)fs["test_int"];
 double test_real = (double)fs["test_real"];
 String test_string = (String)fs["test_string"];

 Mat M;
 fs["test_mat"] >> M;

 FileNode tl = fs["test_list"];
 CV_Assert(tl.type() == FileNode::SEQ && tl.size() == 6);
 double tl0 = (double)tl[0];
 int tl1 = (int)tl[1];
 double tl2 = (double)tl[2];
 int tl3 = (int)tl[3];
 String tl4 = (String)tl[4];
 CV_Assert(tl[5].type() == FileNode::MAP && tl[5].size() == 3);

 int month = (int)tl[5]["month"];
 int day = (int)tl[5]["day"];
 int year = (int)tl[5]["year"];

 FileNode tm = fs["test_map"];

 int x = (int)tm["x"];
 int y = (int)tm["y"];
 int width = (int)tm["width"];
 int height = (int)tm["height"];

 int lbp_val = 0;
 FileNodeIterator it = tm["lbp"].begin();

 for(int k = 0; k < 8; k++, ++it)
    lbp_val |= ((int)*it) << k;
 \endcode
*/
class CV_EXPORTS_W FileStorage
{
public:
    //! file storage mode
    enum
    {
        READ        = 0, //! read mode
        WRITE       = 1, //! write mode
        APPEND      = 2, //! append mode
        MEMORY      = 4,
        FORMAT_MASK = (7<<3),
        FORMAT_AUTO = 0,
        FORMAT_XML  = (1<<3),
        FORMAT_YAML = (2<<3)
    };
    enum
    {
        UNDEFINED      = 0,
        VALUE_EXPECTED = 1,
        NAME_EXPECTED  = 2,
        INSIDE_MAP     = 4
    };
    //! the default constructor
    CV_WRAP FileStorage();
    //! the full constructor that opens file storage for reading or writing
    CV_WRAP FileStorage(const String& source, int flags, const String& encoding=String());
    //! the constructor that takes pointer to the C FileStorage structure
    FileStorage(CvFileStorage* fs, bool owning=true);
    //! the destructor. calls release()
    virtual ~FileStorage();

    //! opens file storage for reading or writing. The previous storage is closed with release()
    CV_WRAP virtual bool open(const String& filename, int flags, const String& encoding=String());
    //! returns true if the object is associated with currently opened file.
    CV_WRAP virtual bool isOpened() const;
    //! closes the file and releases all the memory buffers
    CV_WRAP virtual void release();
    //! closes the file, releases all the memory buffers and returns the text string
    CV_WRAP virtual String releaseAndGetString();

    //! returns the first element of the top-level mapping
    CV_WRAP FileNode getFirstTopLevelNode() const;
    //! returns the top-level mapping. YAML supports multiple streams
    CV_WRAP FileNode root(int streamidx=0) const;
    //! returns the specified element of the top-level mapping
    FileNode operator[](const String& nodename) const;
    //! returns the specified element of the top-level mapping
    CV_WRAP FileNode operator[](const char* nodename) const;

    //! returns pointer to the underlying C FileStorage structure
    CvFileStorage* operator *() { return fs.get(); }
    //! returns pointer to the underlying C FileStorage structure
    const CvFileStorage* operator *() const { return fs.get(); }
    //! writes one or more numbers of the specified format to the currently written structure
    void writeRaw( const String& fmt, const uchar* vec, size_t len );
    //! writes the registered C structure (CvMat, CvMatND, CvSeq). See cvWrite()
    void writeObj( const String& name, const void* obj );

    //! returns the normalized object name for the specified file name
    static String getDefaultObjectName(const String& filename);

    Ptr<CvFileStorage> fs; //!< the underlying C FileStorage structure
    String elname; //!< the currently written element
    std::vector<char> structs; //!< the stack of written structures
    int state; //!< the writer state
};

template<> CV_EXPORTS void DefaultDeleter<CvFileStorage>::operator ()(CvFileStorage* obj) const;

/*!
 File Storage Node class

 The node is used to store each and every element of the file storage opened for reading -
 from the primitive objects, such as numbers and text strings, to the complex nodes:
 sequences, mappings and the registered objects.

 Note that file nodes are only used for navigating file storages opened for reading.
 When a file storage is opened for writing, no data is stored in memory after it is written.
*/
class CV_EXPORTS_W_SIMPLE FileNode
{
public:
    //! type of the file storage node
    enum
    {
        NONE      = 0, //!< empty node
        INT       = 1, //!< an integer
        REAL      = 2, //!< floating-point number
        FLOAT     = REAL, //!< synonym or REAL
        STR       = 3, //!< text string in UTF-8 encoding
        STRING    = STR, //!< synonym for STR
        REF       = 4, //!< integer of size size_t. Typically used for storing complex dynamic structures where some elements reference the others
        SEQ       = 5, //!< sequence
        MAP       = 6, //!< mapping
        TYPE_MASK = 7,
        FLOW      = 8,  //!< compact representation of a sequence or mapping. Used only by YAML writer
        USER      = 16, //!< a registered object (e.g. a matrix)
        EMPTY     = 32, //!< empty structure (sequence or mapping)
        NAMED     = 64  //!< the node has a name (i.e. it is element of a mapping)
    };
    //! the default constructor
    CV_WRAP FileNode();
    //! the full constructor wrapping CvFileNode structure.
    FileNode(const CvFileStorage* fs, const CvFileNode* node);
    //! the copy constructor
    FileNode(const FileNode& node);
    //! returns element of a mapping node
    FileNode operator[](const String& nodename) const;
    //! returns element of a mapping node
    CV_WRAP FileNode operator[](const char* nodename) const;
    //! returns element of a sequence node
    CV_WRAP FileNode operator[](int i) const;
    //! returns type of the node
    CV_WRAP int type() const;

    //! returns true if the node is empty
    CV_WRAP bool empty() const;
    //! returns true if the node is a "none" object
    CV_WRAP bool isNone() const;
    //! returns true if the node is a sequence
    CV_WRAP bool isSeq() const;
    //! returns true if the node is a mapping
    CV_WRAP bool isMap() const;
    //! returns true if the node is an integer
    CV_WRAP bool isInt() const;
    //! returns true if the node is a floating-point number
    CV_WRAP bool isReal() const;
    //! returns true if the node is a text string
    CV_WRAP bool isString() const;
    //! returns true if the node has a name
    CV_WRAP bool isNamed() const;
    //! returns the node name or an empty string if the node is nameless
    CV_WRAP String name() const;
    //! returns the number of elements in the node, if it is a sequence or mapping, or 1 otherwise.
    CV_WRAP size_t size() const;
    //! returns the node content as an integer. If the node stores floating-point number, it is rounded.
    operator int() const;
    //! returns the node content as float
    operator float() const;
    //! returns the node content as double
    operator double() const;
    //! returns the node content as text string
    operator String() const;
#ifndef OPENCV_NOSTL
    operator std::string() const;
#endif

    //! returns pointer to the underlying file node
    CvFileNode* operator *();
    //! returns pointer to the underlying file node
    const CvFileNode* operator* () const;

    //! returns iterator pointing to the first node element
    FileNodeIterator begin() const;
    //! returns iterator pointing to the element following the last node element
    FileNodeIterator end() const;

    //! reads node elements to the buffer with the specified format
    void readRaw( const String& fmt, uchar* vec, size_t len ) const;
    //! reads the registered object and returns pointer to it
    void* readObj() const;

    // do not use wrapper pointer classes for better efficiency
    const CvFileStorage* fs;
    const CvFileNode* node;
};


/*!
 File Node Iterator

 The class is used for iterating sequences (usually) and mappings.
 */
class CV_EXPORTS FileNodeIterator
{
public:
    //! the default constructor
    FileNodeIterator();
    //! the full constructor set to the ofs-th element of the node
    FileNodeIterator(const CvFileStorage* fs, const CvFileNode* node, size_t ofs=0);
    //! the copy constructor
    FileNodeIterator(const FileNodeIterator& it);
    //! returns the currently observed element
    FileNode operator *() const;
    //! accesses the currently observed element methods
    FileNode operator ->() const;

    //! moves iterator to the next node
    FileNodeIterator& operator ++ ();
    //! moves iterator to the next node
    FileNodeIterator operator ++ (int);
    //! moves iterator to the previous node
    FileNodeIterator& operator -- ();
    //! moves iterator to the previous node
    FileNodeIterator operator -- (int);
    //! moves iterator forward by the specified offset (possibly negative)
    FileNodeIterator& operator += (int ofs);
    //! moves iterator backward by the specified offset (possibly negative)
    FileNodeIterator& operator -= (int ofs);

    //! reads the next maxCount elements (or less, if the sequence/mapping last element occurs earlier) to the buffer with the specified format
    FileNodeIterator& readRaw( const String& fmt, uchar* vec,
                               size_t maxCount=(size_t)INT_MAX );

    struct SeqReader
    {
      int          header_size;
      void*        seq;        /* sequence, beign read; CvSeq      */
      void*        block;      /* current block;        CvSeqBlock */
      schar*       ptr;        /* pointer to element be read next */
      schar*       block_min;  /* pointer to the beginning of block */
      schar*       block_max;  /* pointer to the end of block */
      int          delta_index;/* = seq->first->start_index   */
      schar*       prev_elem;  /* pointer to previous element */
    };

    const CvFileStorage* fs;
    const CvFileNode* container;
    SeqReader reader;
    size_t remaining;
};



/////////////////// XML & YAML I/O implementation //////////////////

CV_EXPORTS void write( FileStorage& fs, const String& name, int value );
CV_EXPORTS void write( FileStorage& fs, const String& name, float value );
CV_EXPORTS void write( FileStorage& fs, const String& name, double value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const String& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const Mat& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const SparseMat& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const std::vector<KeyPoint>& value);

CV_EXPORTS void writeScalar( FileStorage& fs, int value );
CV_EXPORTS void writeScalar( FileStorage& fs, float value );
CV_EXPORTS void writeScalar( FileStorage& fs, double value );
CV_EXPORTS void writeScalar( FileStorage& fs, const String& value );

CV_EXPORTS void read(const FileNode& node, int& value, int default_value);
CV_EXPORTS void read(const FileNode& node, float& value, float default_value);
CV_EXPORTS void read(const FileNode& node, double& value, double default_value);
CV_EXPORTS void read(const FileNode& node, String& value, const String& default_value);
CV_EXPORTS void read(const FileNode& node, Mat& mat, const Mat& default_mat = Mat() );
CV_EXPORTS void read(const FileNode& node, SparseMat& mat, const SparseMat& default_mat = SparseMat() );
CV_EXPORTS void read(const FileNode& node, std::vector<KeyPoint>& keypoints);

template<typename _Tp> static inline void read(const FileNode& node, Point_<_Tp>& value, const Point_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 2 ? default_value : Point_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]));
}

template<typename _Tp> static inline void read(const FileNode& node, Point3_<_Tp>& value, const Point3_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 3 ? default_value : Point3_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]),
                                                            saturate_cast<_Tp>(temp[2]));
}

template<typename _Tp> static inline void read(const FileNode& node, Size_<_Tp>& value, const Size_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 2 ? default_value : Size_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]));
}

template<typename _Tp> static inline void read(const FileNode& node, Complex<_Tp>& value, const Complex<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 2 ? default_value : Complex<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]));
}

template<typename _Tp> static inline void read(const FileNode& node, Rect_<_Tp>& value, const Rect_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 4 ? default_value : Rect_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]),
                                                          saturate_cast<_Tp>(temp[2]), saturate_cast<_Tp>(temp[3]));
}

template<typename _Tp, int cn> static inline void read(const FileNode& node, Vec<_Tp, cn>& value, const Vec<_Tp, cn>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != cn ? default_value : Vec<_Tp, cn>(&temp[0]);
}

template<typename _Tp> static inline void read(const FileNode& node, Scalar_<_Tp>& value, const Scalar_<_Tp>& default_value)
{
    std::vector<_Tp> temp; FileNodeIterator it = node.begin(); it >> temp;
    value = temp.size() != 4 ? default_value : Scalar_<_Tp>(saturate_cast<_Tp>(temp[0]), saturate_cast<_Tp>(temp[1]),
                                                            saturate_cast<_Tp>(temp[2]), saturate_cast<_Tp>(temp[3]));
}

static inline void read(const FileNode& node, Range& value, const Range& default_value)
{
    Point2i temp(value.start, value.end); const Point2i default_temp = Point2i(default_value.start, default_value.end);
    read(node, temp, default_temp);
    value.start = temp.x; value.end = temp.y;
}


CV_EXPORTS FileStorage& operator << (FileStorage& fs, const String& str);


namespace internal
{
    class CV_EXPORTS WriteStructContext
    {
    public:
        WriteStructContext(FileStorage& _fs, const String& name, int flags, const String& typeName = String());
        ~WriteStructContext();
    private:
        FileStorage* fs;
    };

    template<typename _Tp, int numflag> class VecWriterProxy
    {
    public:
        VecWriterProxy( FileStorage* _fs ) : fs(_fs) {}
        void operator()(const std::vector<_Tp>& vec) const
        {
            size_t count = vec.size();
            for (size_t i = 0; i < count; i++)
                write(*fs, vec[i]);
        }
    private:
        FileStorage* fs;
    };

    template<typename _Tp> class VecWriterProxy<_Tp, 1>
    {
    public:
        VecWriterProxy( FileStorage* _fs ) : fs(_fs) {}
        void operator()(const std::vector<_Tp>& vec) const
        {
            int _fmt = DataType<_Tp>::fmt;
            char fmt[] = { (char)((_fmt >> 8) + '1'), (char)_fmt, '\0' };
            fs->writeRaw(fmt, !vec.empty() ? (uchar*)&vec[0] : 0, vec.size() * sizeof(_Tp));
        }
    private:
        FileStorage* fs;
    };

    template<typename _Tp, int numflag> class VecReaderProxy
    {
    public:
        VecReaderProxy( FileNodeIterator* _it ) : it(_it) {}
        void operator()(std::vector<_Tp>& vec, size_t count) const
        {
            count = std::min(count, it->remaining);
            vec.resize(count);
            for (size_t i = 0; i < count; i++, ++(*it))
                read(**it, vec[i], _Tp());
        }
    private:
        FileNodeIterator* it;
    };

    template<typename _Tp> class VecReaderProxy<_Tp, 1>
    {
    public:
        VecReaderProxy( FileNodeIterator* _it ) : it(_it) {}
        void operator()(std::vector<_Tp>& vec, size_t count) const
        {
            size_t remaining = it->remaining;
            size_t cn = DataType<_Tp>::channels;
            int _fmt = DataType<_Tp>::fmt;
            char fmt[] = { (char)((_fmt >> 8)+'1'), (char)_fmt, '\0' };
            size_t remaining1 = remaining / cn;
            count = count < remaining1 ? count : remaining1;
            vec.resize(count);
            it->readRaw(fmt, !vec.empty() ? (uchar*)&vec[0] : 0, count*sizeof(_Tp));
        }
    private:
        FileNodeIterator* it;
    };

} // internal



template<typename _Tp> static inline
void write(FileStorage& fs, const _Tp& value)
{
    write(fs, String(), value);
}

template<> inline
void write( FileStorage& fs, const int& value )
{
    writeScalar(fs, value);
}

template<> inline
void write( FileStorage& fs, const float& value )
{
    writeScalar(fs, value);
}

template<> inline
void write( FileStorage& fs, const double& value )
{
    writeScalar(fs, value);
}

template<> inline
void write( FileStorage& fs, const String& value )
{
    writeScalar(fs, value);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Point_<_Tp>& pt )
{
    write(fs, pt.x);
    write(fs, pt.y);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Point3_<_Tp>& pt )
{
    write(fs, pt.x);
    write(fs, pt.y);
    write(fs, pt.z);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Size_<_Tp>& sz )
{
    write(fs, sz.width);
    write(fs, sz.height);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Complex<_Tp>& c )
{
    write(fs, c.re);
    write(fs, c.im);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Rect_<_Tp>& r )
{
    write(fs, r.x);
    write(fs, r.y);
    write(fs, r.width);
    write(fs, r.height);
}

template<typename _Tp, int cn> static inline
void write(FileStorage& fs, const Vec<_Tp, cn>& v )
{
    for(int i = 0; i < cn; i++)
        write(fs, v.val[i]);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const Scalar_<_Tp>& s )
{
    write(fs, s.val[0]);
    write(fs, s.val[1]);
    write(fs, s.val[2]);
    write(fs, s.val[3]);
}

static inline
void write(FileStorage& fs, const Range& r )
{
    write(fs, r.start);
    write(fs, r.end);
}

template<typename _Tp> static inline
void write( FileStorage& fs, const std::vector<_Tp>& vec )
{
    internal::VecWriterProxy<_Tp, DataType<_Tp>::fmt != 0> w(&fs);
    w(vec);
}


template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Point_<_Tp>& pt )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, pt);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Point3_<_Tp>& pt )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, pt);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Size_<_Tp>& sz )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, sz);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Complex<_Tp>& c )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, c);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Rect_<_Tp>& r )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r);
}

template<typename _Tp, int cn> static inline
void write(FileStorage& fs, const String& name, const Vec<_Tp, cn>& v )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, v);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Scalar_<_Tp>& s )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, s);
}

static inline
void write(FileStorage& fs, const String& name, const Range& r )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r);
}

template<typename _Tp> static inline
void write( FileStorage& fs, const String& name, const std::vector<_Tp>& vec )
{
    internal::WriteStructContext ws(fs, name, FileNode::SEQ+(DataType<_Tp>::fmt != 0 ? FileNode::FLOW : 0));
    write(fs, vec);
}


static inline
void read(const FileNode& node, bool& value, bool default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = temp != 0;
}

static inline
void read(const FileNode& node, uchar& value, uchar default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = saturate_cast<uchar>(temp);
}

static inline
void read(const FileNode& node, schar& value, schar default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = saturate_cast<schar>(temp);
}

static inline
void read(const FileNode& node, ushort& value, ushort default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = saturate_cast<ushort>(temp);
}

static inline
void read(const FileNode& node, short& value, short default_value)
{
    int temp;
    read(node, temp, (int)default_value);
    value = saturate_cast<short>(temp);
}

template<typename _Tp> static inline
void read( FileNodeIterator& it, std::vector<_Tp>& vec, size_t maxCount = (size_t)INT_MAX )
{
    internal::VecReaderProxy<_Tp, DataType<_Tp>::fmt != 0> r(&it);
    r(vec, maxCount);
}

template<typename _Tp> static inline
void read( const FileNode& node, std::vector<_Tp>& vec, const std::vector<_Tp>& default_value = std::vector<_Tp>() )
{
    if(!node.node)
        vec = default_value;
    else
    {
        FileNodeIterator it = node.begin();
        read( it, vec );
    }
}


template<typename _Tp> static inline
FileStorage& operator << (FileStorage& fs, const _Tp& value)
{
    if( !fs.isOpened() )
        return fs;
    if( fs.state == FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP )
        CV_Error( Error::StsError, "No element name has been given" );
    write( fs, fs.elname, value );
    if( fs.state & FileStorage::INSIDE_MAP )
        fs.state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
    return fs;
}

static inline
FileStorage& operator << (FileStorage& fs, const char* str)
{
    return (fs << String(str));
}

static inline
FileStorage& operator << (FileStorage& fs, char* value)
{
    return (fs << String(value));
}

template<typename _Tp> static inline
FileNodeIterator& operator >> (FileNodeIterator& it, _Tp& value)
{
    read( *it, value, _Tp());
    return ++it;
}

template<typename _Tp> static inline
FileNodeIterator& operator >> (FileNodeIterator& it, std::vector<_Tp>& vec)
{
    internal::VecReaderProxy<_Tp, DataType<_Tp>::fmt != 0> r(&it);
    r(vec, (size_t)INT_MAX);
    return it;
}

template<typename _Tp> static inline
void operator >> (const FileNode& n, _Tp& value)
{
    read( n, value, _Tp());
}

template<typename _Tp> static inline
void operator >> (const FileNode& n, std::vector<_Tp>& vec)
{
    FileNodeIterator it = n.begin();
    it >> vec;
}


static inline
bool operator == (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it1.fs == it2.fs && it1.container == it2.container &&
        it1.reader.ptr == it2.reader.ptr && it1.remaining == it2.remaining;
}

static inline
bool operator != (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return !(it1 == it2);
}

static inline
ptrdiff_t operator - (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it2.remaining - it1.remaining;
}

static inline
bool operator < (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it1.remaining > it2.remaining;
}

inline FileNode FileStorage::getFirstTopLevelNode() const { FileNode r = root(); FileNodeIterator it = r.begin(); return it != r.end() ? *it : FileNode(); }
inline FileNode::FileNode() : fs(0), node(0) {}
inline FileNode::FileNode(const CvFileStorage* _fs, const CvFileNode* _node) : fs(_fs), node(_node) {}
inline FileNode::FileNode(const FileNode& _node) : fs(_node.fs), node(_node.node) {}
inline bool FileNode::empty() const    { return node   == 0;    }
inline bool FileNode::isNone() const   { return type() == NONE; }
inline bool FileNode::isSeq() const    { return type() == SEQ;  }
inline bool FileNode::isMap() const    { return type() == MAP;  }
inline bool FileNode::isInt() const    { return type() == INT;  }
inline bool FileNode::isReal() const   { return type() == REAL; }
inline bool FileNode::isString() const { return type() == STR;  }
inline CvFileNode* FileNode::operator *() { return (CvFileNode*)node; }
inline const CvFileNode* FileNode::operator* () const { return node; }
inline FileNode::operator int() const    { int value;    read(*this, value, 0);     return value; }
inline FileNode::operator float() const  { float value;  read(*this, value, 0.f);   return value; }
inline FileNode::operator double() const { double value; read(*this, value, 0.);    return value; }
inline FileNode::operator String() const { String value; read(*this, value, value); return value; }
inline FileNodeIterator FileNode::begin() const { return FileNodeIterator(fs, node); }
inline FileNodeIterator FileNode::end() const   { return FileNodeIterator(fs, node, size()); }
inline void FileNode::readRaw( const String& fmt, uchar* vec, size_t len ) const { begin().readRaw( fmt, vec, len ); }
inline FileNode FileNodeIterator::operator *() const  { return FileNode(fs, (const CvFileNode*)(const void*)reader.ptr); }
inline FileNode FileNodeIterator::operator ->() const { return FileNode(fs, (const CvFileNode*)(const void*)reader.ptr); }
inline String::String(const FileNode& fn): cstr_(0), len_(0) { read(fn, *this, *this); }

} // cv

#endif // __OPENCV_CORE_PERSISTENCE_HPP__
