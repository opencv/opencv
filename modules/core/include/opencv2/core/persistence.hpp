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

#ifndef OPENCV_CORE_PERSISTENCE_HPP
#define OPENCV_CORE_PERSISTENCE_HPP

#ifndef CV_DOXYGEN
/// Define to support persistence legacy formats
#define CV__LEGACY_PERSISTENCE
#endif

#ifndef __cplusplus
#  error persistence.hpp header must be compiled as C++
#endif

#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"

namespace cv {

/** @addtogroup core_xml

XML/YAML/JSON file storages.     {#xml_storage}
=======================
Writing to a file storage.
--------------------------
You can store and then restore various OpenCV data structures to/from XML (<http://www.w3c.org/XML>),
YAML (<http://www.yaml.org>) or JSON (<http://www.json.org/>) formats. Also, it is possible to store
and load arbitrarily complex data structures, which include OpenCV data structures, as well as
primitive data types (integer and floating-point numbers and text strings) as their elements.

Use the following procedure to write something to XML, YAML or JSON:
-# Create new FileStorage and open it for writing. It can be done with a single call to
FileStorage::FileStorage constructor that takes a filename, or you can use the default constructor
and then call FileStorage::open. Format of the file (XML, YAML or JSON) is determined from the filename
extension (".xml", ".yml"/".yaml" and ".json", respectively)
-# Write all the data you want using the streaming operator `<<`, just like in the case of STL
streams.
-# Close the file using FileStorage::release. FileStorage destructor also closes the file.

Here is an example:
@code
    #include "opencv2/core.hpp"
    #include <time.h>

    using namespace cv;

    int main(int, char** argv)
    {
        FileStorage fs("test.yml", FileStorage::WRITE);

        fs << "frameCount" << 5;
        time_t rawtime; time(&rawtime);
        fs << "calibrationDate" << asctime(localtime(&rawtime));
        Mat cameraMatrix = (Mat_<double>(3,3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
        Mat distCoeffs = (Mat_<double>(5,1) << 0.1, 0.01, -0.001, 0, 0);
        fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
        fs << "features" << "[";
        for( int i = 0; i < 3; i++ )
        {
            int x = rand() % 640;
            int y = rand() % 480;
            uchar lbp = rand() % 256;

            fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
            for( int j = 0; j < 8; j++ )
                fs << ((lbp >> j) & 1);
            fs << "]" << "}";
        }
        fs << "]";
        fs.release();
        return 0;
    }
@endcode
The sample above stores to YML an integer, a text string (calibration date), 2 matrices, and a custom
structure "feature", which includes feature coordinates and LBP (local binary pattern) value. Here
is output of the sample:
@code{.yaml}
%YAML:1.0
frameCount: 5
calibrationDate: "Fri Jun 17 14:09:29 2011\n"
cameraMatrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1000., 0., 320., 0., 1000., 240., 0., 0., 1. ]
distCoeffs: !!opencv-matrix
   rows: 5
   cols: 1
   dt: d
   data: [ 1.0000000000000001e-01, 1.0000000000000000e-02,
       -1.0000000000000000e-03, 0., 0. ]
features:
   - { x:167, y:49, lbp:[ 1, 0, 0, 1, 1, 0, 1, 1 ] }
   - { x:298, y:130, lbp:[ 0, 0, 0, 1, 0, 0, 1, 1 ] }
   - { x:344, y:158, lbp:[ 1, 1, 0, 0, 0, 0, 1, 0 ] }
@endcode

As an exercise, you can replace ".yml" with ".xml" or ".json" in the sample above and see, how the
corresponding XML file will look like.

Several things can be noted by looking at the sample code and the output:

-   The produced YAML (and XML/JSON) consists of heterogeneous collections that can be nested. There are
    2 types of collections: named collections (mappings) and unnamed collections (sequences). In mappings
    each element has a name and is accessed by name. This is similar to structures and std::map in
    C/C++ and dictionaries in Python. In sequences elements do not have names, they are accessed by
    indices. This is similar to arrays and std::vector in C/C++ and lists, tuples in Python.
    "Heterogeneous" means that elements of each single collection can have different types.

    Top-level collection in YAML/XML/JSON is a mapping. Each matrix is stored as a mapping, and the matrix
    elements are stored as a sequence. Then, there is a sequence of features, where each feature is
    represented a mapping, and lbp value in a nested sequence.

-   When you write to a mapping (a structure), you write element name followed by its value. When you
    write to a sequence, you simply write the elements one by one. OpenCV data structures (such as
    cv::Mat) are written in absolutely the same way as simple C data structures - using `<<`
    operator.

-   To write a mapping, you first write the special string `{` to the storage, then write the
    elements as pairs (`fs << <element_name> << <element_value>`) and then write the closing
    `}`.

-   To write a sequence, you first write the special string `[`, then write the elements, then
    write the closing `]`.

-   In YAML/JSON (but not XML), mappings and sequences can be written in a compact Python-like inline
    form. In the sample above matrix elements, as well as each feature, including its lbp value, is
    stored in such inline form. To store a mapping/sequence in a compact form, put `:` after the
    opening character, e.g. use `{:` instead of `{` and `[:` instead of `[`. When the
    data is written to XML, those extra `:` are ignored.

Reading data from a file storage.
---------------------------------
To read the previously written XML, YAML or JSON file, do the following:
-#  Open the file storage using FileStorage::FileStorage constructor or FileStorage::open method.
    In the current implementation the whole file is parsed and the whole representation of file
    storage is built in memory as a hierarchy of file nodes (see FileNode)

-#  Read the data you are interested in. Use FileStorage::operator [], FileNode::operator []
    and/or FileNodeIterator.

-#  Close the storage using FileStorage::release.

Here is how to read the file created by the code sample above:
@code
    FileStorage fs2("test.yml", FileStorage::READ);

    // first method: use (type) operator on FileNode.
    int frameCount = (int)fs2["frameCount"];

    String date;
    // second method: use FileNode::operator >>
    fs2["calibrationDate"] >> date;

    Mat cameraMatrix2, distCoeffs2;
    fs2["cameraMatrix"] >> cameraMatrix2;
    fs2["distCoeffs"] >> distCoeffs2;

    cout << "frameCount: " << frameCount << endl
         << "calibration date: " << date << endl
         << "camera matrix: " << cameraMatrix2 << endl
         << "distortion coeffs: " << distCoeffs2 << endl;

    FileNode features = fs2["features"];
    FileNodeIterator it = features.begin(), it_end = features.end();
    int idx = 0;
    std::vector<uchar> lbpval;

    // iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it, idx++ )
    {
        cout << "feature #" << idx << ": ";
        cout << "x=" << (int)(*it)["x"] << ", y=" << (int)(*it)["y"] << ", lbp: (";
        // you can also easily read numerical arrays using FileNode >> std::vector operator.
        (*it)["lbp"] >> lbpval;
        for( int i = 0; i < (int)lbpval.size(); i++ )
            cout << " " << (int)lbpval[i];
        cout << ")" << endl;
    }
    fs2.release();
@endcode

Format specification    {#format_spec}
--------------------
`([count]{u|c|w|s|i|f|d})`... where the characters correspond to fundamental C++ types:
-   `u` 8-bit unsigned number
-   `c` 8-bit signed number
-   `w` 16-bit unsigned number
-   `s` 16-bit signed number
-   `i` 32-bit signed number
-   `f` single precision floating-point number
-   `d` double precision floating-point number
-   `r` pointer, 32 lower bits of which are written as a signed integer. The type can be used to
    store structures with links between the elements.

`count` is the optional counter of values of a given type. For example, `2if` means that each array
element is a structure of 2 integers, followed by a single-precision floating-point number. The
equivalent notations of the above specification are `iif`, `2i1f` and so forth. Other examples: `u`
means that the array consists of bytes, and `2d` means the array consists of pairs of doubles.

@see @ref samples/cpp/tutorial_code/core/file_input_output/file_input_output.cpp
*/

//! @{

/** @example samples/cpp/tutorial_code/core/file_input_output/file_input_output.cpp
A complete example using the FileStorage interface
Check @ref tutorial_file_input_output_with_xml_yml "the corresponding tutorial" for more details
*/

////////////////////////// XML & YAML I/O //////////////////////////

class CV_EXPORTS FileNode;
class CV_EXPORTS FileNodeIterator;

/** @brief XML/YAML/JSON file storage class that encapsulates all the information necessary for writing or
reading data to/from a file.
 */
class CV_EXPORTS_W FileStorage
{
public:
    //! file storage mode
    enum Mode
    {
        READ        = 0, //!< value, open the file for reading
        WRITE       = 1, //!< value, open the file for writing
        APPEND      = 2, //!< value, open the file for appending
        MEMORY      = 4, /**< flag, read data from source or write data to the internal buffer (which is
                              returned by FileStorage::release) */
        FORMAT_MASK = (7<<3), //!< mask for format flags
        FORMAT_AUTO = 0,      //!< flag, auto format
        FORMAT_XML  = (1<<3), //!< flag, XML format
        FORMAT_YAML = (2<<3), //!< flag, YAML format
        FORMAT_JSON = (3<<3), //!< flag, JSON format

        BASE64      = 64,     //!< flag, write rawdata in Base64 by default. (consider using WRITE_BASE64)
        WRITE_BASE64 = BASE64 | WRITE, //!< flag, enable both WRITE and BASE64
    };
    enum State
    {
        UNDEFINED      = 0,  //!< Initial or uninitialized state.
        VALUE_EXPECTED = 1,  //!< Expecting a value in the current position.
        NAME_EXPECTED  = 2,  //!< Expecting a key/name in the current position.
        INSIDE_MAP     = 4   //!< Indicates being inside a map (a set of key-value pairs).
    };

    /** @brief The constructors.

     The full constructor opens the file. Alternatively you can use the default constructor and then
     call FileStorage::open.
     */
    CV_WRAP FileStorage();

    /** @overload
     @copydoc open()
     */
    CV_WRAP FileStorage(const String& filename, int flags, const String& encoding=String());

    //! the destructor. calls release()
    virtual ~FileStorage();

    /** @brief Opens a file.

     See description of parameters in FileStorage::FileStorage. The method calls FileStorage::release
     before opening the file.
     @param filename Name of the file to open or the text string to read the data from.
     Extension of the file (.xml, .yml/.yaml or .json) determines its format (XML, YAML or JSON
     respectively). Also you can append .gz to work with compressed files, for example myHugeMatrix.xml.gz. If both
     FileStorage::WRITE and FileStorage::MEMORY flags are specified, source is used just to specify
     the output file format (e.g. mydata.xml, .yml etc.). A file name can also contain parameters.
     You can use this format, "*?base64" (e.g. "file.json?base64" (case sensitive)), as an alternative to
     FileStorage::BASE64 flag.
     @param flags Mode of operation. One of FileStorage::Mode
     @param encoding Encoding of the file. Note that UTF-16 XML encoding is not supported currently and
     you should use 8-bit encoding instead of it.
     */
    CV_WRAP virtual bool open(const String& filename, int flags, const String& encoding=String());

    /** @brief Checks whether the file is opened.

     @returns true if the object is associated with the current file and false otherwise. It is a
     good practice to call this method after you tried to open a file.
     */
    CV_WRAP virtual bool isOpened() const;

    /** @brief Closes the file and releases all the memory buffers.

     Call this method after all I/O operations with the storage are finished.
     */
    CV_WRAP virtual void release();

    /** @brief Closes the file and releases all the memory buffers.

     Call this method after all I/O operations with the storage are finished. If the storage was
     opened for writing data and FileStorage::WRITE was specified
     */
    CV_WRAP virtual String releaseAndGetString();

    /** @brief Returns the first element of the top-level mapping.
     @returns The first element of the top-level mapping.
     */
    CV_WRAP FileNode getFirstTopLevelNode() const;

    /** @brief Returns the top-level mapping
     @param streamidx Zero-based index of the stream. In most cases there is only one stream in the file.
     However, YAML supports multiple streams and so there can be several.
     @returns The top-level mapping.
     */
    CV_WRAP FileNode root(int streamidx=0) const;

    /** @brief Returns the specified element of the top-level mapping.
     @param nodename Name of the file node.
     @returns Node with the given name.
     */
    FileNode operator[](const String& nodename) const;

    /** @overload */
    CV_WRAP_AS(getNode) FileNode operator[](const char* nodename) const;

    /**
     * @brief Simplified writing API to use with bindings.
     * @param name Name of the written object. When writing to sequences (a.k.a. "arrays"), pass an empty string.
     * @param val Value of the written object.
     */
    CV_WRAP void write(const String& name, int val);
    /// @overload
    CV_WRAP void write(const String& name, int64_t val);
    /// @overload
    CV_WRAP void write(const String& name, double val);
    /// @overload
    CV_WRAP void write(const String& name, const String& val);
    /// @overload
    CV_WRAP void write(const String& name, const Mat& val);
    /// @overload
    CV_WRAP void write(const String& name, const std::vector<String>& val);

    /** @brief Writes multiple numbers.

     Writes one or more numbers of the specified format to the currently written structure. Usually it is
     more convenient to use operator `<<` instead of this method.
     @param fmt Specification of each array element, see @ref format_spec "format specification"
     @param vec Pointer to the written array.
     @param len Number of the uchar elements to write.
     */
    void writeRaw( const String& fmt, const void* vec, size_t len );

    /** @brief Writes a comment.

     The function writes a comment into file storage. The comments are skipped when the storage is read.
     @param comment The written comment, single-line or multi-line
     @param append If true, the function tries to put the comment at the end of current line.
     Else if the comment is multi-line, or if it does not fit at the end of the current
     line, the comment starts a new line.
     */
    CV_WRAP void writeComment(const String& comment, bool append = false);

    /** @brief Starts to write a nested structure (sequence or a mapping).
    @param name name of the structure. When writing to sequences (a.k.a. "arrays"), pass an empty string.
    @param flags type of the structure (FileNode::MAP or FileNode::SEQ (both with optional FileNode::FLOW)).
    @param typeName optional name of the type you store. The effect of setting this depends on the storage format.
    I.e. if the format has a specification for storing type information, this parameter is used.
    */
    CV_WRAP void startWriteStruct(const String& name, int flags, const String& typeName=String());

    /** @brief Finishes writing nested structure (should pair startWriteStruct())
    */
    CV_WRAP void endWriteStruct();

    /** @brief Returns the normalized object name for the specified name of a file.
    @param filename Name of a file
    @returns The normalized object name.
     */
    static String getDefaultObjectName(const String& filename);

    /** @brief Returns the current format.
     * @returns The current format, see FileStorage::Mode
     */
    CV_WRAP int getFormat() const;

    int state;
    std::string elname;

    class Impl;
    Ptr<Impl> p;
};

/** @brief File Storage Node class.

The node is used to store each and every element of the file storage opened for reading. When
XML/YAML file is read, it is first parsed and stored in the memory as a hierarchical collection of
nodes. Each node can be a "leaf" that is contain a single number or a string, or be a collection of
other nodes. There can be named collections (mappings) where each element has a name and it is
accessed by a name, and ordered collections (sequences) where elements do not have names but rather
accessed by index. Type of the file node can be determined using FileNode::type method.

Note that file nodes are only used for navigating file storages opened for reading. When a file
storage is opened for writing, no data is stored in memory after it is written.
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
        SEQ       = 4, //!< sequence
        MAP       = 5, //!< mapping
        TYPE_MASK = 7,

        FLOW      = 8,  //!< compact representation of a sequence or mapping. Used only by YAML writer
        UNIFORM   = 8,  //!< if set, means that all the collection elements are numbers of the same type (real's or int's).
        //!< UNIFORM is used only when reading FileStorage; FLOW is used only when writing. So they share the same bit
        EMPTY     = 16, //!< empty structure (sequence or mapping)
        NAMED     = 32  //!< the node has a name (i.e. it is element of a mapping).
    };
    /** @brief The constructors.

     These constructors are used to create a default file node, construct it from obsolete structures or
     from the another file node.
     */
    CV_WRAP FileNode();

    /** @overload
     @param fs Pointer to the file storage structure.
     @param blockIdx Index of the memory block where the file node is stored
     @param ofs Offset in bytes from the beginning of the serialized storage

     @deprecated
     */
    FileNode(const FileStorage* fs, size_t blockIdx, size_t ofs);

    /** @overload
     @param node File node to be used as initialization for the created file node.
     */
    FileNode(const FileNode& node);

    FileNode& operator=(const FileNode& node);

    /** @brief Returns element of a mapping node or a sequence node.
     @param nodename Name of an element in the mapping node.
     @returns Returns the element with the given identifier.
     */
    FileNode operator[](const String& nodename) const;

    /** @overload
     @param nodename Name of an element in the mapping node.
     */
    CV_WRAP_AS(getNode) FileNode operator[](const char* nodename) const;

    /** @overload
     @param i Index of an element in the sequence node.
     */
    CV_WRAP_AS(at) FileNode operator[](int i) const;

    /** @brief Returns keys of a mapping node.
     @returns Keys of a mapping node.
     */
    CV_WRAP std::vector<String> keys() const;

    /** @brief Returns type of the node.
     @returns Type of the node. See FileNode::Type
     */
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
    CV_WRAP std::string name() const;
    //! returns the number of elements in the node, if it is a sequence or mapping, or 1 otherwise.
    CV_WRAP size_t size() const;
    //! returns raw size of the FileNode in bytes
    CV_WRAP size_t rawSize() const;
    //! returns the node content as an integer. If the node stores floating-point number, it is rounded.
    operator int() const;
    //! returns the node content as a signed 64bit integer. If the node stores floating-point number, it is rounded.
    operator int64_t() const;
    //! returns the node content as float
    operator float() const;
    //! returns the node content as double
    operator double() const;
    //! returns the node content as text string
    inline operator std::string() const { return this->string(); }

    static bool isMap(int flags);
    static bool isSeq(int flags);
    static bool isCollection(int flags);
    static bool isEmptyCollection(int flags);
    static bool isFlow(int flags);

    uchar* ptr();
    const uchar* ptr() const;

    //! returns iterator pointing to the first node element
    FileNodeIterator begin() const;
    //! returns iterator pointing to the element following the last node element
    FileNodeIterator end() const;

    /** @brief Reads node elements to the buffer with the specified format.

    Usually it is more convenient to use operator `>>` instead of this method.
    @param fmt Specification of each array element. See @ref format_spec "format specification"
    @param vec Pointer to the destination array.
    @param len Number of bytes to read (buffer size limit). If it is greater than number of
               remaining elements then all of them will be read.
     */
    void readRaw( const String& fmt, void* vec, size_t len ) const;

    /** Internal method used when reading FileStorage.
     Sets the type (int, real or string) and value of the previously created node.
     */
    void setValue( int type, const void* value, int len=-1 );

    //! Simplified reading API to use with bindings.
    CV_WRAP double real() const;
    //! Simplified reading API to use with bindings.
    CV_WRAP std::string string() const;
    //! Simplified reading API to use with bindings.
    CV_WRAP Mat mat() const;

    //protected:
    FileNode(FileStorage::Impl* fs, size_t blockIdx, size_t ofs);

    FileStorage::Impl* fs;
    size_t blockIdx;
    size_t ofs;
};


/** @brief used to iterate through sequences and mappings.

 A standard STL notation, with node.begin(), node.end() denoting the beginning and the end of a
 sequence, stored in node. See the data reading sample in the beginning of the section.
 */
class CV_EXPORTS FileNodeIterator
{
public:
    /** @brief The constructors.

     These constructors are used to create a default iterator, set it to specific element in a file node
     or construct it from another iterator.
     */
    FileNodeIterator();

    /** @overload
     @param node File node - the collection to iterate over;
        it can be a scalar (equivalent to 1-element collection) or "none" (equivalent to empty collection).
     @param seekEnd - true if iterator needs to be set after the last element of the node;
        that is:
            * node.begin() => FileNodeIterator(node, false)
            * node.end() => FileNodeIterator(node, true)
     */
    FileNodeIterator(const FileNode& node, bool seekEnd);

    /** @overload
     @param it Iterator to be used as initialization for the created iterator.
     */
    FileNodeIterator(const FileNodeIterator& it);

    FileNodeIterator& operator=(const FileNodeIterator& it);

    //! returns the currently observed element
    FileNode operator *() const;

    //! moves iterator to the next node
    FileNodeIterator& operator ++ ();
    //! moves iterator to the next node
    FileNodeIterator operator ++ (int);
    //! moves iterator forward by the specified offset (possibly negative)
    FileNodeIterator& operator += (int ofs);

    /** @brief Reads node elements to the buffer with the specified format.

    Usually it is more convenient to use operator `>>` instead of this method.
    @param fmt Specification of each array element. See @ref format_spec "format specification"
    @param vec Pointer to the destination array.
    @param len Number of bytes to read (buffer size limit). If it is greater than number of
               remaining elements then all of them will be read.
     */
    FileNodeIterator& readRaw( const String& fmt, void* vec,
                               size_t len=(size_t)INT_MAX );

    //! returns the number of remaining (not read yet) elements
    size_t remaining() const;

    bool equalTo(const FileNodeIterator& it) const;

protected:
    FileStorage::Impl* fs;
    size_t blockIdx;
    size_t ofs;
    size_t blockSize;
    size_t nodeNElems;
    size_t idx;
};

//! @} core_xml

/////////////////// XML & YAML I/O implementation //////////////////

CV_EXPORTS void write( FileStorage& fs, const String& name, int value );
CV_EXPORTS void write( FileStorage& fs, const String& name, int64_t value );
CV_EXPORTS void write( FileStorage& fs, const String& name, float value );
CV_EXPORTS void write( FileStorage& fs, const String& name, double value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const String& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const Mat& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const SparseMat& value );
#ifdef CV__LEGACY_PERSISTENCE
CV_EXPORTS void write( FileStorage& fs, const String& name, const std::vector<KeyPoint>& value);
CV_EXPORTS void write( FileStorage& fs, const String& name, const std::vector<DMatch>& value);
#endif

CV_EXPORTS void writeScalar( FileStorage& fs, int value );
CV_EXPORTS void writeScalar( FileStorage& fs, int64_t value );
CV_EXPORTS void writeScalar( FileStorage& fs, float value );
CV_EXPORTS void writeScalar( FileStorage& fs, double value );
CV_EXPORTS void writeScalar( FileStorage& fs, const String& value );

CV_EXPORTS void read(const FileNode& node, int& value, int default_value);
CV_EXPORTS void read(const FileNode& node, int64_t& value, int64_t default_value);
CV_EXPORTS void read(const FileNode& node, float& value, float default_value);
CV_EXPORTS void read(const FileNode& node, double& value, double default_value);
CV_EXPORTS void read(const FileNode& node, std::string& value, const std::string& default_value);
CV_EXPORTS void read(const FileNode& node, Mat& mat, const Mat& default_mat = Mat() );
CV_EXPORTS void read(const FileNode& node, SparseMat& mat, const SparseMat& default_mat = SparseMat() );
#ifdef CV__LEGACY_PERSISTENCE
CV_EXPORTS void read(const FileNode& node, std::vector<KeyPoint>& keypoints);
CV_EXPORTS void read(const FileNode& node, std::vector<DMatch>& matches);
#endif
CV_EXPORTS void read(const FileNode& node, KeyPoint& value, const KeyPoint& default_value);
CV_EXPORTS void read(const FileNode& node, DMatch& value, const DMatch& default_value);

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

template<typename _Tp, int m, int n> static inline void read(const FileNode& node, Matx<_Tp, m, n>& value, const Matx<_Tp, m, n>& default_matx = Matx<_Tp, m, n>())
{
    Mat temp;
    read(node, temp); // read as a Mat class

    if (temp.empty())
        value = default_matx;
    else
        value = Matx<_Tp, m, n>(temp);
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

/** @brief Writes string to a file storage.
 */
CV_EXPORTS FileStorage& operator << (FileStorage& fs, const String& str);

//! @cond IGNORED

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
            int _fmt = traits::SafeFmt<_Tp>::fmt;
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
            count = std::min(count, it->remaining());
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
            size_t remaining = it->remaining();
            size_t cn = DataType<_Tp>::channels;
            int _fmt = traits::SafeFmt<_Tp>::fmt;
            CV_Assert((_fmt >> 8) < 9);
            char fmt[] = { (char)((_fmt >> 8)+'1'), (char)_fmt, '\0' };
            CV_Assert((remaining % cn) == 0);
            size_t remaining1 = remaining / cn;
            count = count > remaining1 ? remaining1 : count;
            vec.resize(count);
            it->readRaw(fmt, !vec.empty() ? (uchar*)&vec[0] : 0, count*sizeof(_Tp));
        }
    private:
        FileNodeIterator* it;
    };

} // internal

//! @endcond

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

template<typename _Tp, int m, int n> static inline
void write(FileStorage& fs, const Matx<_Tp, m, n>& x )
{
    write(fs, Mat(x)); // write as a Mat class
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
    cv::internal::VecWriterProxy<_Tp, traits::SafeFmt<_Tp>::fmt != 0> w(&fs);
    w(vec);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Point_<_Tp>& pt )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, pt);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Point3_<_Tp>& pt )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, pt);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Size_<_Tp>& sz )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, sz);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Complex<_Tp>& c )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, c);
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Rect_<_Tp>& r )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r);
}

template<typename _Tp, int cn> static inline
void write(FileStorage& fs, const String& name, const Vec<_Tp, cn>& v )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, v);
}

template<typename _Tp, int m, int n> static inline
void write(FileStorage& fs, const String& name, const Matx<_Tp, m, n>& x )
{
    write(fs, name, Mat(x)); // write as a Mat class
}

template<typename _Tp> static inline
void write(FileStorage& fs, const String& name, const Scalar_<_Tp>& s )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, s);
}

static inline
void write(FileStorage& fs, const String& name, const Range& r )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r);
}

static inline
void write(FileStorage& fs, const String& name, const KeyPoint& kpt)
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, kpt.pt.x);
    write(fs, kpt.pt.y);
    write(fs, kpt.size);
    write(fs, kpt.angle);
    write(fs, kpt.response);
    write(fs, kpt.octave);
    write(fs, kpt.class_id);
}

static inline
void write(FileStorage& fs, const String& name, const DMatch& m)
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, m.queryIdx);
    write(fs, m.trainIdx);
    write(fs, m.imgIdx);
    write(fs, m.distance);
}

template<typename _Tp, typename std::enable_if< std::is_enum<_Tp>::value >::type* = nullptr>
static inline void write( FileStorage& fs, const String& name, const _Tp& val )
{
    write(fs, name, static_cast<int>(val));
}

template<typename _Tp> static inline
void write( FileStorage& fs, const String& name, const std::vector<_Tp>& vec )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ+(traits::SafeFmt<_Tp>::fmt != 0 ? FileNode::FLOW : 0));
    write(fs, vec);
}

template<typename _Tp> static inline
void write( FileStorage& fs, const String& name, const std::vector< std::vector<_Tp> >& vec )
{
    cv::internal::WriteStructContext ws(fs, name, FileNode::SEQ);
    for(size_t i = 0; i < vec.size(); i++)
    {
        cv::internal::WriteStructContext ws_(fs, name, FileNode::SEQ+(traits::SafeFmt<_Tp>::fmt != 0 ? FileNode::FLOW : 0));
        write(fs, vec[i]);
    }
}

#ifdef CV__LEGACY_PERSISTENCE
// This code is not needed anymore, but it is preserved here to keep source compatibility
// Implementation is similar to templates instantiations
static inline void write(FileStorage& fs, const KeyPoint& kpt) { write(fs, String(), kpt); }
static inline void write(FileStorage& fs, const DMatch& m) { write(fs, String(), m); }
static inline void write(FileStorage& fs, const std::vector<KeyPoint>& vec)
{
    cv::internal::VecWriterProxy<KeyPoint, 0> w(&fs);
    w(vec);
}
static inline void write(FileStorage& fs, const std::vector<DMatch>& vec)
{
    cv::internal::VecWriterProxy<DMatch, 0> w(&fs);
    w(vec);

}
#endif


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
    cv::internal::VecReaderProxy<_Tp, traits::SafeFmt<_Tp>::fmt != 0> r(&it);
    r(vec, maxCount);
}

template<typename _Tp, typename std::enable_if< std::is_enum<_Tp>::value >::type* = nullptr>
static inline void read(const FileNode& node, _Tp& value, const _Tp& default_value = static_cast<_Tp>(0))
{
    int temp;
    read(node, temp, static_cast<int>(default_value));
    value = static_cast<_Tp>(temp);
}

template<typename _Tp> static inline
void read( const FileNode& node, std::vector<_Tp>& vec, const std::vector<_Tp>& default_value = std::vector<_Tp>() )
{
    if(node.empty())
        vec = default_value;
    else
    {
        FileNodeIterator it = node.begin();
        read( it, vec );
    }
}

static inline
void read( const FileNode& node, std::vector<KeyPoint>& vec, const std::vector<KeyPoint>& default_value )
{
    if(node.empty())
        vec = default_value;
    else
        read(node, vec);
}

static inline
void read( const FileNode& node, std::vector<DMatch>& vec, const std::vector<DMatch>& default_value )
{
    if(node.empty())
        vec = default_value;
    else
        read(node, vec);
}

/** @brief Writes data to a file storage.
 */
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

/** @brief Writes data to a file storage.
 */
static inline
FileStorage& operator << (FileStorage& fs, const char* str)
{
    return (fs << String(str));
}

/** @brief Writes data to a file storage.
 */
static inline
FileStorage& operator << (FileStorage& fs, char* value)
{
    return (fs << String(value));
}

/** @brief Reads data from a file storage.
 */
template<typename _Tp> static inline
FileNodeIterator& operator >> (FileNodeIterator& it, _Tp& value)
{
    read( *it, value, _Tp());
    return ++it;
}

/** @brief Reads data from a file storage.
 */
template<typename _Tp> static inline
FileNodeIterator& operator >> (FileNodeIterator& it, std::vector<_Tp>& vec)
{
    cv::internal::VecReaderProxy<_Tp, traits::SafeFmt<_Tp>::fmt != 0> r(&it);
    r(vec, (size_t)INT_MAX);
    return it;
}

/** @brief Reads data from a file storage.
 */
template<typename _Tp> static inline
void operator >> (const FileNode& n, _Tp& value)
{
    read( n, value, _Tp());
}

/** @brief Reads data from a file storage.
 */
template<typename _Tp> static inline
void operator >> (const FileNode& n, std::vector<_Tp>& vec)
{
    FileNodeIterator it = n.begin();
    it >> vec;
}

/** @brief Reads KeyPoint from a file storage.
*/
//It needs special handling because it contains two types of fields, int & float.
static inline
void operator >> (const FileNode& n, KeyPoint& kpt)
{
    FileNodeIterator it = n.begin();
    it >> kpt.pt.x >> kpt.pt.y >> kpt.size >> kpt.angle >> kpt.response >> kpt.octave >> kpt.class_id;
}

#ifdef CV__LEGACY_PERSISTENCE
static inline
void operator >> (const FileNode& n, std::vector<KeyPoint>& vec)
{
    read(n, vec);
}
static inline
void operator >> (const FileNode& n, std::vector<DMatch>& vec)
{
    read(n, vec);
}
#endif

/** @brief Reads DMatch from a file storage.
*/
//It needs special handling because it contains two types of fields, int & float.
static inline
void operator >> (const FileNode& n, DMatch& m)
{
    FileNodeIterator it = n.begin();
    it >> m.queryIdx >> m.trainIdx >> m.imgIdx >> m.distance;
}

CV_EXPORTS bool operator == (const FileNodeIterator& it1, const FileNodeIterator& it2);
CV_EXPORTS bool operator != (const FileNodeIterator& it1, const FileNodeIterator& it2);

static inline
ptrdiff_t operator - (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it2.remaining() - it1.remaining();
}

static inline
bool operator < (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it1.remaining() > it2.remaining();
}

} // cv

#endif // OPENCV_CORE_PERSISTENCE_HPP
