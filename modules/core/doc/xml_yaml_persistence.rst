XML/YAML Persistence
====================

.. highlight:: cpp

XML/YAML file storages. Writing to a file storage.
--------------------------------------------------

You can store and then restore various OpenCV data structures to/from XML (http://www.w3c.org/XML) or YAML
(http://www.yaml.org) formats. Also, it is possible store and load arbitrarily complex data structures, which include OpenCV data structures, as well as primitive data types (integer and floating-point numbers and text strings) as their elements.

Use the following procedure to write something to XML or YAML:
 #. Create new :ocv:class:`FileStorage` and open it for writing. It can be done with a single call to :ocv:func:`FileStorage::FileStorage` constructor that takes a filename, or you can use the default constructor and then call :ocv:func:`FileStorage::open`. Format of the file (XML or YAML) is determined from the filename extension (".xml" and ".yml"/".yaml", respectively)
 #. Write all the data you want using the streaming operator ``>>``, just like in the case of STL streams.
 #. Close the file using :ocv:func:`FileStorage::release`. ``FileStorage`` destructor also closes the file.

Here is an example: ::

    #include "opencv2/opencv.hpp"
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

The sample above stores to XML and integer, text string (calibration date), 2 matrices, and a custom structure "feature", which includes feature coordinates and LBP (local binary pattern) value. Here is output of the sample:

.. code-block:: yaml

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

As an exercise, you can replace ".yml" with ".xml" in the sample above and see, how the corresponding XML file will look like.

Several things can be noted by looking at the sample code and the output:
 *
   The produced YAML (and XML) consists of heterogeneous collections that can be nested. There are 2 types of collections: named collections (mappings) and unnamed collections (sequences). In mappings each element has a name and is accessed by name. This is similar to structures and ``std::map`` in C/C++ and dictionaries in Python. In sequences elements do not have names, they are accessed by indices. This is similar to arrays and ``std::vector`` in C/C++ and lists, tuples in Python. "Heterogeneous" means that elements of each single collection can have different types.

   Top-level collection in YAML/XML is a mapping. Each matrix is stored as a mapping, and the matrix elements are stored as a sequence. Then, there is a sequence of features, where each feature is represented a mapping, and lbp value in a nested sequence.

 *
   When you write to a mapping (a structure), you write element name followed by its value. When you write to a sequence, you simply write the elements one by one. OpenCV data structures (such as cv::Mat) are written in absolutely the same way as simple C data structures - using **``<<``** operator.

 *
   To write a mapping, you first write the special string **"{"** to the storage, then write the elements as pairs (``fs << <element_name> << <element_value>``) and then write the closing **"}"**.

 *
   To write a sequence, you first write the special string **"["**, then write the elements, then write the closing **"]"**.

 *
   In YAML (but not XML), mappings and sequences can be written in a compact Python-like inline form. In the sample above matrix elements, as well as each feature, including its lbp value, is stored in such inline form. To store a mapping/sequence in a compact form, put ":" after the opening character, e.g. use **"{:"** instead of **"{"** and **"[:"** instead of **"["**. When the data is written to XML, those extra ":" are ignored.


Reading data from a file storage.
---------------------------------

To read the previously written XML or YAML file, do the following:

 #.
   Open the file storage using :ocv:func:`FileStorage::FileStorage` constructor or :ocv:func:`FileStorage::open` method. In the current implementation the whole file is parsed and the whole representation of file storage is built in memory as a hierarchy of file nodes (see :ocv:class:`FileNode`)

 #.
   Read the data you are interested in. Use :ocv:func:`FileStorage::operator []`, :ocv:func:`FileNode::operator []` and/or :ocv:class:`FileNodeIterator`.

 #.
   Close the storage using :ocv:func:`FileStorage::release`.

Here is how to read the file created by the code sample above: ::

    FileStorage fs2("test.yml", FileStorage::READ);

    // first method: use (type) operator on FileNode.
    int frameCount = (int)fs2["frameCount"];

    std::string date;
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
    fs.release();

FileStorage
-----------
.. ocv:class:: FileStorage

XML/YAML file storage class that encapsulates all the information necessary for writing or reading data to/from a file.

FileStorage::FileStorage
------------------------
The constructors.

.. ocv:function:: FileStorage::FileStorage()

.. ocv:function:: FileStorage::FileStorage(const string& source, int flags, const string& encoding=string())

    :param source: Name of the file to open or the text string to read the data from. Extension of the file (``.xml`` or ``.yml``/``.yaml``) determines its format (XML or YAML respectively). Also you can append ``.gz`` to work with compressed files, for example ``myHugeMatrix.xml.gz``. If both ``FileStorage::WRITE`` and ``FileStorage::MEMORY`` flags are specified, ``source`` is used just to specify the output file format (e.g. ``mydata.xml``, ``.yml`` etc.).

    :param flags: Mode of operation. Possible values are:

        * **FileStorage::READ** Open the file for reading.

        * **FileStorage::WRITE** Open the file for writing.

        * **FileStorage::APPEND** Open the file for appending.
        
        * **FileStorage::MEMORY** Read data from ``source`` or write data to the internal buffer (which is returned by ``FileStorage::release``)

    :param encoding: Encoding of the file. Note that UTF-16 XML encoding is not supported currently and you should use 8-bit encoding instead of it.

The full constructor opens the file. Alternatively you can use the default constructor and then call :ocv:func:`FileStorage::open`.


FileStorage::open
-----------------
Opens a file.

.. ocv:function:: bool FileStorage::open(const string& filename, int flags, const string& encoding=string())

See description of parameters in :ocv:func:`FileStorage::FileStorage`. The method calls :ocv:func:`FileStorage::release` before opening the file.


FileStorage::isOpened
---------------------
Checks whether the file is opened.

.. ocv:function:: bool FileStorage::isOpened() const

    :returns: ``true`` if the object is associated with the current file and ``false`` otherwise.

It is a good practice to call this method after you tried to open a file.


FileStorage::release
--------------------
Closes the file and releases all the memory buffers.

.. ocv:function:: string FileStorage::release()

Call this method after all I/O operations with the storage are finished. If the storage was opened for writing data and ``FileStorage::WRITE`` was specified


FileStorage::getFirstTopLevelNode
---------------------------------
Returns the first element of the top-level mapping.

.. ocv:function:: FileNode FileStorage::getFirstTopLevelNode() const

    :returns: The first element of the top-level mapping.


FileStorage::root
-----------------
Returns the top-level mapping

.. ocv:function:: FileNode FileStorage::root(int streamidx=0) const

    :param streamidx: Zero-based index of the stream. In most cases there is only one stream in the file. However, YAML supports multiple streams and so there can be several.

    :returns: The top-level mapping.


FileStorage::operator[]
-----------------------
Returns the specified element of the top-level mapping.

.. ocv:function:: FileNode FileStorage::operator[](const string& nodename) const

.. ocv:function:: FileNode FileStorage::operator[](const char* nodename) const

    :param nodename: Name of the file node.

    :returns: Node with the given name.


FileStorage::operator*
----------------------
Returns the obsolete C FileStorage structure.

.. ocv:function:: CvFileStorage* FileStorage::operator *()

.. ocv:function:: const CvFileStorage* FileStorage::operator *() const

    :returns: Pointer to the underlying C FileStorage structure


FileStorage::writeRaw
---------------------
Writes multiple numbers.

.. ocv:function:: void FileStorage::writeRaw( const string& fmt, const uchar* vec, size_t len )

     :param fmt: Specification of each array element that has the following format  ``([count]{'u'|'c'|'w'|'s'|'i'|'f'|'d'})...`` where the characters correspond to fundamental C++ types:

            * **u** 8-bit unsigned number

            * **c** 8-bit signed number

            * **w** 16-bit unsigned number

            * **s** 16-bit signed number

            * **i** 32-bit signed number

            * **f** single precision floating-point number

            * **d** double precision floating-point number

            * **r** pointer, 32 lower bits of which are written as a signed integer. The type can be used to store structures with links between the elements.

            ``count``  is the optional counter of values of a given type. For example,  ``2if``  means that each array element is a structure of 2 integers, followed by a single-precision floating-point number. The equivalent notations of the above specification are ' ``iif`` ', ' ``2i1f`` ' and so forth. Other examples:  ``u``  means that the array consists of bytes, and  ``2d``  means the array consists of pairs  of doubles.

     :param vec: Pointer to the written array.

     :param len: Number of the ``uchar`` elements to write.

Writes one or more numbers of the specified format to the currently written structure. Usually it is more convenient to use :ocv:func:`operator <<` instead of this method.

FileStorage::writeObj
---------------------
Writes the registered C structure (CvMat, CvMatND, CvSeq).

.. ocv:function:: void FileStorage::writeObj( const string& name, const void* obj )

    :param name: Name of the written object.

    :param obj: Pointer to the object.

See :ocv:cfunc:`Write` for details.


FileStorage::getDefaultObjectName
---------------------------------
Returns the normalized object name for the specified name of a file.

.. ocv:function:: static string FileStorage::getDefaultObjectName(const string& filename)

   :param filename: Name of a file

   :returns: The normalized object name.


operator <<
-----------
Writes data to a file storage.

.. ocv:function:: template<typename _Tp> FileStorage& operator << (FileStorage& fs, const _Tp& value)

.. ocv:function:: template<typename _Tp> FileStorage& operator << ( FileStorage& fs, const vector<_Tp>& vec )

    :param fs: Opened file storage to write data.

    :param value: Value to be written to the file storage.

    :param vec: Vector of values to be written to the file storage.

It is the main function to write data to a file storage. See an example of its usage at the beginning of the section.


operator >>
-----------
Reads data from a file storage.

.. ocv:function:: template<typename _Tp> void operator >> (const FileNode& n, _Tp& value)

.. ocv:function:: template<typename _Tp> void operator >> (const FileNode& n, vector<_Tp>& vec)

.. ocv:function:: template<typename _Tp> FileNodeIterator& operator >> (FileNodeIterator& it, _Tp& value)

.. ocv:function:: template<typename _Tp> FileNodeIterator& operator >> (FileNodeIterator& it, vector<_Tp>& vec)

    :param n: Node from which data will be read.

    :param it: Iterator from which data will be read.

    :param value: Value to be read from the file storage.

    :param vec: Vector of values to be read from the file storage.

It is the main function to read data from a file storage. See an example of its usage at the beginning of the section.


FileNode
--------
.. ocv:class:: FileNode

File Storage Node class. The node is used to store each and every element of the file storage opened for reading. When XML/YAML file is read, it is first parsed and stored in the memory as a hierarchical collection of nodes. Each node can be a “leaf” that is contain a single number or a string, or be a collection of other nodes. There can be named collections (mappings) where each element has a name and it is accessed by a name, and ordered collections (sequences) where elements do not have names but rather accessed by index. Type of the file node can be determined using :ocv:func:`FileNode::type` method.

Note that file nodes are only used for navigating file storages opened for reading. When a file storage is opened for writing, no data is stored in memory after it is written.


FileNode::FileNode
------------------
The constructors.

.. ocv:function:: FileNode::FileNode()

.. ocv:function:: FileNode::FileNode(const CvFileStorage* fs, const CvFileNode* node)

.. ocv:function:: FileNode::FileNode(const FileNode& node)

    :param fs: Pointer to the obsolete file storage structure.

    :param node: File node to be used as initialization for the created file node.

These constructors are used to create a default file node, construct it from obsolete structures or from the another file node.


FileNode::operator[]
--------------------
Returns element of a mapping node or a sequence node.

.. ocv:function:: FileNode FileNode::operator[](const string& nodename) const

.. ocv:function:: FileNode FileNode::operator[](const char* nodename) const

.. ocv:function:: FileNode FileNode::operator[](int i) const

    :param nodename: Name of an element in the mapping node.

    :param i: Index of an element in the sequence node.

    :returns: Returns the element with the given identifier.


FileNode::type
--------------
Returns type of the node.

.. ocv:function:: int FileNode::type() const

    :returns: Type of the node. Possible values are:

        * **FileNode::NONE** Empty node.

        * **FileNode::INT** Integer.

        * **FileNode::REAL** Floating-point number.

        * **FileNode::FLOAT** Synonym or ``REAL``.

        * **FileNode::STR** Text string in UTF-8 encoding.

        * **FileNode::STRING** Synonym for ``STR``.

        * **FileNode::REF** Integer of type ``size_t``. Typically used for storing complex dynamic structures where some elements reference the others.

        * **FileNode::SEQ** Sequence.

        * **FileNode::MAP** Mapping.

        * **FileNode::FLOW** Compact representation of a sequence or mapping. Used only by the YAML writer.

        * **FileNode::USER** Registered object (e.g. a matrix).

        * **FileNode::EMPTY** Empty structure (sequence or mapping).

        * **FileNode::NAMED** The node has a name (i.e. it is an element of a mapping).


FileNode::empty
---------------
Checks whether the node is empty.

.. ocv:function:: bool FileNode::empty() const

    :returns: ``true`` if the node is empty.


FileNode::isNone
----------------
Checks whether the node is a "none" object

.. ocv:function:: bool FileNode::isNone() const

    :returns: ``true`` if the node is a "none" object.


FileNode::isSeq
---------------
Checks whether the node is a sequence.

.. ocv:function:: bool FileNode::isSeq() const

    :returns: ``true`` if the node is a sequence.


FileNode::isMap
---------------
Checks whether the node is a mapping.

.. ocv:function:: bool FileNode::isMap() const

    :returns: ``true`` if the node is a mapping.


FileNode::isInt
---------------
Checks whether the node is an integer.

.. ocv:function:: bool FileNode::isInt() const

    :returns: ``true`` if the node is an integer.


FileNode::isReal
----------------
Checks whether the node is a floating-point number.

.. ocv:function:: bool FileNode::isReal() const

    :returns: ``true`` if the node is a floating-point number.


FileNode::isString
------------------
Checks whether the node is a text string.

.. ocv:function:: bool FileNode::isString() const

    :returns: ``true`` if the node is a text string.


FileNode::isNamed
-----------------
Checks whether the node has a name.

.. ocv:function:: bool FileNode::isNamed() const

    :returns: ``true`` if the node has a name.


FileNode::name
--------------
Returns the node name.

.. ocv:function:: string FileNode::name() const

    :returns: The node name or an empty string if the node is nameless.


FileNode::size
--------------
Returns the number of elements in the node.

.. ocv:function:: size_t FileNode::size() const

    :returns: The number of elements in the node, if it is a sequence or mapping, or 1 otherwise.


FileNode::operator int
----------------------
Returns the node content as an integer.

.. ocv:function:: FileNode::operator int() const

    :returns: The node content as an integer. If the node stores a floating-point number, it is rounded.


FileNode::operator float
------------------------
Returns the node content as float.

.. ocv:function:: FileNode::operator float() const

    :returns: The node content as float.


FileNode::operator double
-------------------------
Returns the node content as double.

.. ocv:function:: FileNode::operator double() const

    :returns: The node content as double.


FileNode::operator string
-------------------------
Returns the node content as text string.

.. ocv:function:: FileNode::operator string() const

    :returns: The node content as a text string.


FileNode::operator*
-------------------
Returns pointer to the underlying obsolete file node structure.

.. ocv:function:: CvFileNode* FileNode::operator *()

    :returns: Pointer to the underlying obsolete file node structure.


FileNode::begin
---------------
Returns the iterator pointing to the first node element.

.. ocv:function:: FileNodeIterator FileNode::begin() const

   :returns: Iterator pointing to the first node element.


FileNode::end
-------------
Returns the iterator pointing to the element following the last node element.

.. ocv:function:: FileNodeIterator FileNode::end() const

    :returns: Iterator pointing to the element following the last node element.


FileNode::readRaw
-----------------
Reads node elements to the buffer with the specified format.

.. ocv:function:: void FileNode::readRaw( const string& fmt, uchar* vec, size_t len ) const

    :param fmt: Specification of each array element. It has the same format as in :ocv:func:`FileStorage::writeRaw`.

    :param vec: Pointer to the destination array.

    :param len: Number of elements to read. If it is greater than number of remaining elements then all of them will be read.

Usually it is more convenient to use :ocv:func:`operator >>` instead of this method.

FileNode::readObj
-----------------
Reads the registered object.

.. ocv:function:: void* FileNode::readObj() const

    :returns: Pointer to the read object.

See :ocv:cfunc:`Read` for details.

FileNodeIterator
----------------
.. ocv:class:: FileNodeIterator

The class ``FileNodeIterator`` is used to iterate through sequences and mappings. A standard STL notation, with ``node.begin()``, ``node.end()`` denoting the beginning and the end of a sequence, stored in ``node``.  See the data reading sample in the beginning of the section.


FileNodeIterator::FileNodeIterator
----------------------------------
The constructors.

.. ocv:function:: FileNodeIterator::FileNodeIterator()

.. ocv:function:: FileNodeIterator::FileNodeIterator(const CvFileStorage* fs, const CvFileNode* node, size_t ofs=0)

.. ocv:function:: FileNodeIterator::FileNodeIterator(const FileNodeIterator& it)

    :param fs: File storage for the iterator.

    :param node: File node for the iterator.

    :param ofs: Index of the element in the node. The created iterator will point to this element.

    :param it: Iterator to be used as initialization for the created iterator.

These constructors are used to create a default iterator, set it to specific element in a file node or construct it from another iterator.


FileNodeIterator::operator*
---------------------------
Returns the currently observed element.

.. ocv:function:: FileNode FileNodeIterator::operator *() const

    :returns: Currently observed element.


FileNodeIterator::operator->
----------------------------
Accesses methods of the currently observed element.

.. ocv:function:: FileNode FileNodeIterator::operator ->() const


FileNodeIterator::operator ++
-----------------------------
Moves iterator to the next node.

.. ocv:function:: FileNodeIterator& FileNodeIterator::operator ++ ()

.. ocv:function:: FileNodeIterator FileNodeIterator::operator ++ (int)


FileNodeIterator::operator --
-----------------------------
Moves iterator to the previous node.

.. ocv:function:: FileNodeIterator& FileNodeIterator::operator -- ()

.. ocv:function:: FileNodeIterator FileNodeIterator::operator -- (int)


FileNodeIterator::operator +=
-----------------------------
Moves iterator forward by the specified offset.

.. ocv:function:: FileNodeIterator& FileNodeIterator::operator +=( int param )

    :param ofs: Offset (possibly negative) to move the iterator.


FileNodeIterator::operator -=
-----------------------------
Moves iterator backward by the specified offset (possibly negative).

.. ocv:function:: FileNodeIterator& FileNodeIterator::operator -=( int param )

    :param ofs: Offset (possibly negative) to move the iterator.


FileNodeIterator::readRaw
-------------------------
Reads node elements to the buffer with the specified format.

.. ocv:function:: FileNodeIterator& FileNodeIterator::readRaw( const string& fmt, uchar* vec, size_t maxCount=(size_t)INT_MAX )

    :param fmt: Specification of each array element. It has the same format as in :ocv:func:`FileStorage::writeRaw`.

    :param vec: Pointer to the destination array.

    :param maxCount: Number of elements to read. If it is greater than number of remaining elements then all of them will be read.

Usually it is more convenient to use :ocv:func:`operator >>` instead of this method.
