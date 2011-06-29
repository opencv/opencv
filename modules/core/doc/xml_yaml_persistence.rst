XML/YAML Persistence
====================

.. highlight:: cpp

XML/YAML file storages. Writing to a file storage.
--------------------------------------------------

You can store and then restore various OpenCV data structures to/from XML (http://www.w3c.org/XML) or YAML
(http://www.yaml.org) formats. Also, it is possible store and load arbitrarily complex data structures, which include OpenCV data structures, as well as primitive data types (integer and floating-point numbers and text strings) as their elements.

Use the following procedure to write something to XML or YAML:
 #. Create new :ocv:class:`FileStorage` and open it for writing. It can be done with a single call to :ocv:func:`FileStorage::FileStorage` constructor that takes a filename, or you can use the default constructor and then call :ocv:class:`FileStorage::open`. Format of the file (XML or YAML) is determined from the filename extension (".xml" and ".yml"/".yaml", respectively)
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

XML/YAML file storage class that incapsulates all the information necessary for writing or reading data to/from file.


FileNode
--------
.. ocv:class:: FileNode

The class ``FileNode`` represents each element of the file storage, be it a matrix, a matrix element or a top-level node, containing all the file content. That is, a file node may contain either a singe value (integer, floating-point value or a text string), or it can be a sequence of other file nodes, or it can be a mapping. Type of the file node can be determined using :ocv:func:`FileNode::type` method.

FileNodeIterator
----------------
.. ocv:class:: FileNodeIterator

The class ``FileNodeIterator`` is used to iterate through sequences and mappings. A standard STL notation, with ``node.begin()``, ``node.end()`` denoting the beginning and the end of a sequence, stored in ``node``.  See the data reading sample in the beginning of the section.
