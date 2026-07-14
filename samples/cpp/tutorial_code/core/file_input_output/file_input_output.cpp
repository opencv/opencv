#include <opencv2/core.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

static void help(char** av)
{
    cout << endl
        << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl << endl
        << "usage: "                                                                      << endl
        <<  av[0] << " [output file name] (default outputfile.yml.gz)"                    << endl << endl
        << "The output file may be XML (xml), YAML (yml/yaml), or JSON (json)." << endl
        << "You can even compress it by specifying this in its extension like xml.gz yaml.gz etc... " << endl
        << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
        << "For example: - create a class and have it serialized"                         << endl
        << "             - use it to read and write matrices."                            << endl << endl;
}

class MyData
{
public:
    MyData() : A(0), X(0), id()
    {}
    explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
    {}
    //! [inside]
    void write(FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{" << "A" << A << "X" << X << "id" << id << "}";
    }
    void read(const FileNode& node)                          //Read serialization for this class
    {
        A = (int)node["A"];
        X = (double)node["X"];
        id = (string)node["id"];
    }
    //! [inside]
public:   // Data Members
    int A;
    double X;
    string id;
};

//These write and read functions must be defined for the serialization in FileStorage to work
//! [outside]
static void write(FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}
//! [outside]

// This function will print our custom class to the console
static ostream& operator<<(ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}

int main(int ac, char** av)
{
    string filename;

    if (ac != 2)
    {
        help(av);
        filename = "outputfile.yml.gz";
    }
    else
        filename = av[1];

    { //write
        //! [iomati]
        Mat R = Mat_<uchar>::eye(3, 3),
            T = Mat_<double>::zeros(3, 1);
        //! [iomati]
        //! [customIOi]
        MyData m(1);
        //! [customIOi]

        //! [open]
        FileStorage fs(filename, FileStorage::WRITE);
        // or:
        // FileStorage fs;
        // fs.open(filename, FileStorage::WRITE);
        //! [open]

        //! [writeNum]
        fs << "iterationNr" << 100;
        //! [writeNum]
        //! [writeStr]
        fs << "strings" << "[";                              // text - string sequence
        fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
        fs << "]";                                           // close sequence
        //! [writeStr]

        //! [writeMap]
        fs << "Mapping";                              // text - mapping
        fs << "{" << "One" << 1;
        fs <<        "Two" << 2 << "}";
        //! [writeMap]

        //! [iomatw]
        fs << "R" << R;                                      // cv::Mat
        fs << "T" << T;
        //! [iomatw]

        //! [customIOw]
        fs << "MyData" << m;                                // your own data structures
        //! [customIOw]

        //! [close]
        fs.release();                                       // explicit close
        //! [close]
        cout << "Write operation to file:" << filename << " completed successfully." << endl;
    }

    {//read
        cout << endl << "Reading: " << endl;
        FileStorage fs;
        fs.open(filename, FileStorage::READ);

        //! [readNum]
        int itNr;
        //fs["iterationNr"] >> itNr;
        itNr = (int) fs["iterationNr"];
        //! [readNum]
        cout << itNr;
        if (!fs.isOpened())
        {
            cerr << "Failed to open " << filename << endl;
            help(av);
            return 1;
        }

        //! [readStr]
        FileNode n = fs["strings"];                         // Read string sequence - Get node
        if (n.type() != FileNode::SEQ)
        {
            cerr << "strings is not a sequence! FAIL" << endl;
            return 1;
        }

        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it)
            cout << (string)*it << endl;
        //! [readStr]


        //! [readMap]
        n = fs["Mapping"];                                // Read mappings from a sequence
        cout << "Two  " << (int)(n["Two"]) << "; ";
        cout << "One  " << (int)(n["One"]) << endl << endl;
        //! [readMap]


        MyData m;
        Mat R, T;

        //! [iomat]
        fs["R"] >> R;                                      // Read cv::Mat
        fs["T"] >> T;
        //! [iomat]
        //! [customIO]
        fs["MyData"] >> m;                                 // Read your own structure_
        //! [customIO]

        cout << endl
            << "R = " << R << endl;
        cout << "T = " << T << endl << endl;
        cout << "MyData = " << endl << m << endl << endl;

        //Show default behavior for non existing nodes
        //! [nonexist]
        cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
        fs["NonExisting"] >> m;
        cout << endl << "NonExisting = " << endl << m << endl;
        //! [nonexist]
    }

    cout << endl
        << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;

    return 0;
}
