#include "opencv2/core/core.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

void help(char** av)
{
  cout << endl 
       << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl
       << "usage: "                                                                      << endl
       <<  av[0] << " outputfile.yml.gz"                                                 << endl
       << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
       << "specifying this in its extension like xml.gz yaml.gz etc... "                  << endl
      << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
      << "For example: - create a class and have it serialized"                         << endl
      << "             - use it to read and write matrices."                            << endl;
}

class MyData
{
public:
  MyData() : A(0), X(0), id()
  {}
  explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
  {}

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
public:   // Data Members
    int A;
    double X;
    string id;
};

//These write and read functions must be defined for the serialization in FileStorage to work
void write(FileStorage& fs, const std::string&, const MyData& x)
{
  x.write(fs);
}
void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
  if(node.empty())
    x = default_value;
  else
    x.read(node);
}

// This function will print our custom class to the console
ostream& operator<<(ostream& out, const MyData& m) 
{ 
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}

int main(int ac, char** av)
{
  if (ac != 2)
  {
    help(av);
    return 1;
  }

  string filename = av[1];

  //write
  {
    Mat R = Mat_<double>::eye(3, 3),
        T = Mat_<double>::zeros(3, 1);
    
    MyData m(1);

    FileStorage fs(filename, FileStorage::WRITE);

    
    fs << "strings" << "[";
    fs << "image1.jpg" << "Awesomeness" << "baboon.jpg";
    fs << "]";

    fs << "R" << R;
    fs << "T" << T;

    fs << "MyData" << m;

    cout << "Write Done." << endl;
  }
  
  //read
  {
    cout << endl << "Reading: " << endl;
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened())
    {
      cerr << "Failed to open " << filename << endl;
      help(av);
      return 1;
    }

    FileNode n = fs["strings"];
    if (n.type() != FileNode::SEQ)
    {
      cerr << "strings is not a sequence! FAIL" << endl;
      return 1;
    }

    
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
      cout << (string)*it << endl;

    MyData m;
    Mat R, T;

    fs["R"] >> R;
    fs["T"] >> T;
    fs["MyData"] >> m;

    cout << endl 
        << "R = " << R << "\n";
    cout << "T = " << T << endl << endl;
    cout << "MyData = " << endl << m << endl << endl;

     //Show default behavior for non existing nodes
    cout << "Attempt to read NonExisting (should initialize the data structure with its default).";  
    fs["NonExisting"] >> m;
    cout << endl << "NonExisting = " << endl << m << endl;
  }

  cout << endl 
       << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;

  return 0;
}
