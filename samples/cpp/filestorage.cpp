/*
 * filestorage_sample demonstrate the usage of the opencv serialization functionality
 */

#include "opencv2/core/core.hpp"
#include <iostream>
#include <string>

using std::string;
using std::cout;
using std::endl;
using std::cerr;
using std::ostream;
using namespace cv;

static void help(char** av)
{
  cout << "\nfilestorage_sample demonstrate the usage of the opencv serialization functionality.\n"
      << "usage:\n"
      <<  av[0] << " outputfile.yml.gz\n"
      << "\n   outputfile above can have many different extenstions, see below."
      << "\nThis program demonstrates the use of FileStorage for serialization, that is use << and >>  in OpenCV\n"
      << "For example, how to create a class and have it serialize, but also how to use it to read and write matrices.\n"
      << "FileStorage allows you to serialize to various formats specified by the file end type."
          << "\nYou should try using different file extensions.(e.g. yaml yml xml xml.gz yaml.gz etc...)\n" << endl;
}

struct MyData
{
  MyData() :
    A(0), X(0), id()
  {
  }
  explicit MyData(int) :
    A(97), X(CV_PI), id("mydata1234")
  {
  }
  int A;
  double X;
  string id;
  void write(FileStorage& fs) const //Write serialization for this class
  {
    fs << "{" << "A" << A << "X" << X << "id" << id << "}";
  }
  void read(const FileNode& node)  //Read serialization for this class
  {

    A = (int)node["A"];
    X = (double)node["X"];
    id = (string)node["id"];
  }
};

//These write and read functions must exist as per the inline functions in operations.hpp
static void write(FileStorage& fs, const std::string&, const MyData& x){
  x.write(fs);
}
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
  if(node.empty())
    x = default_value;
  else
    x.read(node);
}

static ostream& operator<<(ostream& out, const MyData& m){
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
    FileStorage fs(filename, FileStorage::WRITE);

    cout << "writing images\n";
    fs << "images" << "[";

    fs << "image1.jpg" << "myfi.png" << "baboon.jpg";
    cout << "image1.jpg" << " myfi.png" << " baboon.jpg" << endl;

    fs << "]";

    cout << "writing mats\n";
    Mat R =Mat_<double>::eye(3, 3),T = Mat_<double>::zeros(3, 1);
    cout << "R = " << R << "\n";
    cout << "T = " << T << "\n";
    fs << "R" << R;
    fs << "T" << T;

    cout << "writing MyData struct\n";
    MyData m(1);
    fs << "mdata" << m;
    cout << m << endl;
  }

  //read
  {
    FileStorage fs(filename, FileStorage::READ);

    if (!fs.isOpened())
    {
      cerr << "failed to open " << filename << endl;
      help(av);
      return 1;
    }

    FileNode n = fs["images"];
    if (n.type() != FileNode::SEQ)
    {
      cerr << "images is not a sequence! FAIL" << endl;
      return 1;
    }

    cout << "reading images\n";
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
    {
      cout << (string)*it << "\n";
    }

    Mat R, T;
    cout << "reading R and T" << endl;

    fs["R"] >> R;
    fs["T"] >> T;

    cout << "R = " << R << "\n";
    cout << "T = " << T << endl;

    MyData m;
    fs["mdata"] >> m;

    cout << "read mdata\n";
    cout << m << endl;

    cout << "attempting to read mdata_b\n";   //Show default behavior for empty matrix
    fs["mdata_b"] >> m;
    cout << "read mdata_b\n";
    cout << m << endl;

  }

  cout << "Try opening " << filename << " to see the serialized data." << endl;

  return 0;
}
