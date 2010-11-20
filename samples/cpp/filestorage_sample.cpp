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

void help(char** av)
{
  cout << "usage:\n" << av[0] << " outputfile.yml.gz\n"
      << "Try using different extensions.(e.g. yaml yml xml xml.gz etc...)\n"
      << "This will serialize some matrices and image names to the format specified." << endl;
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
  void write(FileStorage& fs) const
  {
    fs << "{" << "A" << A << "X" << X << "id" << id << "}";
  }
  void read(const FileNode& node)
  {

    A = (int)node["A"];
    X = (double)node["X"];
    id = (string)node["id"];
  }
};

void write(FileStorage& fs, const std::string& name, const MyData& x){
  x.write(fs);
}
void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
  if(node.empty())
    x = default_value;
  else
    x = (MyData)node;
}

ostream& operator<<(ostream& out, const MyData& m){
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

    cout << "attempting to read mdata_b\n";
    fs["mdata_b"] >> m;
    cout << "read mdata_b\n";
    cout << m << endl;

  }

  cout << "Try opening " << filename << " to see the serialized data." << endl;

  return 0;
}
