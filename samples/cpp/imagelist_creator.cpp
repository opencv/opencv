/*this creates a yaml or xml list of files from the command line args
 */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <iostream>

using std::string;
using std::cout;
using std::endl;

using namespace cv;

static void help(char** av)
{
  cout << "\nThis creates a yaml or xml list of files from the command line args\n"
      "usage:\n./" << av[0] << " imagelist.yaml *.png\n"
      << "Try using different extensions.(e.g. yaml yml xml xml.gz etc...)\n"
      << "This will serialize this list of images or whatever with opencv's FileStorage framework" << endl;
}

int main(int ac, char** av)
{
  if (ac < 3)
  {
    help(av);
    return 1;
  }

  string outputname = av[1];

  Mat m = imread(outputname); //check if the output is an image - prevent overwrites!
  if(!m.empty()){
    std::cerr << "fail! Please specify an output file, don't want to overwrite you images!" << endl;
    help(av);
    return 1;
  }

  FileStorage fs(outputname, FileStorage::WRITE);
  fs << "images" << "[";
  for(int i = 2; i < ac; i++){
    fs << string(av[i]);
  }
  fs << "]";
  return 0;
}
