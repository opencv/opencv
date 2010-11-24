/*
 * cvstarter.cpp
 *
 *  Created on: Nov 23, 2010
 *      Author: Ethan Rublee
 *
 * A starter sample for using opencv, load up a list or get a video stream
 * easy as CV_PI right?
 */
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

//hide the local functions in an anon namespace
namespace
{
void help(char** av)
{
  cout << "usage:" << av[0] << " image_list.yaml\n"
      << "\tThis is a starter sample, to get you up and going in a copy pasta fashion\n"
      << "\tTry running imagelist_creator to generate a list of images\n"
      << "\tOr try running this with an integer arg like:\n\n" << av[0] << " 0\n"
      << "\tThis will open up the video device." << endl;
}

bool readStringList(const string& filename, vector<string>& l)
{
  l.resize(0);
  FileStorage fs(filename, FileStorage::READ);
  if (!fs.isOpened())
    return false;
  FileNode n = fs.getFirstTopLevelNode();
  if (n.type() != FileNode::SEQ)
    return false;
  FileNodeIterator it = n.begin(), it_end = n.end();
  for (; it != it_end; ++it)
    l.push_back((string)*it);
  return true;
}

int videoProcess(VideoCapture& capture)
{

  namedWindow("video",CV_WINDOW_KEEPRATIO); //resizable window;
  Mat frame;
  for (;;)
  {
    capture >> frame;
    if (frame.empty())
      continue;
    imshow("video",frame);
    char key = waitKey(5); //delay N millis, usually long enough to display and capture input
    switch (key)
    {
      case 'q':
      case 'Q':
      case 27: //escape key
        return 0;
      default:
        break;
    }
  }
  return 0;
}

int imageListProcess(vector<string> images)
{
  namedWindow("image",CV_WINDOW_KEEPRATIO); //resizable window;
  for (size_t i = 0; i < images.size(); i++)
  {
    Mat image = imread(images[i], CV_LOAD_IMAGE_GRAYSCALE); // do grayscale processing?
    imshow("image",image);
    cout << "Press a key to see the next image in the list." << endl;
    waitKey(); // wait indefinitely
  }
  return 0;
}

}

int main(int ac, char** av)
{

  if (ac != 2)
  {
    help(av);
    return 1;
  }
  std::string arg = av[1];
  vector<string> imagelist;

  if (!readStringList(arg,imagelist))
  {
    int ad = atoi(arg.c_str());
    VideoCapture capture(ad);
    return videoProcess(capture);
  }
  else
  {
    return imageListProcess(imagelist);
  }

  return 0;
}
