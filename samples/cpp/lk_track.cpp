#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int main(int argc, char * argv[])
{
  VideoCapture cap;
  if(argc>1)
  {
    string filename = argv[1];
    cap.open(filename);
  }else
  {
    cap.open(0);
  }


   Mat img, mask, vis;
   Mat gray, prevGray;
   vector <Point2f> p, p0, p0r, p1;
   vector < vector<Point2f> > tracks, new_tracks;
   vector <float> d;
   vector <Point2f> tmp;
   vector <uchar> status;
   vector <float> err;
   Size winSize(15, 15);
   int maxLevel = 3;
   TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.03);
   vector <bool> good;
   int track_len = 10;
   int maxCorners = 500;
   double qualityLevel = 0.3;
   double minDistance = 7;
   int blockSize = 7;
   for(;;)
   {
     cap>>img;
     resize(img,img, Size(720,568));
     if(img.empty()) continue;
     cvtColor(img,gray, COLOR_BGR2GRAY);
     break;
   }
   gray.copyTo(prevGray);
   tracks.clear();
   p1.clear();
   p0r.clear();

   for(;;)
   {
     cap>>img;
     resize(img,img, Size(720,568));
     img.copyTo(vis);
     cvtColor(img,gray, COLOR_BGR2GRAY);
     gray.copyTo(mask);
     mask = Scalar(255);
     if(!tracks.empty())
     {
       for(int i = 0; i < (int)tracks.size(); i++)
       {
         circle(mask, tracks[i].back(), 5, Scalar(0), -1);
       }
     }
     p.clear();
     goodFeaturesToTrack(gray, p, maxCorners, qualityLevel, minDistance, mask, blockSize);

     for(int i = 0; i < (int)p.size(); i++)
     {
       tmp.clear();
       tmp.push_back(p[i]);
       tracks.push_back(tmp);
     }
     if(tracks.size() > 0)
     {
       p0.clear();
       for(int i = 0; i<(int)tracks.size(); i++)
       {
         p0.push_back(tracks[i].back());
       }

       calcOpticalFlowPyrLK(prevGray, gray, p0, p1, status, err, winSize, maxLevel, criteria, 0, 0.001);
       calcOpticalFlowPyrLK(gray, prevGray, p1, p0r, status, err, winSize, maxLevel, criteria, 0, 0.001);
       good.clear();
       for(int i = 0; i <(int) p1.size(); i++)
       {
          float dx = abs(p0[i].x - p0r[i].x);
          float dy = abs(p0[i].y - p0r[i].y);
          good.push_back(std::max(dx,dy) < 1);
       }
       new_tracks.clear();
       for( int i = 0; i <(int) tracks.size(); i++)
       {
          if(!good[i])
          {
             continue;
          }
          tracks[i].push_back(p1[i]);

          if((int)tracks[i].size() > track_len)
          {
             // remove first element
             tracks[i].erase(tracks[i].begin());
          }
          new_tracks.push_back(tracks[i]);
          circle(vis, tracks[i].back(), 2, Scalar(0,255,0), -1);
       }

       std::swap(tracks, new_tracks);
       new_tracks.clear();
       for (int i = 0; i < (int)tracks.size(); i ++)
       {
          vector <Point2i> dst;
          dst.clear();
          std::copy(tracks[i].begin(), tracks[i].end(), std::back_inserter(dst));
          polylines(vis,dst,false,Scalar(0,255,0));
       }
     }
     gray.copyTo(prevGray);

     imshow("IMG",vis);
     int key = waitKey(1);
	 if(key == 27)
		 break;

}
  return 0;
}
