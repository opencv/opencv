#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main11(int argc, char* argv[])
{
    TickMeter timer;
    vector<double> times;

    Scalar x;
    double s = 0.0;
    
    int nIters = 100;
    for (int i = 0; i < nIters; i++)
    {
        timer.start();
        
            Mat m(4096, 1024, CV_32F);
            //m.setTo(Scalar(33.0));
            randu(m, 0, 256);
            x = sum(m);
                  
        timer.stop();
        times.push_back(timer.getTimeMilli());
        timer.reset();
        
        x = sum(m);
        s += x[0];
    }

    for (int i = 0; i < nIters; i++)
        printf("time[%d] = %.2f ms\n", i, times[i]);
    
    printf("s = %f\n", s);
}

int main3(int argc, char* argv[])
{
    int w = 1280;
    int h = 720;
    
    Mat m1(w, h, CV_8U);
    Mat m2(w, h, CV_8U);
    Mat m3(w, h, CV_8U);
    Mat dst(w, h, CV_8U);
    
    Scalar x;
    double s = 0.0;
    
    TickMeter timer1;
    TickMeter timer2;
    TickMeter timer3;
    
    int nIters = 100;
    for (int i = 0; i < nIters; i++)
    {
        randu(m1, 0, 256);
        randu(m2, 0, 256);
        
        equalizeHist(m1, m1);
        equalizeHist(m2, m2);
        
        timer1.start();
        add(m1, m2, dst);
        timer1.stop();
              
        normalize(dst, dst, dst.total());
        
        timer2.start();
        m3 = m1 + m2;
        timer2.stop();
        
        timer3.start();        
        dst = m3 + dst;
        timer3.stop();
        
        x = sum(dst);
        s += x[0];
    }
    
    printf("s = %f\n", s);
    printf("timer1 = %.2f ms\n", timer1.getTimeMilli()/nIters);
    printf("timer2 = %.2f ms\n", timer2.getTimeMilli()/nIters);
    printf("timer3 = %.2f ms\n", timer3.getTimeMilli()/nIters);
}

const char* message = "Hello Android!";

int main2(int argc, char* argv[])
{
  // print message to console
  printf("%s\n", message);
  
  // put message to simple image
  Size textsize = getTextSize(message, CV_FONT_HERSHEY_COMPLEX, 3, 5, 0);
  Mat img(textsize.height + 20, textsize.width + 20, CV_32FC1, Scalar(230,230,230));
  putText(img, message, Point(10, img.rows - 10), CV_FONT_HERSHEY_COMPLEX, 3, Scalar(0, 0, 0), 5);
  
  // save\show resulting image
#if ANDROID
  imwrite("/mnt/sdcard/HelloAndroid.png", img);
#else
  imshow("test", img);
  waitKey();
#endif

return 0;
}

