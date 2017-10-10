#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#ifdef HAVE_OPENCV_VIZ

const String keys =
"{Aide h usage ? help  |     | print this message   }"
"{@arg1                |     | Full path to color imag (3 channels)}"
;


struct Histo3DData {
    Mat histogram;
    int seuil;
    double threshold;
    Ptr<viz::Viz3d> fen3D;
    int nbWidget;
    bool status;
    double maxH;
    int code;
};

void DrawHistogram3D(Histo3DData &);
void AddSlidebar(String sliderName, String windowName, int sliderMin, int sliderMax, int valeurDefaut, int *sliderVal, void(*f)(int, void *), void *r);
void UpdateThreshold(int , void * r);
void  KeyboardViz3d(const viz::KeyboardEvent &w, void *t);


void DrawHistogram3D(Histo3DData &h)
{
    //! [get_cube_size]
    int planSize = h.histogram.step1(0);
    int cols = h.histogram.step1(1);
    int rows = planSize / cols;
    int plans = h.histogram.total() / planSize;
    h.fen3D->removeAllWidgets();
    h.nbWidget=0;
    if (h.nbWidget==0)
        h.fen3D->showWidget("Axis", viz::WCoordinateSystem(10));
    //! [get_cube_size]
    //! [get_cube_values]
    for (int k = 0; k < plans; k++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double x = h.histogram.at<float>(k, i, j);
                if (x >= h.threshold)
                {
                    double r=std::max(x/h.maxH,0.1);
                    viz::WCube s(Point3d(k - r / 2, i - r / 2, j - r / 2), Point3d(k + r / 2, i + r / 2, j + r / 2), false, viz::Color(j / double(plans) * 255, i / double(rows) * 255, k / double(cols) * 255));
                    h.fen3D->showWidget(format("I3d%d", h.nbWidget++), s);
                }
            }
        }
    }
    //! [get_cube_values]
    h.status = false;
}
//! [viz_keyboard_callback]
void  KeyboardViz3d(const viz::KeyboardEvent &w, void *t)
{
   Histo3DData *x=(Histo3DData *)t;
   if (w.action)
       cout << "you pressed "<< w.symbol<< " in viz window "<<x->fen3D->getWindowName()<<"\n";
   x->code= w.code;
   switch (w.code) {
   case '/':
           x->status=true;
           x->threshold *= 0.9;
       break;
   case '*':
       x->status = true;
           x->threshold *= 1.1;
       break;
    }
   if (x->status)
   {
       cout <<  x->threshold << "\n";
       DrawHistogram3D(*x);
   }
}
//! [viz_keyboard_callback]


void AddSlidebar(String sliderName, String windowName, int sliderMin, int sliderMax, int defaultSlider, int *sliderVal, void(*f)(int, void *), void *r)
{
    createTrackbar(sliderName, windowName, sliderVal, 1, f, r);
    setTrackbarMin(sliderName, windowName, sliderMin);
    setTrackbarMax(sliderName, windowName, sliderMax);
    setTrackbarPos(sliderName, windowName, defaultSlider);
}


void UpdateThreshold(int , void * r)
{
    Histo3DData *h = (Histo3DData *)r;
    h->status=true;
    h->threshold = h->seuil/1000000.0;
    cout<<"Widget : "<<h->nbWidget<<","<< h->threshold<<"\n";
}

int main (int argc,char **argv)
{
    //! [command_line_parser]
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    String nomFic = parser.get<String>(0);
    Mat img;
    if (nomFic.length() != 0)
    {
        img = imread(nomFic, IMREAD_COLOR);
        if (img.empty())
        {
            cout << "Image does not exist!";
            return 0;
        }
    }
    //! [command_line_parser]
    //! [synthetic_image]
    else
    {
        img = Mat(512,512,CV_8UC3);
        parser.printMessage();
        RNG r;
        r.fill(img(Rect(0, 0, 256, 256)), RNG::NORMAL, Vec3b(60, 40, 50), Vec3b(10, 5, 20));
        r.fill(img(Rect(256, 0, 256, 256)), RNG::NORMAL, Vec3b(160, 10, 50), Vec3b(20, 5, 10));
        r.fill(img(Rect(0, 256, 256, 256)), RNG::NORMAL, Vec3b(90, 100, 50), Vec3b(10, 20, 20));
        r.fill(img(Rect(256, 256, 256, 256)), RNG::NORMAL, Vec3b(100, 10, 150), Vec3b(10, 5, 40));
    }
    //! [synthetic_image]
    //! [calchist_for_histo3d]
    Histo3DData h;
    h.status=true;
    h.seuil=90;
    h.threshold= h.seuil/1000000.0;
    float hRange[] = { 0, 256 };
    const float* etendu[] = { hRange, hRange,hRange };
    int hBins = 32;
    int histSize[] = { hBins, hBins , hBins  };
    int channel[] = { 2, 1,0 };
    calcHist(&img, 1, channel, Mat(), h.histogram, 3, histSize, etendu, true, false);
    normalize(h.histogram, h.histogram, 100.0/(img.total()), 0, NORM_MINMAX, -1, Mat());
    minMaxIdx(h.histogram,NULL,&h.maxH,NULL,NULL);
    //! [calchist_for_histo3d]
    //! [slide_bar_for_thresh]
    namedWindow("Image");
    imshow("Image",img);
    AddSlidebar("threshold","Image",0,100,h.seuil,&h.seuil, UpdateThreshold,&h);
    waitKey(30);
    //! [slide_bar_for_thresh]
    //! [manage_viz_imshow_window]
    h.fen3D = new viz::Viz3d("3D Histogram");
    h.nbWidget=0;
    h.fen3D->registerKeyboardCallback(KeyboardViz3d,&h);
    DrawHistogram3D(h);
    while (h.code!=27)
    {
        h.fen3D->spinOnce(1);
        if (h.status)
            DrawHistogram3D(h);
        if (h.code!=27)
            h.code= waitKey(30);
    }
    //! [manage_viz_imshow_window]
    return 0;
}
#else

int main(int argc, char **argv)
{
cout << " you need VIZ module\n";
return 0;
}
#endif
