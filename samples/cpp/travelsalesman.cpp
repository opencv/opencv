#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void DrawTravelMap(Mat &img, vector<Point> &p, vector<int> &n);

class TravelSalesman : public ml::SimulatedAnnealingSolver
{
private :
    vector<Point> &posCity;
    vector<int> &next;
    RNG rng;
    int d0,d1,d2,d3;

public:

    TravelSalesman(vector<Point> &p,vector<int> &n):posCity(p),next(n)
    {
        rng = theRNG();
    };
    /** Give energy value for  a state of system.*/
    virtual double energy();
    /** Function which change the state of system (random pertubation).*/
    virtual void changedState();
    /** Function to reverse to the previous state.*/
    virtual void reverseChangedState();

};

void TravelSalesman::changedState()
{
    d0 = rng.uniform(0,static_cast<int>(posCity.size()));
    d1 = next[d0];
    d2 = next[d1];
    d3 = next[d2];
    int d0Tmp = d0;
    int d1Tmp = d1;
    int d2Tmp = d2;

    next[d0Tmp] = d2;
    next[d2Tmp] = d1;
    next[d1Tmp] = d3;
}


void TravelSalesman::reverseChangedState()
{
    next[d0] = d1;
    next[d1] = d2;
    next[d2] = d3;
}

double TravelSalesman::energy()
{
    double e=0;
    for (size_t i = 0; i < next.size(); i++)
    {
        e +=  norm(posCity[i]-posCity[next[i]]);
    }
    return e;
}


void DrawTravelMap(Mat &img, vector<Point> &p, vector<int> &n)
{
    for (size_t i = 0; i < n.size(); i++)
    {
        circle(img,p[i],5,Scalar(0,0,255),2);
        line(img,p[i],p[n[i]],Scalar(0,255,0),2);
    }
}
int main(void)
{
    int nbCity=40;
    Mat img(500,500,CV_8UC3,Scalar::all(0));
    RNG &rng=theRNG();
    int radius=static_cast<int>(img.cols*0.45);
    Point center(img.cols/2,img.rows/2);

    vector<Point> posCity(nbCity);
    vector<int> next(nbCity);
    for (size_t i = 0; i < posCity.size(); i++)
    {
        double theta = rng.uniform(0., 2 * CV_PI);
        posCity[i].x = static_cast<int>(radius*cos(theta)) + center.x;
        posCity[i].y = static_cast<int>(radius*sin(theta)) + center.y;
        next[i]=(i+1)%nbCity;
    }
    TravelSalesman ts(posCity,next);
    ts.setCoolingRatio(0.99);
    ts.setInitialTemperature(100);
    ts.setIterPerStep(10000*nbCity);
    ts.setFinalTemperature(100*0.97);
    DrawTravelMap(img,posCity,next);
    imshow("Map",img);
    waitKey(10);
    for (int i = 0; i < 100; i++)
    {
        ts.run();
        img = Mat::zeros(img.size(),CV_8UC3);
        DrawTravelMap(img, posCity, next);
        imshow("Map", img);
        waitKey(10);
        double ti=ts.getFinalTemperature();
        cout<<ti <<"  -> "<<ts.energy()<<"\n";
        ts.setInitialTemperature(ti);
        ts.setFinalTemperature(ti*0.97);
    }
    return 0;
}
