/*
 * Nanofractal is a simplified version of the Aruco Fractal marker.
 *
 * With this you will be able to detect fractal markers easily. In addition, make use of the potential of fractal
 * markers, robust to occlusions and providing information from all corners of the marker (internal and external).
 *
 * The library detects the predefined fractal markers: https://drive.google.com/file/d/1JO3V-CQIScHu2U_wwKK7kcZ0qteYbSpu/view?usp=sharing
 *
 * You only need to define the marker you are going to use (FRACTAL_3L_6, FRACTAL_4L_6,...), to create the
 * MarkerDetector object. Then call the detect method with the input image as parameter. For example,
 * nanofractal::MarkerDetector("FRACTAL_5L_6"); (See Example1).
 *
 * If you besides the four corners of each detected marker, need all visible corners (also inners corners) of the marker
 * you should call the detect method with the image and the 2d/3d point vectors as parameters (See Example2).
 *
 * Note that the 3d points of the marker are normalized, if you need real 3d information you must indicate the
 * size of the marker when you create the detector. For example, nanofractal::MarkerDetector("FRACTAL_3L_6", 0.85);
 *
 *
 *
 * // Example1: Fractal marker detection
 *
 * auto image=cv::imread("image.jpg");
 * nanofractal::MarkerDetector TheDetector = nanofractal::MarkerDetector("FRACTAL_5L_6");
 * auto markers=TheDetector.detect(image);
 * for(const auto &m:markers)
 *    m.draw(image);
 * cv::imwrite("/path/to/out.png",image);
 *
 *
 * //Example2: Fractal marker detection and get 3d/2d correspondences

 * auto image=cv::imread("image.jpg");
 * nanofractal::MarkerDetector TheDetector = nanofractal::MarkerDetector("FRACTAL_5L_6", 0.85);
 * std::vector<cv::Point2f>p2d; std::vector<cv::Point3f>p3d;
 * auto markers=TheDetector.detect(image, p3d, p2d);
 * //Here you can call solvepnp using p3d and p2d points
 * for(auto pt:p2d)
 *    cv::circle(image,pt,5,cv::Scalar(0,0,255), cv::FILLED);
 * for(const auto &m:markers)
 *    m.draw(image);
 * cv::imwrite("/path/to/out.png",image);
 *
 *
 * If you use this file in your research, you must cite:
 *
 * 1. "Fractal Markers: A New Approach for Long-Range Marker Pose Estimation Under Occlusion,", F. J. Romero-Ramirez,
 * R. Muñoz-Salinas and R. Medina-Carnicer, in IEEE Access, vol. 7, pp. 169908-169919, year 2019.
 * 2. "Speeded up detection of squared fiducial markers", Francisco J. Romero-Ramirez, Rafael Muñoz-Salinas, Rafael
 * Medina-Carnicer, Image and Vision Computing, vol 76, pages 38-47, year 2018.
 *
 *  If you have any further question, please contact fj.romero[at]uco[dot]es
*/

#include "../precomp.hpp"
#include "opencv2/objdetect/aruco2.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/3d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/features.hpp>

#include <map>
#include <vector>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <cstring>
/**
 * The FractalMarkerDetector class detects fractal markers in the images passed
 *
namespace nanofractal{
 class FractalMarkerDetector{
  public:
    //@param fractal_config possible values (FRACTAL_2L_6,FRACTAL_3L_6,FRACTAL_4L_6,FRACTAL_5L_6)
    void setParams(std::string config, float markerSize=-1);
    inline std::vector<FractalMarker> detect(const cv::Mat &img);
    inline std::vector<FractalMarker> detect(const cv::Mat &img, std::vector<cv::Point3f>& p3d,
                                             std::vector<cv::Point2f>& p2d);
  };
}
*/


namespace {
namespace nanofractal {

struct Homographer{
    Homographer(const std::vector<cv::Point2f> & out ){
        std::vector<cv::Point2f>  in={cv::Point2f(0,0),cv::Point2f(1,0),cv::Point2f(1,1),cv::Point2f(0,1)};
        H=cv::getPerspectiveTransform(in, out);
    }
    cv::Point2f operator()(const cv::Point2f &p){
        double *m=H.ptr<double>(0);
        double a=m[0]*p.x+m[1]*p.y+m[2];
        double b=m[3]*p.x+m[4]*p.y+m[5];
        double c=m[6]*p.x+m[7]*p.y+m[8];
        return cv::Point2f(a/c,b/c);
    }
    cv::Mat H;
};

/* KeyPoints Filter. Delete kpoints with low response and duplicated. */
void kfilter(std::vector<cv::KeyPoint> &kpoints)
{
    float minResp = kpoints[0].response;
    float maxResp = kpoints[0].response;
    for (auto &p:kpoints){
        p.size=40;
        if(p.response < minResp) minResp = p.response;
        if(p.response > maxResp) maxResp = p.response;
    }
    float thresoldResp = (maxResp - minResp) * 0.20f + minResp;

    for(uint32_t xi=0; xi<kpoints.size();xi++)
    {
        //Erase keypoints with low response (20%)
        if(kpoints[xi].response < thresoldResp){
            kpoints[xi].size=-1;
            continue;
        }

        //Duplicated keypoints (closer)
        for(uint32_t xj=xi+1; xj<kpoints.size();xj++)
        {
            if(pow(kpoints[xi].pt.x - kpoints[xj].pt.x,2) + pow(kpoints[xi].pt.y - kpoints[xj].pt.y,2) < 100)
            {
                if(kpoints[xj].response > kpoints[xi].response)
                    kpoints[xi] = kpoints[xj];

                kpoints[xj].size=-1;
            }
        }
    }
    kpoints.erase(std::remove_if(kpoints.begin(),kpoints.end(), [](const cv::KeyPoint &kpt){return kpt.size==-1;}), kpoints.end());
}

/*Corners classification*/
void assignClass(const cv::Mat &im, std::vector<cv::KeyPoint>& kpoints, float sizeNorm=0.f, int wsize=5)
{
    if(im.type()!=CV_8UC1)
        throw std::runtime_error("assignClass Input image must be 8UC1");
    int wsizeFull=wsize*2+1;

    cv::Mat labels = cv::Mat::zeros(wsize*2+1,wsize*2+1,CV_8UC1);
    cv::Mat thresIm=cv::Mat(wsize*2+1,wsize*2+1,CV_8UC1);

    for(auto &kp:kpoints)
    {
        float x = kp.pt.x;
        float y = kp.pt.y;

        //Convert point range from norm (-size/2, size/2) to (0,imageSize)
        if(sizeNorm>0){
            x = im.cols * (x/sizeNorm + 0.5f);
            y = im.rows * (-y/sizeNorm + 0.5f);
        }

        x= int(x+0.5f);
        y= int(y+0.5f);

        cv::Rect r= cv::Rect(x-wsize,y-wsize,wsize*2+1,wsize*2+1);
        //Check boundaries
        if(r.x<0 || r.x+r.width>im.cols || r.y<0 ||
            r.y+r.height>im.rows) continue;

        int endX=r.x+r.width;
        int endY=r.y+r.height;
        uchar minV=255,maxV=0;
        for(int row=r.y; row<endY; row++){
            const uchar *ptr=im.ptr<uchar>(row);
            for(int col=r.x; col<endX; col++)
            {
                if(minV>ptr[col]) minV=ptr[col];
                if(maxV<ptr[col]) maxV=ptr[col];
            }
        }

        if ((maxV-minV) < 25) {
            kp.class_id=0;
            continue;
        }

        double thres=(maxV+minV)/2.0;

        unsigned int nZ=0;
        for(int row=0; row<wsizeFull; row++){
            const uchar *ptr=im.ptr<uchar>( r.y+row)+r.x;
            uchar *thresPtr= thresIm.ptr<uchar>(row);
            for(int col=0; col<wsizeFull; col++){
                if( ptr[col]>thres) {
                    nZ++;
                    thresPtr[col]=255;
                }
                else thresPtr[col]=0;
            }
        }
        //set all to zero labels.setTo(cv::Scalar::all(0));
        for(int row=0; row<thresIm.rows; row++){
            uchar *labelsPtr=labels.ptr<uchar>(row);
            for(int col=0; col<thresIm.cols; col++) labelsPtr[col]=0;
        }

        uchar newLab = 1;
        std::map<uchar, uchar> unions;
        for(int row=0; row<thresIm.rows; row++){
            uchar *thresPtr=thresIm.ptr<uchar>(row);
            uchar *labelsPtr=labels.ptr<uchar>(row);
            for(int col=0; col<thresIm.cols; col++)
            {
                uchar reg = thresPtr[col];
                uchar lleft_px = 0;
                uchar ltop_px = 0;

                if(col-1>-1 && reg==thresPtr[col-1])
                    lleft_px =labelsPtr[col-1];

                if(row-1>-1 && reg==thresIm.ptr<uchar>(row-1)[col])
                    ltop_px =  labels.at<uchar>(row-1, col);

                if(lleft_px==0 && ltop_px==0)
                    labelsPtr[col] = newLab++;

                else if(lleft_px!=0 && ltop_px!=0)
                {
                    if(lleft_px < ltop_px)
                    {
                        labelsPtr[col]  = lleft_px;
                        unions[ltop_px] = lleft_px;
                    }
                    else if(lleft_px > ltop_px)
                    {
                        labelsPtr[col]  = ltop_px;
                        unions[lleft_px] = ltop_px;
                    }
                    //Same
                    else labelsPtr[col]  = ltop_px;
                }
                else
                    if(lleft_px!=0) labelsPtr[col]  = lleft_px;
                    else labelsPtr[col]  = ltop_px;
            }
        }

        int nc= newLab-1 - unions.size();
        if(nc==2)
            if(nZ > thresIm.total()-nZ) kp.class_id = 0;
            else kp.class_id = 1;
        else if (nc > 2)
            kp.class_id = 2;
    }
}

/**
 * @brief The Markers that belong to the fractal marker
 */
class FractalMarker : public std::vector<cv::Point2f>
{
public:
    FractalMarker(int id, cv::Mat m, std::vector<cv::Point3f> corners, std::vector<int> id_submarkers);
    FractalMarker(){};

    inline int nBits() { return _M.total(); }
    inline cv::Mat mat(){ return _M; }
    inline cv::Mat mask(){ return _mask; }
    inline std::vector<int> subMarkers(){ return _submarkers; }
    void addSubFractalMarker(FractalMarker submarker);
    // returns the distance of the marker side
    inline float getMarkerSize() const
    {
        return static_cast<float>(cv::norm(keypts[0].pt  - keypts[1].pt));
    }
    inline std::vector<cv::KeyPoint> getKeypts();
    inline void draw(cv::Mat &image,const cv::Scalar color=cv::Scalar(0,0,255))const;

    int id;
    std::vector<cv::KeyPoint> keypts; //Corners & class. First 4 corners are external
private:
    cv::Mat _M;
    cv::Mat _mask;
    std::vector<int> _submarkers;
};


FractalMarker::FractalMarker(int _id, cv::Mat m, std::vector<cv::Point3f> corners, std::vector<int> id_submarkers)
{
    this->id = _id;
    this->_M = m;
    for(auto pt:corners)
        keypts.push_back(cv::KeyPoint(pt.x,pt.y,-1,-1,-1,-1,0));
    _submarkers = id_submarkers;
    _mask = cv::Mat::ones(m.size(), CV_8UC1);
}

std::vector<cv::KeyPoint> FractalMarker::getKeypts()
{
    if(keypts.size() > 4) return keypts;

    int nBitsSquared = int(sqrt(mat().total()));
    float bitSize =  getMarkerSize() / (nBitsSquared+2);

    //Set submarker pixels (=1) and add border
    cv::Mat marker;
    mat().copyTo(marker);
    marker +=  -1 * (mask()-1);
    cv::Mat markerBorder;
    copyMakeBorder(marker, markerBorder, 1,1,1,1,cv::BORDER_CONSTANT,0);

    //Get inner corners
    for(int y=0; y< markerBorder.rows-1; y++)
    {
        for(int x=0; x< markerBorder.cols-1; x++)
        {
            int sum = markerBorder.at<uchar>(y, x) + markerBorder.at<uchar>(y, x+1) +
                      markerBorder.at<uchar>(y+1, x) + markerBorder.at<uchar>(y+1, x+1);

            if(sum==1)
                keypts.push_back(cv::KeyPoint(cv::Point2f(x-nBitsSquared/2.f,-(y-nBitsSquared/2.f))*bitSize,-1,-1,-1,-1,1));
            else if(sum==3)
                keypts.push_back(cv::KeyPoint(cv::Point2f(x-nBitsSquared/2.f,-(y-nBitsSquared/2.f))*bitSize,-1,-1,-1,-1,0));
            else if(sum==2)
            {
                if((markerBorder.at<uchar>(y, x) == markerBorder.at<uchar>(y+1, x+1)) && (markerBorder.at<uchar>(y, x+1) == markerBorder.at<uchar>(y+1, x)))
                    keypts.push_back(cv::KeyPoint(cv::Point2f(x-nBitsSquared/2.f,-(y-nBitsSquared/2.f))*bitSize,-1,-1,-1,-1,2));
            }
        }
    }

    return keypts;
}

void FractalMarker::addSubFractalMarker(FractalMarker submarker)
{
    int nBitsSqrt= sqrt(nBits());
    float bitSize = getMarkerSize() / (nBitsSqrt+2.0f);
    float nsubBits = submarker.getMarkerSize() / bitSize;

    int x_min = int(round(submarker.keypts[0].pt.x / bitSize + nBitsSqrt/2));
    int x_max = x_min + nsubBits;
    int y_min = int(round(-submarker.keypts[0].pt.y / bitSize + nBitsSqrt/2));
    int y_max = y_min + nsubBits;

    for(int y=y_min; y<y_max; y++){
        for(int x=x_min; x<x_max; x++){
            _mask.at<uchar>(y,x)=0;
        }
    }
}

void FractalMarker::draw(cv::Mat &in, const cv::Scalar color) const{
    float flineWidth=  std::max(1.f, std::min(5.f, float(in.cols) / 500.f));
    int lineWidth= round( flineWidth);
    for(int i=0;i<4;i++)
        cv::line(in, (*this)[i], (*this)[(i+1 )%4], color, lineWidth);

    auto p2 =  cv::Point2f(2.f * static_cast<float>(lineWidth), 2.f * static_cast<float>(lineWidth));
    cv::rectangle(in, (*this)[0] - p2, (*this)[0] + p2, cv::Scalar(0, 0, 255, 255), -1);
    cv::rectangle(in, (*this)[1] - p2, (*this)[1] + p2, cv::Scalar(0, 255, 0, 255), lineWidth);
    cv::rectangle(in, (*this)[2] - p2, (*this)[2] + p2, cv::Scalar(255, 0, 0, 255), lineWidth);
}

/**
 * @brief The FractalMarkerSet configurations
 */
class FractalMarkerSet
{
public:
    FractalMarkerSet(){};
    FractalMarkerSet(std::string config);
    void convertToMeters(float size);

    //Fractal configuration. id_marker
    std::map<int, FractalMarker> fractalMarkerCollection;
    //Correspondence number of bits and marker ids
    std::map<int, std::vector<int>> bits_ids;
    // variable indicates if the data is expressed in meters or in pixels or are normalized
    int mInfoType;/* -1:NONE, 0:PIX, 1:METERS, 2:NORMALIZE*/
    int idExternal;
};

void FractalMarkerSet::convertToMeters(float size)
{
    if (!(mInfoType == 0 || mInfoType == 2))
        throw std::runtime_error("The FractalMarkers are not expressed in pixels or normalized");

    mInfoType = 1;

    // now, get the size of a pixel, and change scale
    float pixSizeM = size / float(fractalMarkerCollection[idExternal].getMarkerSize());

    for (size_t i=0; i < fractalMarkerCollection.size(); i++)
        for(auto &kpt:fractalMarkerCollection[i].keypts)
            kpt.pt *= pixSizeM;
}

FractalMarkerSet::FractalMarkerSet(std::string str)
{
    std::stringstream stream;
    if (str=="FRACTAL_2L_6")
    {
        unsigned char _conf_2L_6[] = {
            0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x24, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0xbe, 0xab, 0xaa, 0xaa, 0x3e,
            0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0x3e, 0xab, 0xaa, 0xaa, 0x3e,
            0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0x3e, 0xab, 0xaa, 0xaa, 0xbe,
            0x00, 0x00, 0x00, 0x00, 0xab, 0xaa, 0xaa, 0xbe, 0xab, 0xaa, 0xaa, 0xbe,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00
        };
        unsigned int _conf_2L_6_len = 272;
        stream.write((char*) _conf_2L_6, sizeof(unsigned char)*_conf_2L_6_len);
    }
    else if (str=="FRACTAL_3L_6")
    {
        unsigned char _conf_3L_6[] = {
            0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01,
            0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
            0xb7, 0x6d, 0xdb, 0xbe, 0xb7, 0x6d, 0xdb, 0x3e, 0x00, 0x00, 0x00, 0x00,
            0xb7, 0x6d, 0xdb, 0x3e, 0xb7, 0x6d, 0xdb, 0x3e, 0x00, 0x00, 0x00, 0x00,
            0xb7, 0x6d, 0xdb, 0x3e, 0xb7, 0x6d, 0xdb, 0xbe, 0x00, 0x00, 0x00, 0x00,
            0xb7, 0x6d, 0xdb, 0xbe, 0xb7, 0x6d, 0xdb, 0xbe, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01,
            0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0xbe,
            0x25, 0x49, 0x12, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0x3e,
            0x25, 0x49, 0x12, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0x3e,
            0x25, 0x49, 0x12, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x25, 0x49, 0x12, 0xbe,
            0x25, 0x49, 0x12, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        };
        unsigned int _conf_3L_6_len = 480;
        stream.write((char*) _conf_3L_6, sizeof(unsigned char)*_conf_3L_6_len);
    }
    else if (str=="FRACTAL_4L_6")
    {
        unsigned char _conf_4L_6[] = {
            0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0xa9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x90, 0x00, 0x00,
            0x00, 0xef, 0xee, 0xee, 0xbe, 0xef, 0xee, 0xee, 0x3e, 0x00, 0x00, 0x00,
            0x00, 0xef, 0xee, 0xee, 0x3e, 0xef, 0xee, 0xee, 0x3e, 0x00, 0x00, 0x00,
            0x00, 0xef, 0xee, 0xee, 0x3e, 0xef, 0xee, 0xee, 0xbe, 0x00, 0x00, 0x00,
            0x00, 0xef, 0xee, 0xee, 0xbe, 0xef, 0xee, 0xee, 0xbe, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x01, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
            0x00, 0x64, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0xbe, 0xcd, 0xcc, 0x4c,
            0x3e, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0x3e, 0xcd, 0xcc, 0x4c,
            0x3e, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0x3e, 0xcd, 0xcc, 0x4c,
            0xbe, 0x00, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0x4c, 0xbe, 0xcd, 0xcc, 0x4c,
            0xbe, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00,
            0x00, 0x89, 0x88, 0x88, 0xbd, 0x89, 0x88, 0x88, 0x3d, 0x00, 0x00, 0x00,
            0x00, 0x89, 0x88, 0x88, 0x3d, 0x89, 0x88, 0x88, 0x3d, 0x00, 0x00, 0x00,
            0x00, 0x89, 0x88, 0x88, 0x3d, 0x89, 0x88, 0x88, 0xbd, 0x00, 0x00, 0x00,
            0x00, 0x89, 0x88, 0x88, 0xbd, 0x89, 0x88, 0x88, 0xbd, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00
        };
        unsigned int _conf_4L_6_len = 713;
        stream.write((char*) _conf_4L_6, sizeof(unsigned char)*_conf_4L_6_len);
    }
    else if (str=="FRACTAL_5L_6")
    {
        unsigned char _conf_5L_6[] = {
            0x02, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x79, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xbf,
            0x00, 0x00, 0x80, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xa9, 0x00, 0x00,
            0x00, 0x4f, 0xec, 0xc4, 0xbe, 0x4f, 0xec, 0xc4, 0x3e, 0x00, 0x00, 0x00,
            0x00, 0x4f, 0xec, 0xc4, 0x3e, 0x4f, 0xec, 0xc4, 0x3e, 0x00, 0x00, 0x00,
            0x00, 0x4f, 0xec, 0xc4, 0x3e, 0x4f, 0xec, 0xc4, 0xbe, 0x00, 0x00, 0x00,
            0x00, 0x4f, 0xec, 0xc4, 0xbe, 0x4f, 0xec, 0xc4, 0xbe, 0x00, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01,
            0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00,
            0x00, 0x00, 0x90, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0xbe, 0x7d, 0xcb,
            0x37, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0x3e, 0x7d, 0xcb,
            0x37, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0x3e, 0x7d, 0xcb,
            0x37, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x7d, 0xcb, 0x37, 0xbe, 0x7d, 0xcb,
            0x37, 0xbe, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00,
            0x01, 0x01, 0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0xd9, 0x89,
            0x9d, 0xbd, 0xd9, 0x89, 0x9d, 0x3d, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
            0x9d, 0x3d, 0xd9, 0x89, 0x9d, 0x3d, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
            0x9d, 0x3d, 0xd9, 0x89, 0x9d, 0xbd, 0x00, 0x00, 0x00, 0x00, 0xd9, 0x89,
            0x9d, 0xbd, 0xd9, 0x89, 0x9d, 0xbd, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00,
            0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0xbc, 0x21, 0x0d,
            0xd2, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0x3c, 0x21, 0x0d,
            0xd2, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0x3c, 0x21, 0x0d,
            0xd2, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x21, 0x0d, 0xd2, 0xbc, 0x21, 0x0d,
            0xd2, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x00, 0x01, 0x00, 0x01,
            0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x00, 0x00, 0x01, 0x01, 0x01,
            0x01, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        };
        unsigned int _conf_5L_6_len = 898;
        stream.write((char*) _conf_5L_6, sizeof(unsigned char)*_conf_5L_6_len);
    }
    else
        throw std::runtime_error("Configuration no valid: "+str+". Use: FRACTAL_2L_6, FRACTAL_3L_6, FRACTAL_4L_6 or FRACTAL_5L_6.");

    stream.read((char*)&mInfoType,sizeof(mInfoType));
    /*Number of markers*/
    int _nmarkers;
    stream.read((char*)&_nmarkers,sizeof(_nmarkers));
    stream.read((char*)&idExternal,sizeof(idExternal));

    for(int i=0; i<_nmarkers; i++)
    {
        //ID
        int id;
        stream.read((char*)&id,sizeof(id));

        //NBITS
        int nbits;
        stream.read((char*)&nbits,sizeof(nbits));

        //CORNERS
        std::vector<cv::Point3f> corners(4);
        stream.read((char*)&corners[0],sizeof(cv::Point3f)*4);

        //MAT
        cv::Mat mat;
        mat.create(sqrt(nbits), sqrt(nbits), CV_8UC1);
        stream.read((char*)mat.data, mat.elemSize() * mat.total());

        //SUBMARKERS
        int nsub;
        stream.read((char*)&nsub,sizeof(nsub));
        std::vector<int> id_submarkers(nsub);
        if (nsub > 0)
            stream.read((char*)&id_submarkers[0],sizeof(int)*nsub);

        fractalMarkerCollection[id] = FractalMarker(id, mat, corners, id_submarkers);
    }

    //Add subfractals
    for(auto &id_marker:fractalMarkerCollection)
    {
        FractalMarker &marker = id_marker.second;
        for(auto id:id_marker.second.subMarkers())
            marker.addSubFractalMarker(fractalMarkerCollection[id]);

        //Init marker kpts
        marker.getKeypts();

        bits_ids[marker.nBits()].push_back(marker.id);
    }
}


/**
 * @brief The MarkerDetector class is detecting the markers in the image passed
 *
 */
class FractalMarkerDetector{
public:
    /**@param fractal_config possible values (FRACTAL_2L_6,FRACTAL_3L_6,FRACTAL_4L_6,FRACTAL_5L_6)
     */
    void setParams(std::string fractal_config, float markerSize=-1);
    inline std::vector<FractalMarker> detect(const cv::Mat &img);
    inline std::vector<FractalMarker> detect(const cv::Mat &img, std::vector<cv::Point3f>& p3d,
                                             std::vector<cv::Point2f>& p2d);
private:
    FractalMarkerSet fractalMarkerSet;
    static inline  std::vector<cv::Point2f> sort( const  std::vector<cv::Point2f> &marker);
    static inline  float  getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p);
    static inline  int    getMarkerId(const cv::Mat &bits,int &nrotations, const std::vector<int>& markersId, const FractalMarkerSet& markerSet);
    static inline  int    perimeter(const std::vector<cv::Point2f>& a);

};


void FractalMarkerDetector::setParams(std::string config, float markerSize)
{
    fractalMarkerSet = FractalMarkerSet(config);
    if(markerSize != -1) fractalMarkerSet.convertToMeters(markerSize);

}

std::vector<FractalMarker> FractalMarkerDetector::detect(const cv::Mat &img, std::vector<cv::Point3f>& p3d,
                                                         std::vector<cv::Point2f>& p2d)
{
    cv::Mat bwimage;
    if(img.channels()==3)
        cv::cvtColor(img,bwimage,cv::COLOR_BGR2GRAY);
    else bwimage=img;

    //Fractal marker detection
    std::vector<FractalMarker> detected =  detect(bwimage);

    if(detected.size() ==0 ) return detected;
    //External corners to compute homography
    std::vector<cv::Point2f>imgpoints;
    std::vector<cv::Point3f>objpoints;
    for(auto marker:detected)
    {
        for(auto p2d_marker:marker)
            imgpoints.push_back(p2d_marker);

        for(int c=0; c<4; c++)
        {
            cv::KeyPoint kpt = fractalMarkerSet.fractalMarkerCollection[marker.id].getKeypts()[c];
            objpoints.push_back(cv::Point3f(kpt.pt.x, kpt.pt.y, 0));
        }
    }

    //FAST
    std::vector<cv::KeyPoint> kpoints;
    cv::Ptr<cv::FastFeatureDetector> fd = cv::FastFeatureDetector::create();
    fd->detect(bwimage, kpoints);
    //Filter kpoints (low response) and removing duplicated.
    kfilter(kpoints);
    assignClass(bwimage, kpoints);

    if(kpoints.size()>0){
        // Build a 2-D kd-tree from the detected keypoints using cv::flann
        cv::Mat kpointsMat((int)kpoints.size(), 2, CV_32F);
        for (int i = 0; i < (int)kpoints.size(); i++) {
            kpointsMat.at<float>(i, 0) = kpoints[i].pt.x;
            kpointsMat.at<float>(i, 1) = kpoints[i].pt.y;
        }
        cv::flann::Index kdtree(kpointsMat, cv::flann::KDTreeIndexParams(1));

        cv::Mat H = cv::findHomography(objpoints, imgpoints);

        for(auto &fm:fractalMarkerSet.fractalMarkerCollection)
        {
            std::vector<cv::Point2f> imgPoints;
            std::vector<cv::Point2f> objPoints;
            std::vector<cv::KeyPoint> objKeyPoints = fm.second.getKeypts();

            for(auto kpt : objKeyPoints)
                objPoints.push_back(cv::Point2f(kpt.pt.x, kpt.pt.y));

            cv::perspectiveTransform(objPoints, imgPoints, H);

            //We consider only markers whose internal points are separated by a specific distance.
            bool consider=true;
            for(size_t i=0; i<imgPoints.size()-1 && consider; i++)
                for(size_t j=i+1; j<imgPoints.size() && consider; j++)
                    if(pow(imgPoints[i].x-imgPoints[j].x, 2) + pow(imgPoints[i].y-imgPoints[j].y, 2) < 150)
                        consider=false;

            if(consider)
            {
                for(size_t idx=0; idx<imgPoints.size(); idx++)
                {
                    if(imgPoints[idx].x > 0 && imgPoints[idx].x < img.cols
                        && imgPoints[idx].y>0 && imgPoints[idx].y<img.rows)
                    {
                        cv::Mat query(1, 2, CV_32F);
                        query.at<float>(0, 0) = imgPoints[idx].x;
                        query.at<float>(0, 1) = imgPoints[idx].y;
                        cv::Mat resIdx, resDist;
                        // radius=10 → squared-L2 ≤ 10, same threshold as the original picoflann search
                        int found = kdtree.radiusSearch(query, resIdx, resDist, 10.0, 2);
                        if (found == 1)
                        {
                            int ki = resIdx.at<int>(0, 0);
                            if(kpoints[ki].class_id == objKeyPoints[idx].class_id)
                            {
                                p2d.push_back(kpoints[ki].pt);
                                p3d.push_back(cv::Point3f(objPoints[idx].x, objPoints[idx].y, 0));
                            }
                        }
                    }
                }
            }
            else
            {
                //If a marker is detected and it is not possible take all their corners,
                //at least take the external one!
                for(auto markerDetected:detected)
                {
                    if(markerDetected.id == fm.first)
                    {
                        for(int c=0; c<4; c++)
                        {
                            cv::Point2f pt = markerDetected.keypts[c].pt;
                            p3d.push_back(cv::Point3f(pt.x,pt.y,0));
                            p2d.push_back(markerDetected[c]);
                        }
                        break;
                    }
                }
            }
        }

        if(p2d.size()>0)
        {
            //corner subpixel
            cv::Size winSize = cv::Size(4, 4);
            cv::Size zeroZone = cv::Size( -1, -1 );
            cv::TermCriteria criteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005);      //cv::cornerSubPix(image, points, winSize, zeroZone, criteria);
            cornerSubPix(bwimage, p2d, winSize, zeroZone, criteria);
        }
    }
    return detected;
}

std::vector<FractalMarker>  FractalMarkerDetector::detect(const cv::Mat &img){

    cv::Mat bwimage,thresImage;

    std::vector<std::pair<int, std::vector<cv::Point2f>>> candidates;

    std::vector<FractalMarker> DetectedFractalMarkers;

    //first, convert to bw
    if(img.channels()==3)
        cv::cvtColor(img,bwimage,cv::COLOR_BGR2GRAY);
    else bwimage=img;


    ///////////////////////////////////////////////////
    // Adaptive Threshold to detect border
    int adaptiveWindowSize=std::max(int(3),int(15*float(bwimage.cols)/1920.));
    if( adaptiveWindowSize%2==0) adaptiveWindowSize++;
    cv::adaptiveThreshold(bwimage, thresImage, 255.,cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, adaptiveWindowSize, 7);

    ///////////////////////////////////////////////////
    // compute marker candidates by detecting contours
    //if image is eroded, minSize must be adapted
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> approxCurve;
    cv::findContours(thresImage, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    //analyze  it is a paralelepiped likely to be the marker
    for (unsigned int i = 0; i < contours.size(); i++)
    {
        // check it is a possible element by first checking that is is large enough
        if (120 > int(contours[i].size())  ) continue;
        // can approximate to a convex rect?
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.05, true);

        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve)) continue;
        // add the points
        std::vector<cv::Point2f> markerCandidate;
        for (int j = 0; j < 4; j++)
            markerCandidate.push_back( cv::Point2f( approxCurve[j].x,approxCurve[j].y));

        //sort corner in clockwise direction
        markerCandidate=sort(markerCandidate);

        //extract the code
        //obtain the intensities of the bits using homography
        Homographer hom(markerCandidate);

        for(auto b_vm:fractalMarkerSet.bits_ids)
        {
            int nbitsWithBorder = sqrt(b_vm.first)+2;
            cv::Mat bits(nbitsWithBorder,nbitsWithBorder,CV_8UC1);
            int pixelSum=0;

            for(int r=0;r<bits.rows;r++){
                for(int c=0;c<bits.cols;c++){
                    auto pixelValue=uchar(0.5+getSubpixelValue(bwimage,hom(cv::Point2f(  float(c+0.5) / float(bits.cols) ,  float(r+0.5) / float(bits.rows)  ))));
                    bits.at<uchar>(r,c)=pixelValue;
                    pixelSum+=pixelValue;
                }
            }

            //threshold by the average value
            double mean=double(pixelSum)/double(bits.cols*bits.rows);
            cv::threshold(bits,bits,mean,255,cv::THRESH_BINARY);

            //now, analyze the inner code to see if is a marker.
            //  If so, rotate to have the points properly sorted
            int nrotations=0;

            int id=getMarkerId(bits, nrotations, b_vm.second, fractalMarkerSet);

            if(id==-1) continue;//not a marker
            std::rotate(markerCandidate.begin(),markerCandidate.begin() + 4 - nrotations,markerCandidate.end());
            candidates.push_back(std::make_pair(id,markerCandidate));
        }
    }

    ////////////////////////////////////////////
    //remove duplicates
    // sort by id and within same id set the largest first
    std::sort(candidates.begin(), candidates.end(),[](const std::pair<int, std::vector<cv::Point2f>> &a,const std::pair<int, std::vector<cv::Point2f>> &b){
        if( a.first<b.first) return true;
        else if( a.first==b.first) return perimeter(a.second)>perimeter(b.second);
        else return false;
    });

    // Using std::unique remove duplicates
    auto ip = std::unique(candidates.begin(), candidates.end(),[](const std::pair<int, std::vector<cv::Point2f>> &a,const std::pair<int, std::vector<cv::Point2f>> &b){return a.first==b.first;});
    candidates.resize(std::distance(candidates.begin(), ip));

    if(candidates.size()>0){
        ////////////////////////////////////////////
        //finally subpixel corner refinement
        int halfwsize= 4*float(bwimage.cols)/float(bwimage.cols) +0.5 ;
        std::vector<cv::Point2f> Corners;
        for (const auto &m:candidates)
            Corners.insert(Corners.end(), m.second.begin(),m.second.end());
        cv::cornerSubPix(bwimage, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
        // copy back to the markers
        for (unsigned int i = 0; i < candidates.size(); i++)
        {
            DetectedFractalMarkers.push_back(fractalMarkerSet.fractalMarkerCollection[candidates[i].first]);
            for (int c = 0; c < 4; c++) DetectedFractalMarkers[i].push_back(Corners[i * 4 + c]);
        }
    }

    //Done
    return DetectedFractalMarkers;
}

int  FractalMarkerDetector::perimeter(const std::vector<cv::Point2f>& a)
{
    int sum = 0;
    for (size_t i = 0; i < a.size(); i++)
        sum+=cv::norm( a[i]-a[(i + 1) % a.size()]);
    return sum;
}

int FractalMarkerDetector:: getMarkerId(const cv::Mat &bits, int &nrotations, const std::vector<int>& markersId, const FractalMarkerSet& fmset){

    auto rotate=[](const cv::Mat& in)
    {
        cv::Mat out(in.size(),in.type());
        for (int i = 0; i < in.rows; i++)
            for (int j = 0; j < in.cols; j++)
                out.at<uchar>(i, j) = in.at<uchar>(in.cols - j - 1, i);
        return out;
    };

    //first check that outer is all black
    for(int x=0;x<bits.cols;x++){
        if( bits.at<uchar>(0,x)!=0)return -1;
        if( bits.at<uchar>(bits.rows-1,x)!=0)return -1;
        if( bits.at<uchar>(x,0)!=0)return -1;
        if( bits.at<uchar>(x,bits.cols-1)!=0)return -1;
    }

    //now, get the inner bits wo the black border
    cv::Mat bit_inner(bits.cols-2,bits.rows-2,CV_8UC1);
    for(int r=0;r<bit_inner.rows;r++)
        for(int c=0;c<bit_inner.cols;c++)
            bit_inner.at<uchar>(r,c)=bits.at<uchar>(r+1,c+1);

    nrotations = 0;
    do
    {
        for(auto idx:markersId)
        {
            FractalMarker fm = fmset.fractalMarkerCollection.at(idx);

            //Apply mask to substract submarkers

            cv::Mat masked;
            bit_inner.copyTo(masked, fm.mask());

            //Code without submarkers == fractal marker?
            if (cv::countNonZero(masked != fm.mat()*255) == 0)
                return idx;
        }
        bit_inner = rotate(bit_inner);
        nrotations++;
    } while (nrotations < 4);

    return -1;
}

float FractalMarkerDetector::getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p){

    float intpartX;
    float decpartX=std::modf(p.x,&intpartX);
    float intpartY;
    float decpartY=std::modf(p.y,&intpartY);

    cv::Point tl;

    if (decpartX>0.5) {
        if (decpartY>0.5) tl=cv::Point(intpartX,intpartY);
        else tl=cv::Point(intpartX,intpartY-1);
    }
    else{
        if (decpartY>0.5) tl=cv::Point(intpartX-1,intpartY);
        else tl=cv::Point(intpartX-1,intpartY-1);
    }
    if(tl.x<0) tl.x=0;
    if(tl.y<0) tl.y=0;
    if(tl.x>=im_grey.cols)tl.x=im_grey.cols-1;
    if(tl.y>=im_grey.cols)tl.y=im_grey.rows-1;
    return (1.f-decpartY)*(1.-decpartX)*float(im_grey.at<uchar>(tl.y,tl.x))+
           decpartX*(1-decpartY)*float(im_grey.at<uchar>(tl.y,tl.x+1))+
           (1-decpartX)*decpartY*float(im_grey.at<uchar>(tl.y+1,tl.x))+
           decpartX*decpartY*float(im_grey.at<uchar>(tl.y+1,tl.x+1));
}


std::vector<cv::Point2f>  FractalMarkerDetector::sort( const  std::vector<cv::Point2f> &marker){
    std::vector<cv::Point2f>  res_marker=marker;
    /// sort the points in anti-clockwise order
    // trace a line between the first and second point.
    // if the thrid point is at the right side, then the points are anti-clockwise
    double dx1 = res_marker[1].x - res_marker[0].x;
    double dy1 = res_marker[1].y - res_marker[0].y;
    double dx2 = res_marker[2].x - res_marker[0].x;
    double dy2 = res_marker[2].y - res_marker[0].y;
    double o = (dx1 * dy2) - (dy1 * dx2);

    if (o < 0.0)
    {  // if the third point is in the left side, then sort in anti-clockwise order
        std::swap(res_marker[1], res_marker[3]);
    }
    return res_marker;
}
} // namespace nanofractal
} // anonymous namespace

// ---------------------------------------------------------------------------
// cv::aruco2 wrappers
// ---------------------------------------------------------------------------
namespace cv {
namespace aruco2 {

static std::string fractalTypeName(FractalType ft) {
    switch (ft) {
        case FRACTAL_2L_6: return "FRACTAL_2L_6";
        case FRACTAL_3L_6: return "FRACTAL_3L_6";
        case FRACTAL_4L_6: return "FRACTAL_4L_6";
        case FRACTAL_5L_6: return "FRACTAL_5L_6";
    }
    return "FRACTAL_3L_6";
}

void generateFractalImage(OutputArray _img, const FractalType &ftype, int bitSize) {
    nanofractal::FractalMarkerSet fmset(fractalTypeName(ftype));

    // The innermost marker (highest id) controls the per-bit pixel size.
    // bitSize is the desired pixel size of one bit in that innermost marker.
    auto &innerM  = (--fmset.fractalMarkerCollection.end())->second;
    auto &externM = fmset.fractalMarkerCollection.at(fmset.idExternal);

    int nBitsInner = int(std::round(std::sqrt(float(innerM.mat().total()))));

    // bs = normalized-units per pixel.  Sized so the inner marker's bits are bitSize px each.
    float bs = innerM.getMarkerSize() / float(bitSize * (nBitsInner + 2));

    // Outer marker pixel dimensions
    int markerSizePx = int(std::round(externM.getMarkerSize() / bs));
    int nBitsExtern  = int(std::round(std::sqrt(float(externM.mat().total()))));
    float extMBS     = float(markerSizePx) / float(nBitsExtern + 2); // px per bit (outer)

    cv::Mat img = cv::Mat::zeros(markerSizePx, markerSizePx, CV_8UC1);

    // Render outer marker: bit(y,x)==1 → white rectangle
    {
        cv::Mat m = externM.mat();
        for (int y = m.rows - 1; y >= 0; y--)
            for (int x = m.cols - 1; x >= 0; x--)
                if (m.at<uchar>(y, x) == 1)
                    img(cv::Range(int((1 + y) * extMBS), int((y + 2) * extMBS)),
                        cv::Range(int((1 + x) * extMBS), int((x + 2) * extMBS))).setTo(255);
    }

    // Render sub-markers depth-first; offsets are in the outer marker's pixel frame
    cv::Point2f extTL = externM.keypts[0].pt; // TL corner in normalised space

    std::vector<int> stack;
    for (int id : externM.subMarkers())
        stack.push_back(id);

    while (!stack.empty()) {
        int subId = stack.back(); stack.pop_back();
        auto &subM = fmset.fractalMarkerCollection.at(subId);
        cv::Mat m  = subM.mat();
        int nBitsSub = int(std::round(std::sqrt(float(m.total()))));

        float subSizePx = subM.getMarkerSize() / bs;
        float subMBS    = subSizePx / float(nBitsSub + 2);

        // Pixel offset of this sub-marker's TL corner from the outer image origin
        cv::Point2f subTL = subM.keypts[0].pt;
        float offX = std::fabs(subTL.x - extTL.x) / bs;
        float offY = std::fabs(subTL.y - extTL.y) / bs;

        for (int y = m.rows - 1; y >= 0; y--)
            for (int x = m.cols - 1; x >= 0; x--)
                if (m.at<uchar>(y, x) == 1)
                    img(cv::Range(int((1 + y) * subMBS + offY), int((y + 2) * subMBS + offY)),
                        cv::Range(int((1 + x) * subMBS + offX), int((x + 2) * subMBS + offX))).setTo(255);

        for (int id : subM.subMarkers())
            stack.push_back(id);
    }

    // White outer border, 1 outer-bit wide
    int borderPx = int(extMBS);
    cv::copyMakeBorder(img, img, borderPx, borderPx, borderPx, borderPx,
                       cv::BORDER_CONSTANT, cv::Scalar::all(255));

    _img.assign(img);
}

std::vector<FractalMarker> detectFractals(InputArray _img, FractalType ftype) {
    cv::Mat img = _img.getMat();

    nanofractal::FractalMarkerDetector detector;
    detector.setParams(fractalTypeName(ftype));

    std::vector<cv::Point3f> p3d;
    std::vector<cv::Point2f> p2d;
    auto detected = detector.detect(img, p3d, p2d);

    if (detected.empty()) return {};

    // All entries in detected are sub-markers of the same physical fractal marker.
    // Find the outer marker to use its corners; fall back to detected[0] if not found.
    nanofractal::FractalMarkerSet fmset(fractalTypeName(ftype));
    const nanofractal::FractalMarker *outerDetected = nullptr;
    for (const auto &d : detected)
        if (d.id == fmset.idExternal) { outerDetected = &d; break; }
    if (!outerDetected) outerDetected = &detected[0];

    FractalMarker m;
    m.type = ftype;
    m.id   = outerDetected->id;
    for (int c = 0; c < 4; c++)
        m.corners.push_back((*outerDetected)[c]);

    if (!p2d.empty()) {
        m.imgPoints = p2d;
        m.objPoints = p3d;
    } else {
        // Fallback: use only the 4 outer corners
        auto &tmplOuter = fmset.fractalMarkerCollection.at(fmset.idExternal);
        for (int c = 0; c < 4; c++) {
            m.imgPoints.push_back(m.corners[c]);
            m.objPoints.push_back(cv::Point3f(
                tmplOuter.keypts[c].pt.x, tmplOuter.keypts[c].pt.y, 0.f));
        }
    }
    return {m};
}

void drawDetected(InputOutputArray _image, const std::vector<FractalMarker> &fractals,
                          Scalar color, bool drawAllImagePoints) {
    cv::Mat image = _image.getMat();

    float lineWidthF = std::max(1.f, std::min(5.f, float(image.cols) / 500.f));
    int lineWidth = int(std::round(lineWidthF));
    int dotRadius = std::max(2, int(std::round(lineWidthF * 1.2f)));

    for (const auto &m : fractals) {
        if (m.corners.size() < 4) continue;

        for (int i = 0; i < 4; i++)
            cv::line(image, m.corners[i], m.corners[(i + 1) % 4], color, lineWidth);

        // Red dot on corners[0] to show orientation
        auto p2 = cv::Point2f(2.f * lineWidthF, 2.f * lineWidthF);
        cv::rectangle(image, m.corners[0] - p2, m.corners[0] + p2,
                      cv::Scalar(0, 0, 255), -1);

        // ID at centroid
        cv::Point2f center(0, 0);
        for (const auto &c : m.corners) center += c;
        center *= 0.25f;
        cv::putText(image, std::to_string(m.id), center,
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, lineWidth);

        if (drawAllImagePoints)
            for (const auto &pt : m.imgPoints)
                cv::circle(image, pt, dotRadius, color, -1);
    }
}

void getSolvePnpPoints(const FractalMarker &fractal, OutputArray objPoints,
                       OutputArray imgPoints, float markerSize) {
    // 3D coords are in normalized space [-1, +1] (total span = 2 units)
    float scale = markerSize / 2.0f;
    int n = int(fractal.imgPoints.size());

    if (n == 0) {
        imgPoints.assign(cv::Mat());
        objPoints.assign(cv::Mat());
        return;
    }

    cv::Mat imgMat(n, 1, CV_32FC2);
    cv::Mat objMat(n, 1, CV_32FC3);
    for (int i = 0; i < n; i++) {
        imgMat.at<cv::Vec2f>(i) = {fractal.imgPoints[i].x, fractal.imgPoints[i].y};
        objMat.at<cv::Vec3f>(i) = {fractal.objPoints[i].x * scale,
                                   fractal.objPoints[i].y * scale,
                                   fractal.objPoints[i].z * scale};
    }
    imgPoints.assign(imgMat);
    objPoints.assign(objMat);
}

} // namespace aruco2
} // namespace cv

