#include "charuco2.h"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/flann.hpp"
#include <map>
#include <queue>

namespace     {
/**
 * @brief The Marker class is a marker detectable by the library
 * It is a vector where each corner is a corner of the detected marker
 */
class Marker : public std::vector<cv::Point2f>
{
public:
    // id of  the marker
    int id=-1;
    //id of the dict
    int dict=-1;
    //max distance between any two similar corners
    int cornerMaxDistance(const Marker &m2);
};
struct DetectorParameters {
    int boxFilterSize=15,thres=3; //values for adaptive thresholding
    int minSize=10;//minimum size of a contour side to be considered as a marker candidate
    int maxAttemptsPerCandidate=5;//number of attempts to identify a candidate by slightly altering the corners
    std::vector<cv::aruco::Dictionary> dicts= {cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12)};
    // [0,1] ; maximum number of times a contour can revisit any of its pixels (1 is the minimum which is the starting point)
    //if you set a high value (std::numeric_limits<int>::max()) the algorithm behaves as the normal moore contour tracer
    float maxTimesRevisited=0.05; //1 equals tradional algo,
    /// number of bits of the marker border, i.e. marker border width (default 1).
    float  markerBorderBits=1; //i do not see this useful. all dicts have 1 border bit but its used in opencv  aruco and I keep it here
    double errorCorrectionRate=0;//The default 0.6 value in aruco opencv is very dangerous. It causes many false positives.
    double maxErroneousBitsInBorderRate=0;//maximum rate of erroneous bits in the border. Default 0 means no error allowed.
    bool detectInvertedMarker=false;//if the markers are printed in white over black background
};
/** @brief The MarkerDetector class is detecting the markers in the image passed */
class MarkerDetector{
public:
    // The only function you need to call
    static inline std::vector<Marker> detect(const cv::Mat &img, const DetectorParameters &params=DetectorParameters(),std::vector<Marker> *candidatesOut=nullptr,cv::Mat thresImage=cv::Mat());
private:
    static inline Marker sort( const  Marker &marker);
    static inline float  getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p);
    static inline int   getMarkerId(  cv::Mat  candidateBits,int &idx, int &nrotations, const DetectorParameters &params,int dictIndex);
    static inline int isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) ;
    static std::vector<std::vector<cv::Point>> visitedAwareTracingContour(cv::Mat &padded, size_t minSize = 1,float maxRevisited=0.1) ;
    static int getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) ;
    static void thres255Adaptive(cv::Mat &in,cv::Mat &out,int off=2,int thres=5);
    static bool isAInnerMarker(const cv::Mat &grey,const Marker &m);
};
struct Homographer{
    Homographer(const std::vector<cv::Point2f> & out ){
        std::vector<cv::Point2f>  in={cv::Point2f(0,0),cv::Point2f(1,0),cv::Point2f(1,1),cv::Point2f(0,1)};
        H=cv::getPerspectiveTransform(in, out);
    }
    cv::Point2f operator()(const cv::Point2f &p){
        double *m=H.ptr<double>(0);
        double c=m[6]*p.x+m[7]*p.y+m[8];
        return cv::Point2f((m[0]*p.x+m[1]*p.y+m[2])/c,(m[3]*p.x+m[4]*p.y+m[5])/c);
    }
    cv::Mat H;
};

int Marker::cornerMaxDistance(const Marker &m2)
{
    int md=0;
    for(int i=0;i<4;i++){
        int d=int(((*this)[i].x-m2[i].x)*((*this)[i].x-m2[i].x)+((*this)[i].y-m2[i].y)*((*this)[i].y-m2[i].y));
        if(d>md) md=d;
    }
    return md;
}


//Marker intersection. Tells the marker with most corners into another. 0 if no intersection or tie
int MarkerDetector::isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) {
    // Lambda for point-in-polygon test (Ray Casting)
    auto countInside = [](const std::vector<cv::Point2f>& source, const std::vector<cv::Point2f>& target) -> int {
        int count = 0;
        for (const auto& pt : source) {
            bool inside = false;
            // Fixed 4-side loop logic
            for (int i = 0, j = 3; i < 4; j = i++) {
                if (((target[i].y > pt.y) != (target[j].y > pt.y)) &&
                    (pt.x < (target[j].x - target[i].x) * (pt.y - target[i].y) / (target[j].y - target[i].y) + target[i].x)) {
                    inside = !inside;
                }
            }
            if (inside) count++;
        }
        return count;
    };
    // Count how many corners of A are in B
    int aInB = countInside(a, b);
    // Count how many corners of B are in A
    int bInA = countInside(b, a);
    // Rule 1: Must contain at least one corner
    if (aInB == 0 && bInA == 0) return 0;
    // Rule 2: Compare counts
    if (aInB > bInA) return 1;
    if (bInA > aInB) return 2;
    // Default: Tie or no relative dominance
    return 0;
}

std::vector<Marker>  MarkerDetector::detect(const cv::Mat &img, const DetectorParameters &params,std::vector<Marker> *candidatesOut,cv::Mat ThresImIn){
    cv::Mat greyimage,thresImage;
    std::vector<Marker> DetectedMarkers;
    //first, convert to bw
    if(img.channels()==3)
        cv::cvtColor(img,greyimage,cv::COLOR_BGR2GRAY);
    else greyimage=img;
    /////////////////// Adaptive Threshold to detect border
    //this method achieves a ~1.5 speed up
    if(ThresImIn.empty()){
        cv::boxFilter( greyimage, thresImage, greyimage.type(), cv::Size(params.boxFilterSize, params.boxFilterSize),cv::Point(-1,-1), true, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED );
        thresImage=thresImage-greyimage;
        cv::threshold(thresImage, thresImage, params.thres, 255, cv::THRESH_BINARY);
    }
    else{
        thresImage=ThresImIn;
    }
    /////////////////// compute marker candidates by detecting contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> approxCurve;
    cv::RNG rand;
    int  minSizeSq=params.minSize*params.minSize,minSize4=4*params.minSize;
    contours=visitedAwareTracingContour(thresImage,minSize4,params.maxTimesRevisited);

    //decide where to store the candidates. If candidatesOut is not null, store there, otherwise use a local variable
    std::vector<Marker> candidateslocal;
    if(candidatesOut!=nullptr) {
        candidatesOut->clear();
    }
    else{
        candidatesOut=&candidateslocal;
    }
    ///////////////// for each contour, approx to a rectangle
    for (unsigned int i = 0; i < contours.size(); i++)
    {
        // can approximate to a convex rect?
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.03, true);
        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve)) continue;
        //check distance  between corners at least minSize pix
        if(  ((approxCurve[0].x-approxCurve[1].x)*(approxCurve[0].x-approxCurve[1].x) + (approxCurve[0].y-approxCurve[1].y)*(approxCurve[0].y-approxCurve[1].y))<minSizeSq) continue;
        if(  ((approxCurve[1].x-approxCurve[2].x)*(approxCurve[1].x-approxCurve[2].x) + (approxCurve[1].y-approxCurve[2].y)*(approxCurve[1].y-approxCurve[2].y))<minSizeSq) continue;
        if(  ((approxCurve[2].x-approxCurve[3].x)*(approxCurve[2].x-approxCurve[3].x) + (approxCurve[2].y-approxCurve[3].y)*(approxCurve[2].y-approxCurve[3].y))<minSizeSq) continue;
        if(  ((approxCurve[3].x-approxCurve[0].x)*(approxCurve[3].x-approxCurve[0].x) + (approxCurve[3].y-approxCurve[0].y)*(approxCurve[3].y-approxCurve[0].y))<minSizeSq) continue;
        // add the points
        Marker marker;marker.reserve(4);
        for (int j = 0; j < 4; j++)
            marker.emplace_back( cv::Point2f( approxCurve[j].x,approxCurve[j].y));
        //sort corner in clockwise direction
        marker=sort(marker);
        //check no point is too close to the border (which may cause error in corner subpix), leave at least 11 pixs
        bool discardBcTooCloseToBorder=false;
        for(auto p:marker){
            if(p.x<11 || p.y<11 || p.x>greyimage.cols-11 || p.y>greyimage.rows-11) {discardBcTooCloseToBorder=true;break;}
        }
        if(discardBcTooCloseToBorder) continue;
        candidatesOut->push_back(marker);
    }
    //now, for each candidate check bits inside
    int dictIndex=-1;
    for(const auto &dict:params.dicts){
        std::vector<Marker> currDirMarkerDetected;
        dictIndex++;
        cv::Mat bits(dict.markerSize+2,dict.markerSize+2,CV_8UC1),bitadaptive(dict.markerSize+2,dict.markerSize+2,CV_8UC1);

        for(auto it=candidatesOut->begin();it!=candidatesOut->end();){
            auto marker=*it;

            ////// extract the code. Obtain the intensities of the bits using  homography
            for(int i=0;i<int(params.maxAttemptsPerCandidate) && marker.id==-1;i++){
                //if not first attempt, we may wanna produce small random alteration of the corners
                auto marker2=marker;
                if( i!=0) for(int c=0;c<4;c++) {marker2[c].x+=rand.gaussian(0.75);marker2[c].y+=rand.gaussian(0.75);}//if not first, alter corner location
                Homographer hom(marker2);
                for(int r=0;r<bits.rows;r++){
                    for(int c=0;c<bits.cols;c++){
                        bits.at<uchar>(r,c)=uchar(0.5+getSubpixelValue(greyimage,hom(cv::Point2f(  float(c+0.5) / float(bits.cols) ,  float(r+0.5) / float(bits.rows)  ))));
                    }
                }
                if(i==2){ // if not working the first time, try this time adaptive threshold into the bits to improve robustness to lighting
                    thres255Adaptive(bits,bitadaptive);
                    bitadaptive.copyTo(bits);
                }
                else{
                    cv::threshold(bits,bits,0,255,cv::THRESH_OTSU);
                }
                //now, analyze the inner code to see it if is a marker. If so, rotate to have the points properly sorted
                int nrotations=0;
                if(getMarkerId(bits,marker.id,nrotations,params,dictIndex)==0) continue;
                std::rotate(marker.begin(),marker.begin() + 4 - nrotations,marker.end());
            }
            if(marker.id!=-1) {
                marker.dict=dictIndex;
                currDirMarkerDetected.push_back(marker);
                //remove from candidate list
                it=candidatesOut->erase(it);
            }
            else it++;//go to next
        }

        /// REMOVAL OF INNER DUPLICATED DETECTIONS OF THE SAME MARKER(INNER AND OUTER BORDER)
        std::sort(currDirMarkerDetected.begin(), currDirMarkerDetected.end(),[](const Marker &a,const Marker &b){return a.id<b.id;});
        {
            std::vector<bool> toRemove(currDirMarkerDetected.size(), false);
            for (int i = 0; i < int(currDirMarkerDetected.size()) - 1; i++)
            {
                for (int j = i + 1; j < int(currDirMarkerDetected.size()) && !toRemove[i]; j++)
                {
                    if (currDirMarkerDetected[i].id == currDirMarkerDetected[j].id )
                    {
                        auto res=isInto(currDirMarkerDetected[i],currDirMarkerDetected[j]);
                        if( res==1)toRemove[i]=true;
                        else if( res==2)toRemove[j]=true;

                    }
                }
            }
            //now move to DetectedMarkers these not marked for removal
            for (unsigned int i = 0; i < currDirMarkerDetected.size(); i++)
                if (!toRemove[i]) DetectedMarkers.push_back(currDirMarkerDetected[i]);
        }
    }

    //we want to remove the markers that are detected only with inner border, and thus are wrong
    //for that, we will analyze points on both sides along the line between the corners, and remove these for which the
    // color diff between them is below a threshold
    for(size_t i=0;i<DetectedMarkers.size();i++){
        if( isAInnerMarker(greyimage,DetectedMarkers[i])){
            //swap it with the last one (if not this) and resize
            if(i!=DetectedMarkers.size()-1) std::swap(DetectedMarkers[i],DetectedMarkers.back());
            DetectedMarkers.pop_back();
            i--;//stay in the same index to analyze the swapped one

        }
    }
    //no corner subpixel here
    return DetectedMarkers;//DONE
}
/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
int MarkerDetector:: getMarkerId(cv::Mat candidateBits, int &idx, int &nrotations, const DetectorParameters &params,int dictIndex){
    uint8_t typ=1;

    if(params.detectInvertedMarker ) candidateBits=~candidateBits;
    // analyze border bits
    int maximumErrorsInBorder =int(params.dicts[dictIndex].markerSize * params.dicts[dictIndex].markerSize * params.maxErroneousBitsInBorderRate);
    int borderErrors =getBorderErrors(candidateBits, params.dicts[dictIndex].markerSize, params.markerBorderBits);
    if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong
    // take only inner bits
    int borderBits = static_cast<int>(params.markerBorderBits);
    cv::Mat onlyBits = candidateBits.rowRange(borderBits, candidateBits.rows - borderBits).colRange(borderBits, candidateBits.cols - borderBits);
    onlyBits/=255;
    // try to indentify the marker
    if(!params.dicts[dictIndex].identify(onlyBits, idx, nrotations, params.errorCorrectionRate))
        return 0;
    return typ;
}
/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
int MarkerDetector::getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) {
    int sizeWithBorders = markerSize + 2 * borderSize;
    int totalErrors = 0;
    for(int y = 0; y < sizeWithBorders; y++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(y)[k] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(y)[sizeWithBorders - 1 - k] != 0) totalErrors++;
        }
    }
    for(int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(k)[x] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(sizeWithBorders - 1 - k)[x] != 0) totalErrors++;
        }
    }
    return totalErrors;
}
float MarkerDetector::getSubpixelValue(const cv::Mat &im_grey, const cv::Point2f &p) {
    // 1. Get integer coordinates
    const int ix = static_cast<int>(p.x);
    const int iy = static_cast<int>(p.y);


    //   Boundary Check: Ensure the 2x2 patch is within limits
    // We check ix+1 and iy+1 because the interpolation looks at the next pixel over.
    if (ix < 0 || iy < 0 || ix >= im_grey.cols - 1 || iy >= im_grey.rows - 1) {
        // Option A: Return a default value
        // Option B: Clamp the point to the nearest valid boundary
        return 0.0f;
    }

    // 2. Get fractional parts
    const float dx = p.x - ix;
    const float dy = p.y - iy;
    // 3. Optimized Pointer Access
    const uchar* ptr = im_grey.ptr<uchar>(iy) + ix;
    const size_t step = im_grey.step;
    // 4. Fetch the four pixels immediately as floats
    const float p00 = static_cast<float>(ptr[0]);        // Top-Left
    const float p01 = static_cast<float>(ptr[1]);        // Top-Right
    const float p10 = static_cast<float>(ptr[step]);     // Bottom-Left
    const float p11 = static_cast<float>(ptr[step + 1]); // Bottom-Right
    // 5. Separable Interpolation (3 Multiplications total)
    const float top = p00 + dx * (p01 - p00);// Interpolate Top Row Horizontally
    const float bot = p10 + dx * (p11 - p10);    // Interpolate Bottom Row Horizontallys
    // Interpolate Vertically between Top and Bottom results
    return top + dy * (bot - top);
}
Marker  MarkerDetector::sort( const  Marker &marker){
    Marker res_marker=marker;
    /// sort the points in anti-clockwise order
    double dx1 = res_marker[1].x - res_marker[0].x;
    double dy1 = res_marker[1].y - res_marker[0].y;
    double dx2 = res_marker[2].x - res_marker[0].x;
    double dy2 = res_marker[2].y - res_marker[0].y;
    double o = (dx1 * dy2) - (dy1 * dx2);
    // if the third point is in the left side, then sort in anti-clockwise order
    if (o < 0.0)  std::swap(res_marker[1], res_marker[3]);
    return res_marker;
}
/**
 * @brief Traces the contours of a binary image using our visited aware Tracing algorithm.
 *
 * This function scans a binary image (foreground as 255, background as 0) and
 * finds the external boundaries of all distinct objects.
 */
std::vector<std::vector<cv::Point>> MarkerDetector::visitedAwareTracingContour(cv::Mat &padded, size_t minSize, float maxRevisited ) {
    if (padded.empty() || padded.type() != CV_8UC1) return {};
    // 1. Fast Initialization and Padding
    int rows = padded.rows;
    int cols = padded.cols;
    int32_t step = padded.step;
    uchar* data = padded.data;
    // Fast clear of top and bottom rows
    memset(data, 0, cols);
    memset(data + (rows - 1) * step, 0, cols);
    // Fast clear of left and right columns
    for (int r = 1; r < rows - 1; ++r) {
        uchar* row_ptr = data + r * step;
        row_ptr[0] = 0;
        row_ptr[cols - 1] = 0;
    }
    // 2. Precompute Neighbor Offsets based on image stride This removes the need for Point arithmetic in the loop
    const int offsets[16]={-1,-step-1,-step,-step+1,1,step+1,step,step-1, -1,-step-1,-step,-step+1,1,step+1,step,step-1, };
    // Use static tables to avoid initialization overhead on every call // 8-connectivity offsets relative to center (0,0) // Order: W, NW, N, NE, E, SE, S, SW
    const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 }, dy[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };
    // Pre-allocate results
    std::vector<std::vector<cv::Point>> contours;contours.reserve(2048);
    std::vector<cv::Point> buffer;buffer.reserve(2048);
    const uchar FOREGROUND = 255, BACKGROUND = 0,VISITED = 100;
    // 3. Scanning Loop
    // We iterate using raw pointers for maximum speed
    int rowStep=1;//std::max(1,int(minSize/6));
    for (int r = 1; r < rows - 1; r+=rowStep) {
        uchar* row_ptr = data + r * step;
        for (int c = 1; c < cols - 1;  ) {
            ////findStartContourPoint
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                cv::v_uint8 v_zero = cv::vx_setzero_u8();
                for (; c <= cols - cv::VTraits<cv::v_uint8>::vlanes(); c+= cv::VTraits<cv::v_uint8>::vlanes())
                {
                    cv::v_uint8 vmask = (cv::v_ne(cv::vx_load((uchar*)(row_ptr + c)), v_zero));
                    if (v_check_any(vmask))
                    {
                        c += v_scan_forward(vmask);
                        break;
                    }
                }
#endif
                //process last tail
                for (; c < cols && !row_ptr[c]; ++c) ;//last tail
            }
            if( c==cols) break;//reached end of row
            if (row_ptr[c] == FOREGROUND ) {// --- 4. Tracing Loop  if is foreground
                buffer.clear();
                int curr_x = c, curr_y = r,search_idx = 1 ;
                uchar* curr_ptr = row_ptr + c,*start_ptr=curr_ptr;
                size_t ntimesRevisited=0;
                do {
                    buffer.emplace_back(curr_x, curr_y);// Add point
                    *curr_ptr = VISITED;// Mark as visited
                    //showImage(padded);
                    // Search for next foreground pixel. We search 8 neighbors starting from search_idx
                    for (int i = 0; i < 8; ++i) {
                        int idx = search_idx + i; // index into offsets (0..15)
                        uchar* neighbor = curr_ptr + offsets[idx]; // Fast pointer arithmetic
                        if (*neighbor != BACKGROUND) {
                            // Found next boundary pixel
                            curr_ptr = neighbor;
                            int dir = (idx & 7);                             // Update Integer Coordinates using the small static tables(Use modulo 8 to get the distinct direction 0-7)
                            int next_x=curr_x+dx[dir], next_y=curr_y+dy[dir];
                            ntimesRevisited+= int(*neighbor == VISITED);
                            curr_x = next_x;curr_y = next_y;
                            search_idx = (dir + 5) & 7;
                            break;
                        }
                    }
                } while (curr_ptr != start_ptr );
                size_t bufsize=buffer.size();
                if (ntimesRevisited<= float(bufsize)*maxRevisited && bufsize >= minSize) {
                    contours.push_back(buffer);
                }
            }
            c++;//move to next pixel
            ////findEndContourPoint
            if ( row_ptr[c]){
                {
#if (CV_SIMD || CV_SIMD_SCALABLE)

                    cv::v_uint8 v_zero = cv::vx_setzero_u8();
                    for (; c <=  cols - cv::VTraits<cv::v_uint8>::vlanes(); c += cv::VTraits<cv::v_uint8>::vlanes())
                    {
                        cv::v_uint8 vmask = (cv::v_eq(cv::vx_load((uchar*)(row_ptr + c)), v_zero));
                        if (cv::v_check_any(vmask))
                        {
                            c += cv::v_scan_forward(vmask);
                            break;
                        }
                    }
#endif
                }
                for (; c < cols && row_ptr[c]; ++c) ;//last tail
            }
        }
    }
    return contours;
}
bool MarkerDetector::isAInnerMarker(const cv::Mat &grey,const Marker &marker)
{
    // --- tunable parameters ---
    const int   nSamplesPerEdge = 5;     // pairs along each edge
    const float perpOffset      = 2.0f;  // pixels perpendicular to the edge
    const float colorThreshold  = 30.0f; // mean |L-R| below this => inner marker
    // --------------------------

    // NOTE: replace `grey` with whatever member of MarkerDetector
    // holds the grayscale input image (e.g. _grey, grayImage, ...).
    const cv::Mat &img = grey;
    if (img.empty() || img.type() != CV_8UC1) return false;


    float totalDiff   = 0.f;
    int   validPairs  = 0;

    for (int i = 0; i < 4; ++i) {
        const cv::Point2f &p1 = marker[i];
        const cv::Point2f &p2 = marker[(i + 1) % 4];

        cv::Point2f edge = p2 - p1;
        float edgeLen = cv::norm(edge);
        if (edgeLen < 1e-6f) continue;

        cv::Point2f edgeDir = edge / edgeLen;
        // perpendicular (rotate 90°): one side will be "outside",
        // the other "inside" the polygon — we don't actually need
        // to know which is which, only the magnitude of the difference.
        cv::Point2f perp(-edgeDir.y, edgeDir.x);

        for (int j = 1; j <= nSamplesPerEdge; ++j) {
            // t in (0,1) — skip exact corners to avoid noise there
            float t = float(j) / float(nSamplesPerEdge + 1);
            cv::Point2f midPt = p1 + t * edge;

            cv::Point2f leftPt  = midPt + perp * perpOffset;
            cv::Point2f rightPt = midPt - perp * perpOffset;

            float cL = getSubpixelValue(grey,leftPt);
            float cR = getSubpixelValue(grey,rightPt);
            if (cL < 0.f || cR < 0.f) continue; // out of image

            totalDiff += std::fabs(cL - cR);
            ++validPairs;
        }
    }

    if (validPairs == 0) return false;

    float avgDiff = totalDiff / validPairs;
    return avgDiff < colorThreshold; // low contrast across the line => inner marker
}


void MarkerDetector::thres255Adaptive(cv::Mat &in,cv::Mat &out,int off,int thres){
    cv::boxFilter( in, out, in.type(), cv::Size(off*2+1, off*2+1),
                  cv::Point(-1,-1), true, 4 );

    for(int i = 0; i < in.rows; i++ )
    {
        const uchar* sdata = in.ptr(i);
        uchar* ddata = out.ptr(i);
        for(int j = 0; j < in.cols; j++ )
            ddata[j] = ((int)ddata[j] - thres < (int)sdata[j]) ? 255 : 0;
    }

}
void copyVector2Output(std::vector<Marker> &vec, cv::OutputArrayOfArrays out )   {
    out.create((int)vec.size(), 1, CV_32FC2);
    if(out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            cv::Mat &m = out.getMatRef(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if(out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            cv::UMat &m = out.getUMatRef(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if(out.kind() == cv::_OutputArray::STD_VECTOR_VECTOR){
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            cv::Mat m = out.getMat(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}


void copyVector2Output(std::vector<std::vector<cv::Point2f> > &vec, cv::OutputArrayOfArrays out )   {
    out.create((int)vec.size(), 1, CV_32FC2);
    if(out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create((int)vec[i].size(), 1, CV_32FC2, i);
            cv::Mat &m = out.getMatRef(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if(out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create((int)vec[i].size(), 1, CV_32FC2, i);
            cv::UMat &m = out.getUMatRef(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if(out.kind() == cv::_OutputArray::STD_VECTOR_VECTOR){
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(vec[i].size(), 1, CV_32FC2, i);
            cv::Mat m = out.getMat(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}

std::vector<Marker> detect(cv::aruco::Dictionary dict, cv::Mat & src_gray,cv::Mat & thresImage,   int erosionIt){

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::erode(thresImage, thresImage, kernel,{-1,-1},erosionIt);
    DetectorParameters params;
    //params.expansionDueToErosion=erosionIt;
    params.dicts.push_back(dict);
    return MarkerDetector::detect(src_gray,params,nullptr,thresImage);


}
std::vector<std::vector<int>> getConnectedComponents(cv::Mat graph_8uc1) {
    int n = graph_8uc1.rows;
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> components;

    // Ensure the matrix is square
    if (n != graph_8uc1.cols) {
        return components;
    }

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            // Start of a new connected component
            std::vector<int> currentComponent;
            std::queue<int> q;

            q.push(i);
            visited[i] = true;

            while (!q.empty()) {
                int u = q.front();
                q.pop();
                currentComponent.push_back(u);

                // Check the row in the adjacency matrix for neighbors
                // Using ptr<uchar> for faster row access than .at<uchar>
                const uchar* rowPtr = graph_8uc1.ptr<uchar>(u);
                for (int v = 0; v < n; ++v) {
                    if (rowPtr[v] != 0 && !visited[v]) {
                        visited[v] = true;
                        q.push(v);
                    }
                }
            }
            components.push_back(currentComponent);
        }
    }

    return components;
}

std::shared_ptr<cv::flann::Index> buildFlannIndex(const std::vector<Marker> &markers) {
    //create a vector with all corners
    std::vector<cv::Point2f> corners;
    corners.reserve(markers.size()*4);
    for(const auto &m:markers){
        for(const auto &c:m){
            corners.push_back(c);
        }
    }
    //now, create a flann index with these corners
    cv::flann::KDTreeIndexParams indexParams(1);
    cv::Mat data = cv::Mat(corners).reshape(1, static_cast<int>(corners.size()));
    return std::make_shared<cv::flann::Index>(data, indexParams);
}
//returns a set of markers that are connected, i.e. that have at least one corner closer than a threshold.
std::vector<std::vector<Marker>> connectedComponents(const std::vector<Marker> &markers){
    //create a vector with all corners
    //now, create a flann index with these corners
    auto flannIndex=buildFlannIndex(markers);

    cv::Mat  graph(markers.size(), markers.size(), CV_8UC1, cv::Scalar(0));

    //now, for each corner, find the neighbors closer than a threshold
    const float threshold=10.0f;
    for(size_t m=0;m<markers.size();m++){
        for(int i=0;i<4;i++){
            std::vector<int> indices;
            std::vector<float> dists;
            cv::Mat query = (cv::Mat_<float>(1, 2) << markers[m][i].x, markers[m][i].y); // Single 2D query point
            int n=flannIndex->radiusSearch(query, indices, dists, threshold*threshold, 4*markers.size());
            for(int x=0;x<n;x++){
                int idx=indices[x];
                graph.at<uchar>(m,idx/4)=1;
                graph.at<uchar>(idx/4,m)=1;
            }
        }
    }

    //std::cout<<"Graph:\n"<<graph<<std::endl;
    //obtain the different connected components available
    auto ccomps=getConnectedComponents(graph);


    //now, for each component, obtain the corresponding markers
    std::vector<std::vector<Marker>> res;
    for(const auto &comp:ccomps){
        res.push_back({});
        for(auto idx:comp){
            res.back().push_back(markers[idx]);
        }
    }
    return res;
}


std::vector<Marker> detectBWMarkers(const cv::aruco::CharucoBoard2 &board,cv::Mat &src_gray){


     std::vector<Marker>  markers_black,markers_white;
    cv::Mat thresImage;

    //BLACK MARKERS
    cv::boxFilter( src_gray, thresImage, src_gray.type(), cv::Size(25,25),cv::Point(-1,-1), true, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED );
    thresImage=thresImage-src_gray;
    cv::threshold(thresImage, thresImage, 3, 255, cv::THRESH_BINARY);
    //determine how many erosion iterations we will do, depending on the size of the image
    int maxErodeIterations= std::max(2, int( (2.*src_gray.cols/2000.)+0.5));
    std::vector<std::vector<Marker>  > markers_blackv(maxErodeIterations);
    cv::Range range(1, maxErodeIterations);

   cv::parallel_for_(range, [&](const cv::Range& r) {
       for(int i=r.start;i<r.end;i++){
         cv::Mat thres=thresImage.clone();
            markers_blackv[i]=detect(board.dictionary,src_gray,thres,i);//black markers
            //because we have shrink the borders for black markers, we will expand the corners a bit from the center
            for(auto & marker:markers_blackv[i]){
                std::vector<cv::Point2f> newPoints;
                for(int j=0;j<4;j++){
                    int idx0=j;
                    int idx1=(j+2) % 4;
                    auto dif=marker[idx0]-marker[idx1];
                    auto norm=cv::norm(dif);
                    auto p= marker[idx1]+  ((dif)/norm)*(norm+ 2+i   );
                    newPoints.push_back(p);
                }
                //replace the points by the new ones
                for(int j=0;j<4;j++){
                    marker[j]=newPoints[j];
                }
            }
        }
    });

    //merge them all into markers_black
    for(const auto &v:markers_blackv){
        for(const auto &m:v){
            markers_black.push_back(m);
        }
    }
    //now, we need to merge the different markers
    if(markers_blackv.size()>1){

        std::vector<Marker> unduplicated;
        /// REMOVAL OF INNER DUPLICATED DETECTIONS OF THE SAME MARKER(INNER AND OUTER BORDER)
        std::sort(markers_black.begin(), markers_black.end(),[](const Marker &a,const Marker &b){return a.id<b.id;});
        std::vector<bool> toRemove(markers_black.size(), false);
        for (int i = 0; i < int(markers_black.size()) - 1; i++)
        {
            for (int j = i + 1; j < int(markers_black.size()) && !toRemove[i]; j++)
            {
                if (markers_black[i].id == markers_black[j].id )
                {
                    if( markers_black[i].cornerMaxDistance(markers_black[j])<100)
                        toRemove[i]=true;
                }
            }
        }
        //now move to DetectedMarkers these not marked for removal
        for (unsigned int i = 0; i < markers_black.size(); i++)
            if (!toRemove[i]) unduplicated.push_back(markers_black[i]);
        markers_black=unduplicated;
    }


//    markers_black=detect(board.dictionary,src_gray,thresImage,erodeIterations);//black markers





    ///WHITE MARKERS
    cv::boxFilter( src_gray, thresImage, src_gray.type(), cv::Size(15,15),cv::Point(-1,-1), true, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED );
    thresImage=thresImage-src_gray;
    cv::threshold(thresImage, thresImage, 3, 255, cv::THRESH_BINARY);


    //draw a line between opposite corners to break the continous contour of the white markers, which will help to detect them.
    for(auto m:markers_black){
        auto n02=m[2]-m[0];
        auto n02norm=cv::norm(n02);
        auto p0=m[0]-(n02/n02norm)* (n02norm/8);
        auto p1=m[0]+(n02/n02norm)* (n02norm/8);
        cv::line(thresImage,p0,p1,cv::Scalar::all(255),2);
        p0=m[2]-(n02/n02norm)* (n02norm/8);
        p1=m[2]+(n02/n02norm)* (n02norm/8);
        cv::line(thresImage,p0,p1,cv::Scalar::all(255),2);

        //same for the other diagonal
        auto n13=m[3]-m[1];
        auto n13norm=cv::norm(n13);
        p0=m[1]-(n13/n13norm)* (n13norm/8);
        p1=m[1]+(n13/n13norm)* (n13norm/8);
        cv::line(thresImage,p0,p1,cv::Scalar::all(255),2);
        p0=m[3]-(n13/n13norm)* (n13norm/8);
        p1=m[3]+(n13/n13norm)* (n13norm/8);
        cv::line(thresImage,p0,p1,cv::Scalar::all(255),2);
    }
    thresImage=255-thresImage;
    cv::Mat src_gray_inv = 255 - src_gray;
    markers_white = detect(board.dictionary, src_gray_inv, thresImage, 1); // white markers


    // Combine results from both
    std::vector<Marker> allMarkers;
    allMarkers.reserve(markers_black.size() + markers_white.size());
    allMarkers.insert(allMarkers.end(), markers_black.begin(), markers_black.end());
    allMarkers.insert(allMarkers.end(), markers_white.begin(), markers_white.end());





      return allMarkers;
}


//given a marker id and one of its corners, return the global corner id of that corner, which is a unique id for that corner in the whole board,

    int getGlobalCornerID(int marker_id, int corner_id,const  cv::aruco::CharucoBoard2 &Board)
{
    //obtain the row, col of the marker_id
    auto row_col=Board.getIdPos(marker_id);
    if(corner_id<=1){
        return (Board.bSize.width+1) *row_col.first +  row_col.second+  corner_id;
    }
    else if(corner_id==2){
        return (Board.bSize.width+1) *(row_col.first+1) +  row_col.second+ 1;
    }
    else  {
        return (Board.bSize.width+1) *(row_col.first+1) +  row_col.second;

    }
}
//opposite of getGlobalCornerID, given a global corner id, return the marker ids and corner ids of that corner

std::vector<std::pair<int,int>> getMarkerCornersFromGlobalCornerID( int gid,const  cv::aruco::CharucoBoard2 &Board)
{
    std::vector<std::pair<int,int>> result;
    const int W = Board.bSize.width;

    // Decompose gid into (cr, cc) in the (H+1) x (W+1) corner grid.
    // Inverse of  gid = (W+1) * cr + cc  used in getGlobalCornerID.
    const int cr = gid / (W + 1);
    const int cc = gid % (W + 1);

    // Up to 4 markers share a global corner. Sort order: 0=TL, 1=TR, 2=BR, 3=BL.
    // Board.getId() returns -1 for out-of-range (row,col), so it doubles as a bounds check.
    int id;

    // Marker at (cr, cc) sees this point as its top-left (0)
    id = Board.getId(cr,     cc);
    if (id != -1) result.emplace_back(id, 0);

    // Marker at (cr, cc-1) sees this point as its top-right (1)
    id = Board.getId(cr,     cc - 1);
    if (id != -1) result.emplace_back(id, 1);

    // Marker at (cr-1, cc-1) sees this point as its bottom-right (2)
    id = Board.getId(cr - 1, cc - 1);
    if (id != -1) result.emplace_back(id, 2);

    // Marker at (cr-1, cc) sees this point as its bottom-left (3)
    id = Board.getId(cr - 1, cc);
    if (id != -1) result.emplace_back(id, 3);

    return result;
}

}


cv::aruco::CharucoBoard2::CharucoBoard2()
{

}

cv::aruco::CharucoBoard2::CharucoBoard2(cv::Size _bSize, float _markerLength, float _markerSeparation,  const  cv::aruco::Dictionary  &_dictionary, cv::InputArray _ids)
{
    (void)_markerSeparation;
    this->bSize = _bSize;
    this->dictionary = _dictionary;
    this->markerLength = _markerLength;
    this->markerSeparation = 0;
    //set the ids
    if(_ids.empty()){
        int markerId=0;
        for(int y=0;y<_bSize.height;y++){
            for(int x= 0;x<_bSize.width;x++,markerId++){
                if(markerId>=_dictionary.bytesList.rows){
                    CV_Error(cv::Error::StsBadArg, "Number of markers exceeds the number of markers in the dictionary");
                }
                this->ids.push_back(markerId);
            }
        }
    }
    else{
        cv::Mat idsMat=_ids.getMat();
        if(idsMat.total()!=size_t(_bSize.area()) || idsMat.type()!=CV_32SC1){
            CV_Error(cv::Error::StsBadArg, "Ids must be a vector of int with the same number of elements as the board size");
        }
        this->ids=std::vector<int>(idsMat.begin<int>(),idsMat.end<int>());
    }
}

void cv::aruco::CharucoBoard2::generateImage(int markerSizePix, cv::Mat &outImage) const
{
    int border=markerSizePix/4;
    int nmarkers=bSize.area();
    if(nmarkers > dictionary.bytesList.rows)
        CV_Error(cv::Error::StsBadArg, "Number of markers exceeds the number of markers in the dictionary");

    cv::Size imgSize(markerSizePix*bSize.width + 2*border , markerSizePix*bSize.height+2*border);
    int markerIdx=0;
    outImage = cv::Mat::zeros(imgSize, CV_8UC1);
    outImage=255;
    int startLineColor = 0;
    for(int y=0; y<bSize.height; y++){
        int curMarkerColor=startLineColor;
        for(int x= 0; x<bSize.width; x++,markerIdx++){
            cv::Mat markerImg;
            dictionary.generateImageMarker(ids[markerIdx], markerSizePix, markerImg);
            int posX = x * markerSizePix + border;
            int posY = y * markerSizePix + border;
            if(curMarkerColor)
                markerImg=255-markerImg;
            markerImg.copyTo(outImage(cv::Rect(posX, posY, markerSizePix, markerSizePix)));
            curMarkerColor= curMarkerColor==1?0:1;
        }
        startLineColor=startLineColor==1?0:1;
    }

    for(int x=1;x<bSize.width;x+=2)
        cv::rectangle(outImage,cv::Rect(border+x*markerSizePix,0,markerSizePix,border),cv::Scalar::all(0),cv::FILLED);
    for(int x=bSize.height%2;x<bSize.width;x+=2)
        cv::rectangle(outImage,cv::Rect(border+x*markerSizePix,border+markerSizePix*bSize.height,markerSizePix,border),cv::Scalar::all(0),cv::FILLED);
    for(int y=1;y<bSize.height;y+=2)
        cv::rectangle(outImage,cv::Rect(0,border+y*markerSizePix,border,markerSizePix),cv::Scalar::all(0),cv::FILLED);
    for(int y=bSize.width%2;y<bSize.height;y+=2)
        cv::rectangle(outImage,cv::Rect(border+markerSizePix*bSize.width,border+y*markerSizePix,border,markerSizePix),cv::Scalar::all(0),cv::FILLED);

    cv::rectangle(outImage,cv::Rect(0,0,border,border),cv::Scalar::all(0),cv::FILLED);
    cv::rectangle(outImage,cv::Rect(outImage.cols-border,0,border,border),cv::Scalar::all(0),cv::FILLED);
    cv::rectangle(outImage,cv::Rect(0,outImage.rows-border,border,border),cv::Scalar::all(0),cv::FILLED);
    cv::rectangle(outImage,cv::Rect(outImage.cols-border,outImage.rows-border,border,border),cv::Scalar::all(0),cv::FILLED);
}

void cv::aruco::CharucoBoard2::generateImage(cv::Size outSize, cv::Mat &outImage, int marginSize, int borderBits) const
{
    (void)borderBits; // kept for API compatibility
    int markerSizePix = std::min((outSize.width  - 2*marginSize) / bSize.width,
                                  (outSize.height - 2*marginSize) / bSize.height);
    if (markerSizePix < 1)
        CV_Error(cv::Error::StsBadArg, "Output image size too small for the board dimensions");

    cv::Mat boardImg;
    generateImage(markerSizePix, boardImg);

    // scale to fill the available area while preserving the board's aspect ratio
    int availW = outSize.width  - 2*marginSize;
    int availH = outSize.height - 2*marginSize;
    float scale = std::min((float)availW / boardImg.cols, (float)availH / boardImg.rows);
    cv::Size scaledSize(std::round(boardImg.cols * scale), std::round(boardImg.rows * scale));
    int interp = (scale < 1.0f) ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(boardImg, boardImg, scaledSize, 0, 0, interp);

    outImage = cv::Mat(outSize, CV_8UC1, cv::Scalar::all(255));
    int offsetX = (outSize.width  - boardImg.cols) / 2;
    int offsetY = (outSize.height - boardImg.rows) / 2;
    boardImg.copyTo(outImage(cv::Rect(offsetX, offsetY, boardImg.cols, boardImg.rows)));
}


std::pair<int, int> cv::aruco::CharucoBoard2::getIdPos(int id) const
{
    auto it=std::find(ids.begin(),ids.end(),id);
    if(it==ids.end()) return {-1,-1};
    int idx=std::distance(ids.begin(),it);
    int row = idx / bSize.width;
    int col = idx % bSize.width;
    return {row, col};
}

int cv::aruco::CharucoBoard2::getId(int row, int col) const
{
    if(row<0 || row>=bSize.height || col<0 || col>=bSize.width) return -1;
    int idx=row*bSize.width+col;
    return ids[idx];
}

void cv::aruco::CharucoBoard2::matchImagePoints(cv::InputArrayOfArrays detectedCorners, cv::InputArray detectedIds, cv::OutputArray objPoints, cv::OutputArray imgPoints) const
{

    CV_Assert(detectedIds.total() > 0ull);
    CV_Assert(detectedIds.total() == detectedCorners.total() || (detectedIds.total() == 4 && detectedCorners.total()==9 && bSize.width==2 && bSize.height==2));

    size_t nDetectedMarkers = detectedIds.total();

    std::vector<cv::Point3f> objPnts;
    objPnts.reserve(nDetectedMarkers);

    std::vector<cv::Point2f> imgPnts;
    imgPnts.reserve(nDetectedMarkers);

    cv::Mat detectedIdsMat = detectedIds.getMat();
    cv::Mat detectedCornersMat;
    std::vector<cv::Mat> detectedCornersVecMat;

    if (detectedCorners.isMatVector()) {
        detectedCorners.getMatVector(detectedCornersVecMat);
    } else {
        detectedCornersMat = detectedCorners.getMat();
    }

    //its a diamond?
    if(detectedIds.total() == 4 && detectedCorners.total()==9 && bSize.width==2 && bSize.height==2){
        //points are given in order, we do not need the detectedIds
        int idx=0;
        for(int row=0;row<3;row++){
            for(int col=0;col<3;col++,idx++){
                cv::Point3f p3d;
                p3d.x = col * markerLength;
                p3d.y = row * markerLength;
                p3d.z = 0;
                objPnts.push_back(p3d);
                if (detectedCorners.isMatVector()) {
                    imgPnts.push_back(detectedCornersVecMat[idx].ptr<cv::Point2f>(0)[0]);
                } else {
                    imgPnts.push_back(detectedCornersMat.ptr<cv::Point2f>(0)[idx]);
                }
            }
        }
    }
    //its a chessboard
    else{

        for(unsigned int i = 0; i < detectedIdsMat.total(); i++) {
            if (detectedCorners.isMatVector()) {
                imgPnts.push_back(detectedCornersVecMat[i].ptr<cv::Point2f>(0)[0]);
            } else {
                imgPnts.push_back(detectedCornersMat.ptr<cv::Point2f>(0)[i]);
            }
            int currentId = detectedIdsMat.at<int>(i);
            int row = currentId / (bSize.width+1);
            int col = currentId % (bSize.width+1);
            cv::Point3f p3d;
            p3d.x = col * markerLength;
            p3d.y = row * markerLength;
            p3d.z = 0;
            objPnts.push_back(p3d);
        }

    }


    // create output
    cv::Mat(objPnts).copyTo(objPoints);
    cv::Mat(imgPnts).copyTo(imgPoints);

}
cv::aruco::CharucoDetector2::CharucoDetector2(const CharucoBoard2 &_board)
{
this->board=_board;

}

cv::aruco::CharucoDetector2::CharucoDetector2()
{

}
void cv::aruco::CharucoDetector2::setBoard(const CharucoBoard2 &_board){
    this->board=_board;
}
void cv::aruco::CharucoDetector2::detectBoard(cv::InputArray image, cv::OutputArray charucoCorners, cv::OutputArray charucoIds,
                                              cv::InputOutputArrayOfArrays markerCorners, cv::InputOutputArray markerIds)
{
    CV_Assert(board.bSize.width > 0 && board.bSize.height > 0 && board.markerLength > 0.f);
    CV_Assert(image.channels() == 1 || image.channels() == 3);
    cv::Mat src_gray;
    //obtain the gray image
    if(image.channels()==3)
        cvtColor(image, src_gray, cv::COLOR_BGR2GRAY);
    else src_gray=image.getMat();

    //detect all markers
    auto allMarkers=detectBWMarkers(board, src_gray);
    //remove markers not belonging to the list of ids of the board
    allMarkers.erase(std::remove_if(allMarkers.begin(), allMarkers.end(),
                                    [this](const Marker &m) {
                                        return std::find(board.ids.begin(), board.ids.end(), m.id) == board.ids.end();
                                    }), allMarkers.end());


    if(allMarkers.empty())return;
    //obtain the connected components
    std::vector<std::vector<Marker> > connected_markers=connectedComponents(allMarkers);

    //lets detect possible inconsistencies, i.e., markers in the wrong order, possibly belonging to another board configuration
    std::vector<std::vector<Marker> > consistent_connected_markers;
    for(auto &comp:connected_markers){
        int threshold=10;
        auto findex=buildFlannIndex(comp);
        bool is_consistent=true;
        //for each corner, of each marker we will analyze its nearst neighbor. If is really connected, we will check if
        //the connection is consistent
        for(auto marker:comp){
            std::vector<int> indices;
            std::vector<float> dists;
             for(int c=0;c<4;c++){
                cv::Mat query = (cv::Mat_<float>(1, 2) << marker[c].x, marker[c].y); // Single 2D query point
                int nn=findex->radiusSearch(query, indices, dists, threshold*threshold, marker.size());
                for(int ix=0;ix<std::min(nn,int(indices.size()));ix++){
                    int idx=indices[ix];
                     if(comp[idx/4].id==marker.id) continue;//same marker
                    //check if the connection is consistent, i.e., if the global corner ids are the same
                    int gid1=getGlobalCornerID(marker.id,c,board);
                    int gid2=getGlobalCornerID(comp[idx/4].id,idx%4,board);
                    if(gid1!=gid2){
                        CV_LOG_WARNING(NULL, "Marker " << marker.id << " corner " << c << " connected to marker " << comp[idx/4].id << " corner " << (idx%4) << " but global corner ids differ: " << gid1 << " vs " << gid2);
                        is_consistent=false;
                    }
                }
            }
            if(!is_consistent) break;
        }
        if(is_consistent)
            consistent_connected_markers.push_back(comp);
    }

    if(consistent_connected_markers.empty())return;

    //select the one with more markers belonging to the board (at least 2)
    std::vector<Marker> connected_markers_filtered;
     for(const auto &comp:consistent_connected_markers){
        size_t validMarkers=0;
        for(const auto &m:comp){
            if(std::find(board.ids.begin(),board.ids.end(),m.id)!=board.ids.end()){
                validMarkers++;
            }
        }
        if(validMarkers>connected_markers_filtered.size() && validMarkers>=2){
            connected_markers_filtered=comp;
        }
     }


     if(connected_markers_filtered.empty()) return;
     allMarkers=connected_markers_filtered;

    //now, lets focus on the global corners. Unify the corners for subpixel analysis
    //first, obtain the average position for each global corner id found
    std::map<int,std::pair<cv::Point2f,int> > global_corners;
    for(auto m:allMarkers){
        for(int c=0;c<4;c++){
            int gid=getGlobalCornerID(m.id,c,board);
            if (global_corners.find(gid)==global_corners.end())
                global_corners[gid]={m[c],1};
            else{
                auto &p=global_corners[gid];
                p.first+=m[c];
                p.second++;
            }
        }
    }
    //now, average
    for(auto &gc:global_corners)
        gc.second.first=gc.second.first / float(gc.second.second);

    //////  subpixel corner refinement
    // compute the window size for the subpixel refinement based on the size of the markers. Bigger makers, bigger search zone
    double avrgLen=0;
    for(auto m:allMarkers){
        for(int i=0;i<4;i++){
            avrgLen+=cv::norm(m[i]-m[(i+1)%4]);
        }
    }
    avrgLen/=4*allMarkers.size();
    //here is the formula to compute the half window size
    int halfwsize= std::min(int(3* std::max(1.f,float(avrgLen)/float(34) )),9);

    //add the global corners to a single vector
    std::vector<cv::Point2f> Corners;
    for (const auto &m:global_corners)
        Corners.push_back(m.second.first);
    //refine the corners
    cv::cornerSubPix(src_gray, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
    // copy back to the global corners
    int idex=0;
    for(auto &gc:global_corners){
        gc.second.first=Corners[idex++];
    }

    //now, assign the refined global corner positions back to the markers
    for(auto it_gc = global_corners.begin(); it_gc != global_corners.end(); ++it_gc){
        int gid = it_gc->first;
        cv::Point2f corner_c = it_gc->second.first;
        //find in how many markers it is involved
        auto marker_corners = getMarkerCornersFromGlobalCornerID(gid,board);
        for(auto it_mc = marker_corners.begin(); it_mc != marker_corners.end(); ++it_mc){
            int markerid = it_mc->first;
            int cornerid = it_mc->second;
            //see if the makrer id is detected
            auto markerid_copy=markerid;//capturing markerid only in C++20, lets use a copy for prev compilers
            auto it=std::find_if(allMarkers.begin(),allMarkers.end(),[markerid_copy](const Marker &m){return m.id==markerid_copy;});
            if(it!=allMarkers.end()){
                (*it)[cornerid]=corner_c;
            }
        }
    }


    //copy results to output arrays
    std::vector<int> gidv;
    std::vector<cv::Point2f> gcorners;
    for(auto it_gc = global_corners.begin(); it_gc != global_corners.end(); ++it_gc){
        gidv.push_back(it_gc->first);
        gcorners.push_back(it_gc->second.first);
    }
    if (charucoIds.needed()) {
        cv::Mat(gidv).copyTo(charucoIds);
    }

    if (charucoCorners.needed()) {
        cv::Mat(gcorners).copyTo(charucoCorners);
    }

    //Markers
    // 2. Unpack results into OutputArrayOfArrays
    if(markerCorners.needed())
        copyVector2Output(allMarkers, markerCorners);
    if( markerIds.needed()){
        // 3. Assign to output ids
        std::vector<int> marker_idsVec;
        marker_idsVec.reserve(allMarkers.size());
        for (const auto& m : allMarkers) marker_idsVec.push_back(m.id);
        // Allocate and copy IDs
        markerIds.create((int)marker_idsVec.size(), 1, CV_32SC1);
        cv::Mat(marker_idsVec).copyTo(markerIds);
    }
}
void cv::aruco::CharucoDetector2::detectDiamonds(cv::InputArray image, cv::OutputArrayOfArrays _diamondCorners, cv::OutputArray _diamondIds,
                                      cv::InputOutputArrayOfArrays inMarkerCorners, cv::InputOutputArray inMarkerIds)    {

    CV_Assert(image.channels() == 1 || image.channels() == 3);
    cv::Mat src_gray;
    //obtain the gray image
    if(image.channels()==3)
        cvtColor(image, src_gray, cv::COLOR_BGR2GRAY);
    else src_gray=image.getMat();

    //detect all markers
    auto allMarkers=detectBWMarkers(board, src_gray);

    if(allMarkers.empty())return;
    //obtain the connected components
    std::vector<std::vector<Marker> > connected_markers=connectedComponents(allMarkers);

    //discard these that have more or less than 4 elements

    std::vector<cv::Vec4i> diammons_ids;
    std::vector<  std::vector<Marker> > diammons_markers;
    std::vector<std::vector<cv::Point2f> > diamondCorners;

    int threshold=10*10;
    std::vector<int> indices;
    std::vector<float> dists;
    cv::Mat query(1,2,CV_32F);
    for(const auto &comp:connected_markers){
        if(comp.size()!=4) continue;
        //make sure they   share 3 corners
        auto flannIndex=buildFlannIndex(comp);
        int totalSharedCorners=0,cornerWithMostSharedCorners=-1;
        std::pair<int,int> centralCorner;//this is the central poisition, that let us know the order of the markers in the diamond
        for(size_t m=0;m<comp.size();m++){
            auto &marker=comp[m];
             for(int c=0;c<4;c++){
                 query.ptr<float>(0)[0]=marker[c].x;
                 query.ptr<float>(0)[1]=marker[c].y;
                 int nn=flannIndex->radiusSearch(query, indices, dists, threshold, 4);
                 nn=std::min(int(indices.size()),nn);
               //  std::cout<<"nn="<<nn<<" for marker "<<marker.id<<" corner "<<c<<std::endl;
                 totalSharedCorners+=nn-1;
                 if( nn>cornerWithMostSharedCorners)
                 {
                     cornerWithMostSharedCorners=nn;
                     centralCorner={m,c};
                 }
            }
        }
        // std::cout<<"Component with markers "<<comp[0].id<<","<<comp[1].id<<","<<comp[2].id<<","<<comp[3].id<<" has "<<totalSharedCorners<<" shared corners"<<std::endl;
        // std::cout<<"Central corner is marker "<<centralCorner.first<<" corner "<<centralCorner.second<<" with "<<cornerWithMostSharedCorners<<" shared corners"<<std::endl;
        if(totalSharedCorners!=20) continue;

        //find the diamon id eximiing the poition of the central corner, which is the one with more shared corners.
        cv::Vec4i diamondIds;
        query.ptr<float>(0)[0]=comp[centralCorner.first][centralCorner.second].x;
        query.ptr<float>(0)[1]=comp[centralCorner.first][centralCorner.second].y;
        int nn=flannIndex->radiusSearch(query, indices, dists, threshold, 4);
        for(int i=0;i<nn;i++){

            int markerIdx=indices[i]/4;
            int cornerIdx=indices[i]%4;

            int idIndex;
            if( cornerIdx==2) idIndex=0;
            else if(cornerIdx==3) idIndex=1;
            else if(cornerIdx==1) idIndex=2;
            else idIndex=3;
            diamondIds[idIndex]=comp[markerIdx].id;

        }

        CharucoBoard2 dboard(cv::Size(2,2),1,1,board.dictionary,std::vector<int>{diamondIds[0],diamondIds[1],diamondIds[2],diamondIds[3]} );
        std::vector<cv::Point2f> board_corners(9);
        for(size_t m=0;m<comp.size();m++){
            for(int c=0;c<4;c++){
                auto gid=getGlobalCornerID(comp[m].id,c,dboard);
                board_corners[gid]=comp[m][c];
            }
        }

        //lets apply a corner refinement to the corners. first, estimate average lenght

        // compute the window size for the subpixel refinement based on the size of the markers. Bigger makers, bigger search zone
        double avrgLen=0;
        for(auto m:comp){
            for(int i=0;i<4;i++){
                avrgLen+=cv::norm(m[i]-m[(i+1)%4]);
            }
        }
        avrgLen/=4*comp.size();
        //here is the formula to compute the half window size
        int halfwsize= std::min(int(3* std::max(1.f,float(avrgLen)/float(34) )),9);
        //now, subpix
        cv::cornerSubPix(src_gray, board_corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));


        diamondCorners.push_back(board_corners);
        diammons_markers.push_back(comp);
        diammons_ids.push_back(diamondIds);
    }
    if(diammons_markers.empty())return;




     // //now, copy data to output
     copyVector2Output(diamondCorners,_diamondCorners);

     cv::Mat(diammons_ids).copyTo(_diamondIds);

     //group the markers and their ids into a single vector
     if( inMarkerCorners.needed() || inMarkerIds.needed()){
         std::vector<Marker> markerCorners;
         std::vector<int> markerIds;
         for(size_t i=0;i<diammons_markers.size();i++){
             for(size_t j=0;j<diammons_markers[i].size();j++){
                 markerCorners.push_back(diammons_markers[i][j]);
                 markerIds.push_back(diammons_markers[i][j].id);
             }
         }
         if( inMarkerCorners.needed())
             copyVector2Output(markerCorners,inMarkerCorners);
         if( inMarkerIds.needed())
             cv::Mat(markerIds).copyTo(inMarkerIds);
     }

}

