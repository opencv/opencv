// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "../precomp.hpp"
#include <opencv2/calib3d.hpp>

#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/aruco_board.hpp"
#include "apriltag/apriltag_quad_thresh.hpp"
#include "aruco_utils.hpp"
#include <cmath>
#include <map>

namespace cv {
namespace aruco {

using namespace std;

static inline bool readWrite(DetectorParameters &params, const FileNode* readNode,
                             FileStorage* writeStorage = nullptr)
{
    CV_Assert(readNode || writeStorage);
    bool check = false;

    check |= readWriteParameter("adaptiveThreshWinSizeMin", params.adaptiveThreshWinSizeMin, readNode, writeStorage);
    check |= readWriteParameter("adaptiveThreshWinSizeMax", params.adaptiveThreshWinSizeMax, readNode, writeStorage);
    check |= readWriteParameter("adaptiveThreshWinSizeStep", params.adaptiveThreshWinSizeStep, readNode, writeStorage);
    check |= readWriteParameter("adaptiveThreshConstant", params.adaptiveThreshConstant, readNode, writeStorage);
    check |= readWriteParameter("minMarkerPerimeterRate", params.minMarkerPerimeterRate, readNode, writeStorage);
    check |= readWriteParameter("maxMarkerPerimeterRate", params.maxMarkerPerimeterRate, readNode, writeStorage);
    check |= readWriteParameter("polygonalApproxAccuracyRate", params.polygonalApproxAccuracyRate,
                                readNode, writeStorage);
    check |= readWriteParameter("minCornerDistanceRate", params.minCornerDistanceRate, readNode, writeStorage);
    check |= readWriteParameter("minDistanceToBorder", params.minDistanceToBorder, readNode, writeStorage);
    check |= readWriteParameter("minMarkerDistanceRate", params.minMarkerDistanceRate, readNode, writeStorage);
    check |= readWriteParameter("cornerRefinementMethod", params.cornerRefinementMethod, readNode, writeStorage);
    check |= readWriteParameter("cornerRefinementWinSize", params.cornerRefinementWinSize, readNode, writeStorage);
    check |= readWriteParameter("relativeCornerRefinmentWinSize", params.relativeCornerRefinmentWinSize, readNode,
                                writeStorage);
    check |= readWriteParameter("cornerRefinementMaxIterations", params.cornerRefinementMaxIterations,
                                readNode, writeStorage);
    check |= readWriteParameter("cornerRefinementMinAccuracy", params.cornerRefinementMinAccuracy,
                                readNode, writeStorage);
    check |= readWriteParameter("markerBorderBits", params.markerBorderBits, readNode, writeStorage);
    check |= readWriteParameter("perspectiveRemovePixelPerCell", params.perspectiveRemovePixelPerCell,
                                readNode, writeStorage);
    check |= readWriteParameter("perspectiveRemoveIgnoredMarginPerCell", params.perspectiveRemoveIgnoredMarginPerCell,
                                readNode, writeStorage);
    check |= readWriteParameter("maxErroneousBitsInBorderRate", params.maxErroneousBitsInBorderRate,
                                readNode, writeStorage);
    check |= readWriteParameter("minOtsuStdDev", params.minOtsuStdDev, readNode, writeStorage);
    check |= readWriteParameter("errorCorrectionRate", params.errorCorrectionRate, readNode, writeStorage);
    check |= readWriteParameter("minGroupDistance", params.minGroupDistance, readNode, writeStorage);
    // new aruco 3 functionality
    check |= readWriteParameter("useAruco3Detection", params.useAruco3Detection, readNode, writeStorage);
    check |= readWriteParameter("minSideLengthCanonicalImg", params.minSideLengthCanonicalImg, readNode, writeStorage);
    check |= readWriteParameter("minMarkerLengthRatioOriginalImg", params.minMarkerLengthRatioOriginalImg,
                                readNode, writeStorage);
    return check;
}

bool DetectorParameters::readDetectorParameters(const FileNode& fn)
{
    if (fn.empty())
        return false;
    return readWrite(*this, &fn);
}

bool DetectorParameters::writeDetectorParameters(FileStorage& fs, const String& name)
{
    CV_Assert(fs.isOpened());
    if (!name.empty())
        fs << name << "{";
    bool res = readWrite(*this, nullptr, &fs);
    if (!name.empty())
        fs << "}";
    return res;
}

static inline bool readWrite(RefineParameters& refineParameters, const FileNode* readNode,
                             FileStorage* writeStorage = nullptr)
{
    CV_Assert(readNode || writeStorage);
    bool check = false;

    check |= readWriteParameter("minRepDistance", refineParameters.minRepDistance, readNode, writeStorage);
    check |= readWriteParameter("errorCorrectionRate", refineParameters.errorCorrectionRate, readNode, writeStorage);
    check |= readWriteParameter("checkAllOrders", refineParameters.checkAllOrders, readNode, writeStorage);
    return check;
}

RefineParameters::RefineParameters(float _minRepDistance, float _errorCorrectionRate, bool _checkAllOrders):
                                   minRepDistance(_minRepDistance), errorCorrectionRate(_errorCorrectionRate),
                                   checkAllOrders(_checkAllOrders){}

bool RefineParameters::readRefineParameters(const FileNode &fn)
{
    if (fn.empty())
        return false;
    return readWrite(*this, &fn);
}

bool RefineParameters::writeRefineParameters(FileStorage& fs, const String& name)
{
    CV_Assert(fs.isOpened());
    if (!name.empty())
        fs << name << "{";
    bool res = readWrite(*this, nullptr, &fs);
    if (!name.empty())
        fs << "}";
    return res;
}

/**
  * @brief Threshold input image using adaptive thresholding
  */
static void _threshold(InputArray _in, OutputArray _out, int winSize, double constant) {

    CV_Assert(winSize >= 3);
    if(winSize % 2 == 0) winSize++; // win size must be odd
    adaptiveThreshold(_in, _out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, winSize, constant);
}


/**
  * @brief Given a tresholded image, find the contours, calculate their polygonal approximation
  * and take those that accomplish some conditions
  */
static void _findMarkerContours(const Mat &in, vector<vector<Point2f> > &candidates,
                                vector<vector<Point> > &contoursOut, double minPerimeterRate,
                                double maxPerimeterRate, double accuracyRate,
                                double minCornerDistanceRate, int minSize) {

    CV_Assert(minPerimeterRate > 0 && maxPerimeterRate > 0 && accuracyRate > 0 &&
              minCornerDistanceRate >= 0);

    // calculate maximum and minimum sizes in pixels
    unsigned int minPerimeterPixels =
        (unsigned int)(minPerimeterRate * max(in.cols, in.rows));
    unsigned int maxPerimeterPixels =
        (unsigned int)(maxPerimeterRate * max(in.cols, in.rows));

    // for aruco3 functionality
    if (minSize != 0) {
        minPerimeterPixels = 4*minSize;
    }

    vector<vector<Point> > contours;
    findContours(in, contours, RETR_LIST, CHAIN_APPROX_NONE);
    // now filter list of contours
    for(unsigned int i = 0; i < contours.size(); i++) {
        // check perimeter
        if(contours[i].size() < minPerimeterPixels || contours[i].size() > maxPerimeterPixels)
            continue;

        // check is square and is convex
        vector<Point> approxCurve;
        approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * accuracyRate, true);
        if(approxCurve.size() != 4 || !isContourConvex(approxCurve)) continue;

        // check min distance between corners
        double minDistSq = max(in.cols, in.rows) * max(in.cols, in.rows);
        for(int j = 0; j < 4; j++) {
            double d = (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) *
                           (double)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                       (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y) *
                           (double)(approxCurve[j].y - approxCurve[(j + 1) % 4].y);
            minDistSq = min(minDistSq, d);
        }
        double minCornerDistancePixels = double(contours[i].size()) * minCornerDistanceRate;
        if(minDistSq < minCornerDistancePixels * minCornerDistancePixels) continue;

        // if it passes all the test, add to candidates vector
        vector<Point2f> currentCandidate;
        currentCandidate.resize(4);
        for(int j = 0; j < 4; j++) {
            currentCandidate[j] = Point2f((float)approxCurve[j].x, (float)approxCurve[j].y);
        }
        candidates.push_back(currentCandidate);
        contoursOut.push_back(contours[i]);
    }
}


/**
  * @brief Assure order of candidate corners is clockwise direction
  */
static void _reorderCandidatesCorners(vector<vector<Point2f> > &candidates) {

    for(unsigned int i = 0; i < candidates.size(); i++) {
        double dx1 = candidates[i][1].x - candidates[i][0].x;
        double dy1 = candidates[i][1].y - candidates[i][0].y;
        double dx2 = candidates[i][2].x - candidates[i][0].x;
        double dy2 = candidates[i][2].y - candidates[i][0].y;
        double crossProduct = (dx1 * dy2) - (dy1 * dx2);

        if(crossProduct < 0.0) { // not clockwise direction
            swap(candidates[i][1], candidates[i][3]);
        }
    }
}

static float getAverageModuleSize(const vector<Point2f>& markerCorners, int markerSize, int markerBorderBits) {
    float averageArucoModuleSize = 0.f;
    for (size_t i = 0ull; i < 4ull; i++) {
        averageArucoModuleSize += sqrt(normL2Sqr<float>(Point2f(markerCorners[i] - markerCorners[(i+1ull) % 4ull])));
    }
    int numModules = markerSize + markerBorderBits * 2;
    averageArucoModuleSize /= ((float)markerCorners.size()*numModules);
    return averageArucoModuleSize;
}

static bool checkMarker1InMarker2(const vector<Point2f>& marker1, const vector<Point2f>& marker2) {
    return pointPolygonTest(marker2, marker1[0], false) >= 0 && pointPolygonTest(marker2, marker1[1], false) >= 0 &&
           pointPolygonTest(marker2, marker1[2], false) >= 0 && pointPolygonTest(marker2, marker1[3], false) >= 0;
}

struct MarkerCandidate {
    vector<Point2f> corners;
    vector<Point> contour;
    float perimeter = 0.f;
};

struct MarkerCandidateTree : MarkerCandidate{
    int parent = -1;
    int depth = 0;
    vector<MarkerCandidate> closeContours;

    MarkerCandidateTree() {}

    MarkerCandidateTree(vector<Point2f>&& corners_, vector<Point>&& contour_) {
        corners = std::move(corners_);
        contour = std::move(contour_);
        perimeter = 0.f;
        for (size_t i = 0ull; i < 4ull; i++) {
            perimeter += sqrt(normL2Sqr<float>(Point2f(corners[i] - corners[(i+1ull) % 4ull])));
        }
    }

    bool operator<(const MarkerCandidateTree& m) const {
        // sorting the contors in descending order
        return perimeter > m.perimeter;
    }
};


// returns the average distance between the marker points
float static inline getAverageDistance(const std::vector<Point2f>& marker1, const std::vector<Point2f>& marker2) {
    float minDistSq = std::numeric_limits<float>::max();
    // fc is the first corner considered on one of the markers, 4 combinations are possible
    for(int fc = 0; fc < 4; fc++) {
        float distSq = 0;
        for(int c = 0; c < 4; c++) {
            // modC is the corner considering first corner is fc
            int modC = (c + fc) % 4;
            distSq += normL2Sqr<float>(marker1[modC] - marker2[c]);
        }
        distSq /= 4.f;
        minDistSq = min(minDistSq, distSq);
    }
    return sqrt(minDistSq);
}

/**
 * @brief Initial steps on finding square candidates
 */
static void _detectInitialCandidates(const Mat &grey, vector<vector<Point2f> > &candidates,
                                     vector<vector<Point> > &contours,
                                     const DetectorParameters &params) {

    CV_Assert(params.adaptiveThreshWinSizeMin >= 3 && params.adaptiveThreshWinSizeMax >= 3);
    CV_Assert(params.adaptiveThreshWinSizeMax >= params.adaptiveThreshWinSizeMin);
    CV_Assert(params.adaptiveThreshWinSizeStep > 0);

    // number of window sizes (scales) to apply adaptive thresholding
    int nScales =  (params.adaptiveThreshWinSizeMax - params.adaptiveThreshWinSizeMin) /
                      params.adaptiveThreshWinSizeStep + 1;

    vector<vector<vector<Point2f> > > candidatesArrays((size_t) nScales);
    vector<vector<vector<Point> > > contoursArrays((size_t) nScales);

    ////for each value in the interval of thresholding window sizes
    parallel_for_(Range(0, nScales), [&](const Range& range) {
        const int begin = range.start;
        const int end = range.end;

        for (int i = begin; i < end; i++) {
            int currScale = params.adaptiveThreshWinSizeMin + i * params.adaptiveThreshWinSizeStep;
            // threshold
            Mat thresh;
            _threshold(grey, thresh, currScale, params.adaptiveThreshConstant);

            // detect rectangles
            _findMarkerContours(thresh, candidatesArrays[i], contoursArrays[i],
                                params.minMarkerPerimeterRate, params.maxMarkerPerimeterRate,
                                params.polygonalApproxAccuracyRate, params.minCornerDistanceRate,
                                params.minSideLengthCanonicalImg);
        }
    });
    // join candidates
    for(int i = 0; i < nScales; i++) {
        for(unsigned int j = 0; j < candidatesArrays[i].size(); j++) {
            candidates.push_back(candidatesArrays[i][j]);
            contours.push_back(contoursArrays[i][j]);
        }
    }
}


/**
  * @brief Given an input image and candidate corners, extract the bits of the candidate, including
  * the border bits
  */
static Mat _extractBits(InputArray _image, const vector<Point2f>& corners, int markerSize,
                        int markerBorderBits, int cellSize, double cellMarginRate, double minStdDevOtsu) {
    CV_Assert(_image.getMat().channels() == 1);
    CV_Assert(corners.size() == 4ull);
    CV_Assert(markerBorderBits > 0 && cellSize > 0 && cellMarginRate >= 0 && cellMarginRate <= 1);
    CV_Assert(minStdDevOtsu >= 0);

    // number of bits in the marker
    int markerSizeWithBorders = markerSize + 2 * markerBorderBits;
    int cellMarginPixels = int(cellMarginRate * cellSize);

    Mat resultImg; // marker image after removing perspective
    int resultImgSize = markerSizeWithBorders * cellSize;
    Mat resultImgCorners(4, 1, CV_32FC2);
    resultImgCorners.ptr<Point2f>(0)[0] = Point2f(0, 0);
    resultImgCorners.ptr<Point2f>(0)[1] = Point2f((float)resultImgSize - 1, 0);
    resultImgCorners.ptr<Point2f>(0)[2] =
        Point2f((float)resultImgSize - 1, (float)resultImgSize - 1);
    resultImgCorners.ptr<Point2f>(0)[3] = Point2f(0, (float)resultImgSize - 1);

    // remove perspective
    Mat transformation = getPerspectiveTransform(corners, resultImgCorners);
    warpPerspective(_image, resultImg, transformation, Size(resultImgSize, resultImgSize),
                    INTER_NEAREST);

    // output image containing the bits
    Mat bits(markerSizeWithBorders, markerSizeWithBorders, CV_8UC1, Scalar::all(0));

    // check if standard deviation is enough to apply Otsu
    // if not enough, it probably means all bits are the same color (black or white)
    Mat mean, stddev;
    // Remove some border just to avoid border noise from perspective transformation
    Mat innerRegion = resultImg.colRange(cellSize / 2, resultImg.cols - cellSize / 2)
                          .rowRange(cellSize / 2, resultImg.rows - cellSize / 2);
    meanStdDev(innerRegion, mean, stddev);
    if(stddev.ptr< double >(0)[0] < minStdDevOtsu) {
        // all black or all white, depending on mean value
        if(mean.ptr< double >(0)[0] > 127)
            bits.setTo(1);
        else
            bits.setTo(0);
        return bits;
    }

    // now extract code, first threshold using Otsu
    threshold(resultImg, resultImg, 125, 255, THRESH_BINARY | THRESH_OTSU);

    // for each cell
    for(int y = 0; y < markerSizeWithBorders; y++) {
        for(int x = 0; x < markerSizeWithBorders; x++) {
            int Xstart = x * (cellSize) + cellMarginPixels;
            int Ystart = y * (cellSize) + cellMarginPixels;
            Mat square = resultImg(Rect(Xstart, Ystart, cellSize - 2 * cellMarginPixels,
                                        cellSize - 2 * cellMarginPixels));
            // count white pixels on each cell to assign its value
            size_t nZ = (size_t) countNonZero(square);
            if(nZ > square.total() / 2) bits.at<unsigned char>(y, x) = 1;
        }
    }

    return bits;
}



/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
static int _getBorderErrors(const Mat &bits, int markerSize, int borderSize) {

    int sizeWithBorders = markerSize + 2 * borderSize;

    CV_Assert(markerSize > 0 && bits.cols == sizeWithBorders && bits.rows == sizeWithBorders);

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


/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
static uint8_t _identifyOneCandidate(const Dictionary& dictionary, const Mat& _image,
                                     const vector<Point2f>& _corners, int& idx,
                                     const DetectorParameters& params, int& rotation,
                                     const float scale = 1.f) {
    CV_DbgAssert(params.markerBorderBits > 0);
    uint8_t typ=1;
    // get bits
    // scale corners to the correct size to search on the corresponding image pyramid
    vector<Point2f> scaled_corners(4);
    for (int i = 0; i < 4; ++i) {
        scaled_corners[i].x = _corners[i].x * scale;
        scaled_corners[i].y = _corners[i].y * scale;
    }

    Mat candidateBits =
        _extractBits(_image, scaled_corners, dictionary.markerSize, params.markerBorderBits,
                     params.perspectiveRemovePixelPerCell,
                     params.perspectiveRemoveIgnoredMarginPerCell, params.minOtsuStdDev);

    // analyze border bits
    int maximumErrorsInBorder =
        int(dictionary.markerSize * dictionary.markerSize * params.maxErroneousBitsInBorderRate);
    int borderErrors =
        _getBorderErrors(candidateBits, dictionary.markerSize, params.markerBorderBits);

    // check if it is a white marker
    if(params.detectInvertedMarker){
        // to get from 255 to 1
        Mat invertedImg = ~candidateBits-254;
        int invBError = _getBorderErrors(invertedImg, dictionary.markerSize, params.markerBorderBits);
        // white marker
        if(invBError<borderErrors){
            borderErrors = invBError;
            invertedImg.copyTo(candidateBits);
            typ=2;
        }
    }
    if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong

    // take only inner bits
    Mat onlyBits =
        candidateBits.rowRange(params.markerBorderBits,
                               candidateBits.rows - params.markerBorderBits)
            .colRange(params.markerBorderBits, candidateBits.cols - params.markerBorderBits);

    // try to indentify the marker
    if(!dictionary.identify(onlyBits, idx, rotation, params.errorCorrectionRate))
        return 0;

    return typ;
}

/**
 * @brief rotate the initial corner to get to the right position
 */
static void correctCornerPosition(vector<Point2f>& _candidate, int rotate){
    std::rotate(_candidate.begin(), _candidate.begin() + 4 - rotate, _candidate.end());
}

static size_t _findOptPyrImageForCanonicalImg(
        const vector<Mat>& img_pyr,
        const int scaled_width,
        const int cur_perimeter,
        const int min_perimeter) {
    CV_Assert(scaled_width > 0);
    size_t optLevel = 0;
    float dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < img_pyr.size(); ++i) {
        const float scale = img_pyr[i].cols / static_cast<float>(scaled_width);
        const float perimeter_scaled = cur_perimeter * scale;
        // instead of std::abs() favor the larger pyramid level by checking if the distance is postive
        // will slow down the algorithm but find more corners in the end
        const float new_dist = perimeter_scaled - min_perimeter;
        if (new_dist < dist && new_dist > 0.f) {
            dist = new_dist;
            optLevel = i;
        }
    }
    return optLevel;
}


/**
 * Line fitting  A * B = C :: Called from function refineCandidateLines
 * @param nContours contour-container
 */
static Point3f _interpolate2Dline(const vector<Point2f>& nContours){
    CV_Assert(nContours.size() >= 2);
    float minX, minY, maxX, maxY;
    minX = maxX = nContours[0].x;
    minY = maxY = nContours[0].y;

    for(unsigned int i = 0; i< nContours.size(); i++){
        minX = nContours[i].x < minX ? nContours[i].x : minX;
        minY = nContours[i].y < minY ? nContours[i].y : minY;
        maxX = nContours[i].x > maxX ? nContours[i].x : maxX;
        maxY = nContours[i].y > maxY ? nContours[i].y : maxY;
    }

    Mat A = Mat::ones((int)nContours.size(), 2, CV_32F); // Coefficient Matrix (N x 2)
    Mat B((int)nContours.size(), 1, CV_32F);                // Variables   Matrix (N x 1)
    Mat C;                                            // Constant

    if(maxX - minX > maxY - minY){
        for(unsigned int i =0; i < nContours.size(); i++){
            A.at<float>(i,0)= nContours[i].x;
            B.at<float>(i,0)= nContours[i].y;
        }

        solve(A, B, C, DECOMP_NORMAL);

        return Point3f(C.at<float>(0, 0), -1., C.at<float>(1, 0));
    }
    else{
        for(unsigned int i =0; i < nContours.size(); i++){
            A.at<float>(i,0)= nContours[i].y;
            B.at<float>(i,0)= nContours[i].x;
        }

        solve(A, B, C, DECOMP_NORMAL);

        return Point3f(-1., C.at<float>(0, 0), C.at<float>(1, 0));
    }

}

/**
 * Find the Point where the lines crosses :: Called from function refineCandidateLines
 * @param nLine1
 * @param nLine2
 * @return Crossed Point
 */
static Point2f _getCrossPoint(Point3f nLine1, Point3f nLine2){
    Matx22f A(nLine1.x, nLine1.y, nLine2.x, nLine2.y);
    Vec2f B(-nLine1.z, -nLine2.z);
    return Vec2f(A.solve(B).val);
}

/**
 * Refine Corners using the contour vector :: Called from function detectMarkers
 * @param nContours contour-container
 * @param nCorners candidate Corners
 */
static void _refineCandidateLines(vector<Point>& nContours, vector<Point2f>& nCorners){
    vector<Point2f> contour2f(nContours.begin(), nContours.end());
    /* 5 groups :: to group the edges
     * 4 - classified by its corner
     * extra group - (temporary) if contours do not begin with a corner
     */
    vector<Point2f> cntPts[5];
    int cornerIndex[4]={-1};
    int group=4;

    for ( unsigned int i =0; i < nContours.size(); i++ ) {
        for(unsigned int j=0; j<4; j++){
            if ( nCorners[j] == contour2f[i] ){
                cornerIndex[j] = i;
                group=j;
            }
        }
        cntPts[group].push_back(contour2f[i]);
    }
    for (int i = 0; i < 4; i++)
    {
        CV_Assert(cornerIndex[i] != -1);
    }
    // saves extra group into corresponding
    if( !cntPts[4].empty() ){
        for( unsigned int i=0; i < cntPts[4].size() ; i++ )
            cntPts[group].push_back(cntPts[4].at(i));
        cntPts[4].clear();
    }

    //Evaluate contour direction :: using the position of the detected corners
    int inc=1;

        inc = ( (cornerIndex[0] > cornerIndex[1]) &&  (cornerIndex[3] > cornerIndex[0]) ) ? -1:inc;
    inc = ( (cornerIndex[2] > cornerIndex[3]) &&  (cornerIndex[1] > cornerIndex[2]) ) ? -1:inc;

    // calculate the line :: who passes through the grouped points
    Point3f lines[4];
    for(int i=0; i<4; i++){
        lines[i]=_interpolate2Dline(cntPts[i]);
    }

    /*
     * calculate the corner :: where the lines crosses to each other
     * clockwise direction        no clockwise direction
     *      0                           1
     *      .---. 1                     .---. 2
     *      |   |                       |   |
     *    3 .___.                     0 .___.
     *          2                           3
     */
    for(int i=0; i < 4; i++){
        if(inc<0)
            nCorners[i] = _getCrossPoint(lines[ i ], lines[ (i+1)%4 ]);    // 01 12 23 30
        else
            nCorners[i] = _getCrossPoint(lines[ i ], lines[ (i+3)%4 ]);    // 30 01 12 23
    }
}

static inline void findCornerInPyrImage(const float scale_init, const int closest_pyr_image_idx,
                                        const vector<Mat>& grey_pyramid, Mat corners,
                                        const DetectorParameters& params) {
    // scale them to the closest pyramid level
    if (scale_init != 1.f)
        corners *= scale_init; // scale_init * scale_pyr
    for (int idx = closest_pyr_image_idx - 1; idx >= 0; --idx) {
        // scale them to new pyramid level
        corners *= 2.f; // *= scale_pyr;
        // use larger win size for larger images
        const int subpix_win_size = std::max(grey_pyramid[idx].cols, grey_pyramid[idx].rows) > 1080 ? 5 : 3;
        cornerSubPix(grey_pyramid[idx], corners,
                     Size(subpix_win_size, subpix_win_size),
                     Size(-1, -1),
                     TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                  params.cornerRefinementMaxIterations,
                                  params.cornerRefinementMinAccuracy));
    }
}

enum class DictionaryMode {
    Single,
    Multi
};

struct ArucoDetector::ArucoDetectorImpl {
    /// dictionaries indicates the types of markers that will be searched
    vector<Dictionary> dictionaries;

    /// marker detection parameters, check DetectorParameters docs to see available settings
    DetectorParameters detectorParams;

    /// marker refine parameters
    RefineParameters refineParams;
    ArucoDetectorImpl() {}

    ArucoDetectorImpl(const vector<Dictionary>&_dictionaries, const DetectorParameters &_detectorParams,
                      const RefineParameters& _refineParams): dictionaries(_dictionaries),
                      detectorParams(_detectorParams), refineParams(_refineParams) {
                          CV_Assert(!dictionaries.empty());
                      }

    /*
     * @brief Detect markers either using multiple or just first dictionary
     */
    void detectMarkers(InputArray _image, OutputArrayOfArrays _corners, OutputArray _ids,
            OutputArrayOfArrays _rejectedImgPoints, OutputArray _dictIndices, DictionaryMode dictMode) {
        CV_Assert(!_image.empty());

        CV_Assert(detectorParams.markerBorderBits > 0);
        // check that the parameters are set correctly if Aruco3 is used
        CV_Assert(!(detectorParams.useAruco3Detection == true &&
                    detectorParams.minSideLengthCanonicalImg == 0 &&
                    detectorParams.minMarkerLengthRatioOriginalImg == 0.0));

        Mat grey;
        _convertToGrey(_image, grey);

        // Aruco3 functionality is the extension of Aruco.
        // The description can be found in:
        // [1] Speeded up detection of squared fiducial markers, 2018, FJ Romera-Ramirez et al.
        // if Aruco3 functionality if not wanted
        // change some parameters to be sure to turn it off
        if (!detectorParams.useAruco3Detection) {
            detectorParams.minMarkerLengthRatioOriginalImg = 0.0;
            detectorParams.minSideLengthCanonicalImg = 0;
        }
        else {
            // always turn on corner refinement in case of Aruco3, due to upsampling
            detectorParams.cornerRefinementMethod = (int)CORNER_REFINE_SUBPIX;
            // only CORNER_REFINE_SUBPIX implement correctly for useAruco3Detection
            // Todo: update other CORNER_REFINE methods
        }

        /// Step 0: equation (2) from paper [1]
        const float fxfy = (!detectorParams.useAruco3Detection ? 1.f : detectorParams.minSideLengthCanonicalImg /
                (detectorParams.minSideLengthCanonicalImg + std::max(grey.cols, grey.rows)*
                 detectorParams.minMarkerLengthRatioOriginalImg));

        /// Step 1: create image pyramid. Section 3.4. in [1]
        vector<Mat> grey_pyramid;
        int closest_pyr_image_idx = 0, num_levels = 0;
        //// Step 1.1: resize image with equation (1) from paper [1]
        if (detectorParams.useAruco3Detection) {
            const float scale_pyr = 2.f;
            const float img_area = static_cast<float>(grey.rows*grey.cols);
            const float min_area_marker = static_cast<float>(detectorParams.minSideLengthCanonicalImg*
                    detectorParams.minSideLengthCanonicalImg);
            // find max level
            num_levels = static_cast<int>(log2(img_area / min_area_marker)/scale_pyr);
            // the closest pyramid image to the downsampled segmentation image
            // will later be used as start index for corner upsampling
            const float scale_img_area = img_area * fxfy * fxfy;
            closest_pyr_image_idx = cvRound(log2(img_area / scale_img_area)/scale_pyr);
        }
        buildPyramid(grey, grey_pyramid, num_levels);

        // resize to segmentation image
        // in this reduces size the contours will be detected
        if (fxfy != 1.f)
            resize(grey, grey, Size(cvRound(fxfy * grey.cols), cvRound(fxfy * grey.rows)));

        /// STEP 2: Detect marker candidates
        vector<vector<Point2f> > candidates;
        vector<vector<Point> > contours;
        vector<int> ids;

        /// STEP 2.a Detect marker candidates :: using AprilTag
        if(detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_APRILTAG){
            _apriltag(grey, detectorParams, candidates, contours);
        }
        /// STEP 2.b Detect marker candidates :: traditional way
        else {
            detectCandidates(grey, candidates, contours);
        }

        /// STEP 2.c FILTER OUT NEAR CANDIDATE PAIRS
        vector<int> dictIndices;
        vector<vector<Point2f>> rejectedImgPoints;
        if (DictionaryMode::Single == dictMode) {
            Dictionary& dictionary = dictionaries.at(0);
            auto selectedCandidates = filterTooCloseCandidates(candidates, contours, dictionary.markerSize);
            candidates.clear();
            contours.clear();

            /// STEP 2: Check candidate codification (identify markers)
            identifyCandidates(grey, grey_pyramid, selectedCandidates, candidates, contours,
                    ids, dictionary, rejectedImgPoints);

            /// STEP 3: Corner refinement :: use corner subpix
            if (detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_SUBPIX) {
                performCornerSubpixRefinement(grey, grey_pyramid, closest_pyr_image_idx, candidates, dictionary);
            }
        } else if (DictionaryMode::Multi == dictMode) {
            map<int, vector<MarkerCandidateTree>> candidatesPerDictionarySize;
            for (const Dictionary& dictionary : dictionaries) {
                candidatesPerDictionarySize.emplace(dictionary.markerSize, vector<MarkerCandidateTree>());
            }

            // create candidate trees for each dictionary size
            for (auto& candidatesTreeEntry : candidatesPerDictionarySize) {
                // copy candidates
                vector<vector<Point2f>> candidatesCopy = candidates;
                vector<vector<Point> > contoursCopy = contours;
                candidatesTreeEntry.second = filterTooCloseCandidates(candidatesCopy, contoursCopy, candidatesTreeEntry.first);
            }
            candidates.clear();
            contours.clear();

            /// STEP 2: Check candidate codification (identify markers)
            int dictIndex = 0;
            for (const Dictionary&  currentDictionary : dictionaries) {
                // temporary variable to store the current candidates
                vector<vector<Point2f>> currentCandidates;
                identifyCandidates(grey, grey_pyramid, candidatesPerDictionarySize.at(currentDictionary.markerSize), currentCandidates, contours,
                        ids, currentDictionary, rejectedImgPoints);
                if (_dictIndices.needed()) {
                    dictIndices.insert(dictIndices.end(), currentCandidates.size(), dictIndex);
                }

                /// STEP 3: Corner refinement :: use corner subpix
                if (detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_SUBPIX) {
                    performCornerSubpixRefinement(grey, grey_pyramid, closest_pyr_image_idx, currentCandidates, currentDictionary);
                }
                candidates.insert(candidates.end(), currentCandidates.begin(), currentCandidates.end());
                dictIndex++;
            }

            // Clean up rejectedImgPoints by comparing to itself and all candidates
            const float epsilon = 0.000001f;
            auto compareCandidates = [epsilon](vector<Point2f> a, vector<Point2f> b) {
                for (int i = 0; i < 4; i++) {
                    if (std::abs(a[i].x - b[i].x) > epsilon || std::abs(a[i].y - b[i].y) > epsilon) {
                        return false;
                    }
                }
                return true;
            };
            std::sort(rejectedImgPoints.begin(), rejectedImgPoints.end(), [](const vector<Point2f>& a, const vector<Point2f>&b){
                    float avgX = (a[0].x + a[1].x + a[2].x + a[3].x)*.25f;
                    float avgY = (a[0].y + a[1].y + a[2].y + a[3].y)*.25f;
                    float aDist = avgX*avgX + avgY*avgY;
                    avgX = (b[0].x + b[1].x + b[2].x + b[3].x)*.25f;
                    avgY = (b[0].y + b[1].y + b[2].y + b[3].y)*.25f;
                    float bDist = avgX*avgX + avgY*avgY;
                    return aDist < bDist;
                });
            auto last = std::unique(rejectedImgPoints.begin(), rejectedImgPoints.end(), compareCandidates);
            rejectedImgPoints.erase(last, rejectedImgPoints.end());

            for (auto it = rejectedImgPoints.begin(); it != rejectedImgPoints.end();) {
                bool erased = false;
                for (const auto& candidate : candidates) {
                    if (compareCandidates(candidate, *it)) {
                        it = rejectedImgPoints.erase(it);
                        erased = true;
                        break;
                    }
                }
                if (!erased) {
                    it++;
                }
            }
        }

        /// STEP 3, Optional : Corner refinement :: use contour container
        if (detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_CONTOUR){

            if (!ids.empty()) {

                // do corner refinement using the contours for each detected markers
                parallel_for_(Range(0, (int)candidates.size()), [&](const Range& range) {
                        for (int i = range.start; i < range.end; i++) {
                        _refineCandidateLines(contours[i], candidates[i]);
                        }
                        });
            }
        }

        if (detectorParams.cornerRefinementMethod != (int)CORNER_REFINE_SUBPIX && fxfy != 1.f) {
            // only CORNER_REFINE_SUBPIX implement correctly for useAruco3Detection
            // Todo: update other CORNER_REFINE methods

            // scale to orignal size, this however will lead to inaccurate detections!
            for (auto &vecPoints : candidates)
                for (auto &point : vecPoints)
                    point *= 1.f/fxfy;
        }

        // copy to output arrays
        _copyVector2Output(candidates, _corners);
        Mat(ids).copyTo(_ids);
        if(_rejectedImgPoints.needed()) {
            _copyVector2Output(rejectedImgPoints, _rejectedImgPoints);
        }
        if (_dictIndices.needed()) {
            Mat(dictIndices).copyTo(_dictIndices);
        }
    }

    /**
     * @brief Detect square candidates in the input image
     */
    void detectCandidates(const Mat& grey, vector<vector<Point2f> >& candidates, vector<vector<Point> >& contours) {
        /// 1. DETECT FIRST SET OF CANDIDATES
        _detectInitialCandidates(grey, candidates, contours, detectorParams);
        /// 2. SORT CORNERS
        _reorderCandidatesCorners(candidates);
    }

    /**
     * @brief FILTER OUT NEAR CANDIDATES PAIRS AND TOO NEAR CANDIDATES TO IMAGE BORDER
     *
     * save the outer/inner border (i.e. potential candidates) to vector<MarkerCandidateTree>,
     * clear candidates and contours
     */
    vector<MarkerCandidateTree>
    filterTooCloseCandidates(const Size &imageSize, vector<vector<Point2f> > &candidates, vector<vector<Point> > &contours, int markerSize) {
        CV_Assert(detectorParams.minMarkerDistanceRate >= 0. && detectorParams.minDistanceToBorder >= 0);
        vector<MarkerCandidateTree> candidateTree(candidates.size());
        for(size_t i = 0ull; i < candidates.size(); i++) {
            candidateTree[i] = MarkerCandidateTree(std::move(candidates[i]), std::move(contours[i]));
        }

        // sort candidates from big to small
        std::stable_sort(candidateTree.begin(), candidateTree.end());
        // group index for each candidate
        vector<int> groupId(candidateTree.size(), -1);
        vector<vector<size_t> > groupedCandidates;
        vector<bool> isSelectedContours(candidateTree.size(), true);

        for (size_t i = 0ull; i < candidateTree.size(); i++) {
            for (size_t j = i + 1ull; j < candidateTree.size(); j++) {
                float minDist = getAverageDistance(candidateTree[i].corners, candidateTree[j].corners);
                // if mean distance is too low, group markers
                // the distance between the points of two independent markers should be more than half the side of the marker
                // half the side of the marker = (perimeter / 4) * 0.5 = perimeter * 0.125
                if(minDist < candidateTree[j].perimeter*(float)detectorParams.minMarkerDistanceRate) {
                    isSelectedContours[i] = false;
                    isSelectedContours[j] = false;
                    // i and j are not related to a group
                    if(groupId[i] < 0 && groupId[j] < 0){
                        // mark candidates with their corresponding group number
                        groupId[i] = groupId[j] = (int)groupedCandidates.size();
                        // create group
                        groupedCandidates.push_back({i, j});
                    }
                    // i is related to a group
                    else if(groupId[i] > -1 && groupId[j] == -1) {
                        int group = groupId[i];
                        groupId[j] = group;
                        // add to group
                        groupedCandidates[group].push_back(j);
                    }
                    // j is related to a group
                    else if(groupId[j] > -1 && groupId[i] == -1) {
                        int group = groupId[j];
                        groupId[i] = group;
                        // add to group
                        groupedCandidates[group].push_back(i);
                    }
                }
            }
            // group of one candidate
            if(isSelectedContours[i]) {
                isSelectedContours[i] = false;
                groupId[i] = (int)groupedCandidates.size();
                groupedCandidates.push_back({i});
            }
        }

        for (vector<size_t>& grouped : groupedCandidates) {
            if (detectorParams.detectInvertedMarker) // if detectInvertedMarker choose smallest contours
                std::stable_sort(grouped.begin(), grouped.end(), [](const size_t &a, const size_t &b) {
                    return a > b;
                });
            else // if detectInvertedMarker==false choose largest contours
                std::stable_sort(grouped.begin(), grouped.end());
            size_t currId = grouped[0];
            // check if it is too near to the image border
            bool tooNearBorder = false;
            for (const auto& corner : candidateTree[currId].corners) {
                if (corner.x < detectorParams.minDistanceToBorder ||
                    corner.y < detectorParams.minDistanceToBorder ||
                    corner.x > imageSize.width - 1 - detectorParams.minDistanceToBorder ||
                    corner.y > imageSize.height - 1 - detectorParams.minDistanceToBorder) {
                    tooNearBorder = true;
                    break;
                }
            }
            if (tooNearBorder) {
                isSelectedContours[currId] = false;
                countSelectedContours--;
                continue;
            }
            isSelectedContours[currId] = true;
            for (size_t i = 1ull; i < grouped.size(); i++) {
                size_t id = grouped[i];
                float dist = getAverageDistance(candidateTree[id].corners, candidateTree[currId].corners);
                float moduleSize = getAverageModuleSize(candidateTree[id].corners, markerSize, detectorParams.markerBorderBits);
                if (dist > detectorParams.minGroupDistance*moduleSize) {
                    currId = id;
                    candidateTree[grouped[0]].closeContours.push_back(candidateTree[id]);
                }
            }
        }

        vector<MarkerCandidateTree> selectedCandidates(groupedCandidates.size());
        size_t countSelectedContours = 0ull;
        for (size_t i = 0ull; i < candidateTree.size(); i++) {
            if (isSelectedContours[i]) {
                selectedCandidates[countSelectedContours] = std::move(candidateTree[i]);
                countSelectedContours++;
            }
        }

        // find hierarchy in the candidate tree
        for (int i = (int)selectedCandidates.size()-1; i >= 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                if (checkMarker1InMarker2(selectedCandidates[i].corners, selectedCandidates[j].corners)) {
                    selectedCandidates[i].parent = j;
                    selectedCandidates[j].depth = max(selectedCandidates[j].depth, selectedCandidates[i].depth + 1);
                    break;
                }
            }
        }
        return selectedCandidates;
    }

    /**
     * @brief Identify square candidates according to a marker dictionary
     */
    void identifyCandidates(const Mat& grey, const vector<Mat>& image_pyr, vector<MarkerCandidateTree>& selectedContours,
                            vector<vector<Point2f> >& accepted, vector<vector<Point> >& contours,
                            vector<int>& ids, const Dictionary& currentDictionary, vector<vector<Point2f>>& rejected) const {
        size_t ncandidates = selectedContours.size();

        vector<int> idsTmp(ncandidates, -1);
        vector<int> rotated(ncandidates, 0);
        vector<uint8_t> validCandidates(ncandidates, 0);
        vector<uint8_t> was(ncandidates, false);
        bool checkCloseContours = true;

        int maxDepth = 0;
        for (size_t i = 0ull; i < selectedContours.size(); i++)
            maxDepth = max(selectedContours[i].depth, maxDepth);
        vector<vector<size_t>> depths(maxDepth+1);
        for (size_t i = 0ull; i < selectedContours.size(); i++) {
            depths[selectedContours[i].depth].push_back(i);
        }

        //// Analyze each of the candidates
        int depth = 0;
        size_t counter = 0;
        while (counter < ncandidates) {
            parallel_for_(Range(0, (int)depths[depth].size()), [&](const Range& range) {
                const int begin = range.start;
                const int end = range.end;
                for (int i = begin; i < end; i++) {
                    size_t v = depths[depth][i];
                    was[v] = true;
                    Mat img = grey;
                    // implements equation (4)
                    if (detectorParams.useAruco3Detection) {
                        const int minPerimeter = detectorParams.minSideLengthCanonicalImg * 4;
                        const size_t nearestImgId = _findOptPyrImageForCanonicalImg(image_pyr, grey.cols, static_cast<int>(selectedContours[v].contour.size()), minPerimeter);
                        img = image_pyr[nearestImgId];
                    }
                    const float scale = detectorParams.useAruco3Detection ? img.cols / static_cast<float>(grey.cols) : 1.f;

                    validCandidates[v] = _identifyOneCandidate(currentDictionary, img, selectedContours[v].corners, idsTmp[v], detectorParams, rotated[v], scale);

                    if (validCandidates[v] == 0 && checkCloseContours) {
                        for (const MarkerCandidate& closeMarkerCandidate: selectedContours[v].closeContours) {
                            validCandidates[v] = _identifyOneCandidate(currentDictionary, img, closeMarkerCandidate.corners, idsTmp[v], detectorParams, rotated[v], scale);
                            if (validCandidates[v] > 0) {
                                selectedContours[v].corners = closeMarkerCandidate.corners;
                                selectedContours[v].contour = closeMarkerCandidate.contour;
                                break;
                            }
                        }
                    }
                }
            });

            // visit the parent vertices of the detected markers to skip identify parent contours
            for(size_t v : depths[depth]) {
                if(validCandidates[v] > 0) {
                    int parent = selectedContours[v].parent;
                    while (parent != -1) {
                        if (!was[parent]) {
                            was[parent] = true;
                            counter++;
                        }
                        parent = selectedContours[parent].parent;
                    }
                }
                counter++;
            }
            depth++;
        }

        for (size_t i = 0ull; i < selectedContours.size(); i++) {
            if (validCandidates[i] > 0) {
                    // shift corner positions to the correct rotation
                    correctCornerPosition(selectedContours[i].corners, rotated[i]);

                    accepted.push_back(selectedContours[i].corners);
                    contours.push_back(selectedContours[i].contour);
                    ids.push_back(idsTmp[i]);
            }
            else {
                rejected.push_back(selectedContours[i].corners);
            }
        }
    }

};

    void performCornerSubpixRefinement(const Mat& grey, const vector<Mat>& grey_pyramid, int closest_pyr_image_idx, const vector<vector<Point2f>>& candidates, const Dictionary& dictionary) const {
        CV_Assert(detectorParams.cornerRefinementWinSize > 0 && detectorParams.cornerRefinementMaxIterations > 0 &&
                detectorParams.cornerRefinementMinAccuracy > 0);
        // Do subpixel estimation. In Aruco3 start on the lowest pyramid level and upscale the corners
        parallel_for_(Range(0, (int)candidates.size()), [&](const Range& range) {
            const int begin = range.start;
            const int end = range.end;

            for (int i = begin; i < end; i++) {
                if (detectorParams.useAruco3Detection) {
                    const float scale_init = (float) grey_pyramid[closest_pyr_image_idx].cols / grey.cols;
                    findCornerInPyrImage(scale_init, closest_pyr_image_idx, grey_pyramid, Mat(candidates[i]), detectorParams);
                } else {
                    int cornerRefinementWinSize = std::max(1, cvRound(detectorParams.relativeCornerRefinmentWinSize*
                                getAverageModuleSize(candidates[i], dictionary.markerSize, detectorParams.markerBorderBits)));
                    cornerRefinementWinSize = min(cornerRefinementWinSize, detectorParams.cornerRefinementWinSize);
                    cornerSubPix(grey, Mat(candidates[i]), Size(cornerRefinementWinSize, cornerRefinementWinSize), Size(-1, -1),
                            TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                detectorParams.cornerRefinementMaxIterations,
                                detectorParams.cornerRefinementMinAccuracy));
                }
            }
        });
    }
};

ArucoDetector::ArucoDetector(const Dictionary &_dictionary,
                             const DetectorParameters &_detectorParams,
                             const RefineParameters& _refineParams) {
    arucoDetectorImpl = makePtr<ArucoDetectorImpl>(vector<Dictionary>{_dictionary}, _detectorParams, _refineParams);
}

ArucoDetector::ArucoDetector(const vector<Dictionary> &_dictionaries,
                             const DetectorParameters &_detectorParams,
                             const RefineParameters& _refineParams) {
    arucoDetectorImpl = makePtr<ArucoDetectorImpl>(_dictionaries, _detectorParams, _refineParams);
}

void ArucoDetector::detectMarkers(InputArray _image, OutputArrayOfArrays _corners, OutputArray _ids,
                                  OutputArrayOfArrays _rejectedImgPoints) const {
    arucoDetectorImpl->detectMarkers(_image, _corners, _ids, _rejectedImgPoints, noArray(), DictionaryMode::Single);
}

void ArucoDetector::detectMarkersMultiDict(InputArray _image, OutputArrayOfArrays _corners, OutputArray _ids,
                                  OutputArrayOfArrays _rejectedImgPoints, OutputArray _dictIndices) const {
    arucoDetectorImpl->detectMarkers(_image, _corners, _ids, _rejectedImgPoints, _dictIndices, DictionaryMode::Multi);
}

/**
  * Project board markers that are not included in the list of detected markers
  */
static inline void _projectUndetectedMarkers(const Board &board, InputOutputArrayOfArrays detectedCorners,
                                             InputOutputArray detectedIds, InputArray cameraMatrix, InputArray distCoeffs,
                                             vector<vector<Point2f> >& undetectedMarkersProjectedCorners,
                                             OutputArray undetectedMarkersIds) {
    Mat rvec, tvec; // first estimate board pose with the current avaible markers
    Mat objPoints, imgPoints; // object and image points for the solvePnP function
    // To refine corners of ArUco markers the function refineDetectedMarkers() find an aruco markers pose from 3D-2D point correspondences.
    // To find 3D-2D point correspondences uses matchImagePoints().
    // The method matchImagePoints() works with ArUco corners (in Board/GridBoard cases) or with ChArUco corners (in CharucoBoard case).
    // To refine corners of ArUco markers we need work with ArUco corners only in all boards.
    // To call matchImagePoints() with ArUco corners for all boards we need to call matchImagePoints() from base class Board.
    // The method matchImagePoints() implemented in Pimpl and we need to create temp Board object to call the base method.
    Board(board.getObjPoints(), board.getDictionary(), board.getIds()).matchImagePoints(detectedCorners, detectedIds, objPoints, imgPoints);
    if (objPoints.total() < 4ull) // at least one marker from board so rvec and tvec are valid
        return;
    solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec);

    // search undetected markers and project them using the previous pose
    vector<vector<Point2f> > undetectedCorners;
    const std::vector<int>& ids = board.getIds();
    vector<int> undetectedIds;
    for(unsigned int i = 0; i < ids.size(); i++) {
        int foundIdx = -1;
        for(unsigned int j = 0; j < detectedIds.total(); j++) {
            if(ids[i] == detectedIds.getMat().ptr<int>()[j]) {
                foundIdx = j;
                break;
            }
        }

        // not detected
        if(foundIdx == -1) {
            undetectedCorners.push_back(vector<Point2f>());
            undetectedIds.push_back(ids[i]);
            projectPoints(board.getObjPoints()[i], rvec, tvec, cameraMatrix, distCoeffs,
                          undetectedCorners.back());
        }
    }
    // parse output
    Mat(undetectedIds).copyTo(undetectedMarkersIds);
    undetectedMarkersProjectedCorners = undetectedCorners;
}

/**
  * Interpolate board markers that are not included in the list of detected markers using
  * global homography
  */
static void _projectUndetectedMarkers(const Board &_board, InputOutputArrayOfArrays _detectedCorners,
                               InputOutputArray _detectedIds,
                               vector<vector<Point2f> >& _undetectedMarkersProjectedCorners,
                               OutputArray _undetectedMarkersIds) {
    // check board points are in the same plane, if not, global homography cannot be applied
    CV_Assert(_board.getObjPoints().size() > 0);
    CV_Assert(_board.getObjPoints()[0].size() > 0);
    float boardZ = _board.getObjPoints()[0][0].z;
    for(unsigned int i = 0; i < _board.getObjPoints().size(); i++) {
        for(unsigned int j = 0; j < _board.getObjPoints()[i].size(); j++)
            CV_Assert(boardZ == _board.getObjPoints()[i][j].z);
    }

    vector<Point2f> detectedMarkersObj2DAll; // Object coordinates (without Z) of all the detected
                                             // marker corners in a single vector
    vector<Point2f> imageCornersAll; // Image corners of all detected markers in a single vector
    vector<vector<Point2f> > undetectedMarkersObj2D; // Object coordinates (without Z) of all
                                                        // missing markers in different vectors
    vector<int> undetectedMarkersIds; // ids of missing markers
    // find markers included in board, and missing markers from board. Fill the previous vectors
    for(unsigned int j = 0; j < _board.getIds().size(); j++) {
        bool found = false;
        for(unsigned int i = 0; i < _detectedIds.total(); i++) {
            if(_detectedIds.getMat().ptr<int>()[i] == _board.getIds()[j]) {
                for(int c = 0; c < 4; c++) {
                    imageCornersAll.push_back(_detectedCorners.getMat(i).ptr<Point2f>()[c]);
                    detectedMarkersObj2DAll.push_back(
                        Point2f(_board.getObjPoints()[j][c].x, _board.getObjPoints()[j][c].y));
                }
                found = true;
                break;
            }
        }
        if(!found) {
            undetectedMarkersObj2D.push_back(vector<Point2f>());
            for(int c = 0; c < 4; c++) {
                undetectedMarkersObj2D.back().push_back(
                    Point2f(_board.getObjPoints()[j][c].x, _board.getObjPoints()[j][c].y));
            }
            undetectedMarkersIds.push_back(_board.getIds()[j]);
        }
    }
    if(imageCornersAll.size() == 0) return;

    // get homography from detected markers
    Mat transformation = findHomography(detectedMarkersObj2DAll, imageCornersAll);

    _undetectedMarkersProjectedCorners.resize(undetectedMarkersIds.size());

    // for each undetected marker, apply transformation
    for(unsigned int i = 0; i < undetectedMarkersObj2D.size(); i++) {
        perspectiveTransform(undetectedMarkersObj2D[i], _undetectedMarkersProjectedCorners[i], transformation);
    }
    Mat(undetectedMarkersIds).copyTo(_undetectedMarkersIds);
}

void ArucoDetector::refineDetectedMarkers(InputArray _image, const Board& _board,
                                          InputOutputArrayOfArrays _detectedCorners, InputOutputArray _detectedIds,
                                          InputOutputArrayOfArrays _rejectedCorners, InputArray _cameraMatrix,
                                          InputArray _distCoeffs, OutputArray _recoveredIdxs) const {
    DetectorParameters& detectorParams = arucoDetectorImpl->detectorParams;
    const Dictionary& dictionary = arucoDetectorImpl->dictionaries.at(0);
    RefineParameters& refineParams = arucoDetectorImpl->refineParams;
    CV_Assert(refineParams.minRepDistance > 0);

    if(_detectedIds.total() == 0 || _rejectedCorners.total() == 0) return;

    // get projections of missing markers in the board
    vector<vector<Point2f> > undetectedMarkersCorners;
    vector<int> undetectedMarkersIds;
    if(_cameraMatrix.total() != 0) {
        // reproject based on camera projection model
        _projectUndetectedMarkers(_board, _detectedCorners, _detectedIds, _cameraMatrix, _distCoeffs,
                                  undetectedMarkersCorners, undetectedMarkersIds);

    } else {
        // reproject based on global homography
        _projectUndetectedMarkers(_board, _detectedCorners, _detectedIds, undetectedMarkersCorners,
                                  undetectedMarkersIds);
    }

    // list of missing markers indicating if they have been assigned to a candidate
    vector<bool > alreadyIdentified(_rejectedCorners.total(), false);

    // maximum bits that can be corrected
    int maxCorrectionRecalculated =
        int(double(dictionary.maxCorrectionBits) * refineParams.errorCorrectionRate);

    Mat grey;
    _convertToGrey(_image, grey);

    // vector of final detected marker corners and ids
    vector<vector<Point2f> > finalAcceptedCorners;
    vector<int> finalAcceptedIds;
    // fill with the current markers
    finalAcceptedCorners.resize(_detectedCorners.total());
    finalAcceptedIds.resize(_detectedIds.total());
    for(unsigned int i = 0; i < _detectedIds.total(); i++) {
        finalAcceptedCorners[i] = _detectedCorners.getMat(i).clone();
        finalAcceptedIds[i] = _detectedIds.getMat().ptr<int>()[i];
    }
    vector<int> recoveredIdxs; // original indexes of accepted markers in _rejectedCorners

    // for each missing marker, try to find a correspondence
    for(unsigned int i = 0; i < undetectedMarkersIds.size(); i++) {

        // best match at the moment
        int closestCandidateIdx = -1;
        double closestCandidateDistance = refineParams.minRepDistance * refineParams.minRepDistance + 1;
        Mat closestRotatedMarker;

        for(unsigned int j = 0; j < _rejectedCorners.total(); j++) {
            if(alreadyIdentified[j]) continue;

            // check distance
            double minDistance = closestCandidateDistance + 1;
            bool valid = false;
            int validRot = 0;
            for(int c = 0; c < 4; c++) { // first corner in rejected candidate
                double currentMaxDistance = 0;
                for(int k = 0; k < 4; k++) {
                    Point2f rejCorner = _rejectedCorners.getMat(j).ptr<Point2f>()[(c + k) % 4];
                    Point2f distVector = undetectedMarkersCorners[i][k] - rejCorner;
                    double cornerDist = distVector.x * distVector.x + distVector.y * distVector.y;
                    currentMaxDistance = max(currentMaxDistance, cornerDist);
                }
                // if distance is better than current best distance
                if(currentMaxDistance < closestCandidateDistance) {
                    valid = true;
                    validRot = c;
                    minDistance = currentMaxDistance;
                }
                if(!refineParams.checkAllOrders) break;
            }

            if(!valid) continue;

            // apply rotation
            Mat rotatedMarker;
            if(refineParams.checkAllOrders) {
                rotatedMarker = Mat(4, 1, CV_32FC2);
                for(int c = 0; c < 4; c++)
                    rotatedMarker.ptr<Point2f>()[c] =
                        _rejectedCorners.getMat(j).ptr<Point2f>()[(c + 4 + validRot) % 4];
            }
            else rotatedMarker = _rejectedCorners.getMat(j);

            // last filter, check if inner code is close enough to the assigned marker code
            int codeDistance = 0;
            // if errorCorrectionRate, dont check code
            if(refineParams.errorCorrectionRate >= 0) {

                // extract bits
                Mat bits = _extractBits(
                    grey, rotatedMarker, dictionary.markerSize, detectorParams.markerBorderBits,
                    detectorParams.perspectiveRemovePixelPerCell,
                    detectorParams.perspectiveRemoveIgnoredMarginPerCell, detectorParams.minOtsuStdDev);

                Mat onlyBits =
                    bits.rowRange(detectorParams.markerBorderBits, bits.rows - detectorParams.markerBorderBits)
                        .colRange(detectorParams.markerBorderBits, bits.rows - detectorParams.markerBorderBits);

                codeDistance =
                    dictionary.getDistanceToId(onlyBits, undetectedMarkersIds[i], false);
            }

            // if everythin is ok, assign values to current best match
            if(refineParams.errorCorrectionRate < 0 || codeDistance < maxCorrectionRecalculated) {
                closestCandidateIdx = j;
                closestCandidateDistance = minDistance;
                closestRotatedMarker = rotatedMarker;
            }
        }

        // if at least one good match, we have rescue the missing marker
        if(closestCandidateIdx >= 0) {

            // subpixel refinement
            if(detectorParams.cornerRefinementMethod == (int)CORNER_REFINE_SUBPIX) {
                CV_Assert(detectorParams.cornerRefinementWinSize > 0 &&
                          detectorParams.cornerRefinementMaxIterations > 0 &&
                          detectorParams.cornerRefinementMinAccuracy > 0);

                std::vector<Point2f> marker(closestRotatedMarker.begin<Point2f>(), closestRotatedMarker.end<Point2f>());
                int cornerRefinementWinSize = std::max(1, cvRound(detectorParams.relativeCornerRefinmentWinSize*
                                              getAverageModuleSize(marker, dictionary.markerSize, detectorParams.markerBorderBits)));
                cornerRefinementWinSize = min(cornerRefinementWinSize, detectorParams.cornerRefinementWinSize);
                cornerSubPix(grey, closestRotatedMarker,
                             Size(cornerRefinementWinSize, cornerRefinementWinSize),
                             Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS,
                                                        detectorParams.cornerRefinementMaxIterations,
                                                        detectorParams.cornerRefinementMinAccuracy));
            }

            // remove from rejected
            alreadyIdentified[closestCandidateIdx] = true;

            // add to detected
            finalAcceptedCorners.push_back(closestRotatedMarker);
            finalAcceptedIds.push_back(undetectedMarkersIds[i]);

            // add the original index of the candidate
            recoveredIdxs.push_back(closestCandidateIdx);
        }
    }

    // parse output
    if(finalAcceptedIds.size() != _detectedIds.total()) {
        // parse output
        Mat(finalAcceptedIds).copyTo(_detectedIds);
        _copyVector2Output(finalAcceptedCorners, _detectedCorners);

        // recalculate _rejectedCorners based on alreadyIdentified
        vector<vector<Point2f> > finalRejected;
        for(unsigned int i = 0; i < alreadyIdentified.size(); i++) {
            if(!alreadyIdentified[i]) {
                finalRejected.push_back(_rejectedCorners.getMat(i).clone());
            }
        }
        _copyVector2Output(finalRejected, _rejectedCorners);

        if(_recoveredIdxs.needed()) {
            Mat(recoveredIdxs).copyTo(_recoveredIdxs);
        }
    }
}

void ArucoDetector::write(FileStorage &fs) const {
    // preserve old format for single dictionary case
    if (1 == arucoDetectorImpl->dictionaries.size()) {
        arucoDetectorImpl->dictionaries[0].writeDictionary(fs);
    } else {
        fs << "dictionaries" << "[";
        for (auto& dictionary : arucoDetectorImpl->dictionaries) {
            fs << "{";
            dictionary.writeDictionary(fs);
            fs << "}";
        }
        fs << "]";
    }
    arucoDetectorImpl->detectorParams.writeDetectorParameters(fs);
    arucoDetectorImpl->refineParams.writeRefineParameters(fs);
}

void ArucoDetector::read(const FileNode &fn) {
    arucoDetectorImpl->dictionaries.clear();
    if (!fn.empty() && !fn["dictionaries"].empty() && fn["dictionaries"].isSeq()) {
        for (const auto& dictionaryNode : fn["dictionaries"]) {
            arucoDetectorImpl->dictionaries.emplace_back();
            arucoDetectorImpl->dictionaries.back().readDictionary(dictionaryNode);
        }
    } else {
        // backward compatibility
        arucoDetectorImpl->dictionaries.emplace_back();
        arucoDetectorImpl->dictionaries.back().readDictionary(fn);
    }
    arucoDetectorImpl->detectorParams.readDetectorParameters(fn);
    arucoDetectorImpl->refineParams.readRefineParameters(fn);
}

const Dictionary& ArucoDetector::getDictionary() const {
    return arucoDetectorImpl->dictionaries[0];
}

void ArucoDetector::setDictionary(const Dictionary& dictionary) {
    if (arucoDetectorImpl->dictionaries.empty()) {
        arucoDetectorImpl->dictionaries.push_back(dictionary);
    } else {
        arucoDetectorImpl->dictionaries[0] = dictionary;
    }
}

vector<Dictionary> ArucoDetector::getDictionaries() const {
    return arucoDetectorImpl->dictionaries;
}

void ArucoDetector::setDictionaries(const vector<Dictionary>& dictionaries) {
    CV_Assert(!dictionaries.empty());
    arucoDetectorImpl->dictionaries = dictionaries;
}

const DetectorParameters& ArucoDetector::getDetectorParameters() const {
    return arucoDetectorImpl->detectorParams;
}

void ArucoDetector::setDetectorParameters(const DetectorParameters& detectorParameters) {
    arucoDetectorImpl->detectorParams = detectorParameters;
}

const RefineParameters& ArucoDetector::getRefineParameters() const {
    return arucoDetectorImpl->refineParams;
}

void ArucoDetector::setRefineParameters(const RefineParameters& refineParameters) {
    arucoDetectorImpl->refineParams = refineParameters;
}

void drawDetectedMarkers(InputOutputArray _image, InputArrayOfArrays _corners,
                         InputArray _ids, Scalar borderColor) {
    CV_Assert(_image.getMat().total() != 0 &&
              (_image.getMat().channels() == 1 || _image.getMat().channels() == 3));
    CV_Assert((_corners.total() == _ids.total()) || _ids.total() == 0);

    // calculate colors
    Scalar textColor, cornerColor;
    textColor = cornerColor = borderColor;
    swap(textColor.val[0], textColor.val[1]);     // text color just sawp G and R
    swap(cornerColor.val[1], cornerColor.val[2]); // corner color just sawp G and B

    int nMarkers = (int)_corners.total();
    for(int i = 0; i < nMarkers; i++) {
        Mat currentMarker = _corners.getMat(i);
        CV_Assert(currentMarker.total() == 4 && currentMarker.channels() == 2);
        if (currentMarker.type() != CV_32SC2)
            currentMarker.convertTo(currentMarker, CV_32SC2);

        // draw marker sides
        for(int j = 0; j < 4; j++) {
            Point p0, p1;
            p0 = currentMarker.ptr<Point>(0)[j];
            p1 = currentMarker.ptr<Point>(0)[(j + 1) % 4];
            line(_image, p0, p1, borderColor, 1);
        }
        // draw first corner mark
        rectangle(_image, currentMarker.ptr<Point>(0)[0] - Point(3, 3),
                  currentMarker.ptr<Point>(0)[0] + Point(3, 3), cornerColor, 1, LINE_AA);

        // draw ID
        if(_ids.total() != 0) {
            Point cent(0, 0);
            for(int p = 0; p < 4; p++)
                cent += currentMarker.ptr<Point>(0)[p];
            cent = cent / 4.;
            stringstream s;
            s << "id=" << _ids.getMat().ptr<int>(0)[i];
            putText(_image, s.str(), cent, FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2);
        }
    }
}

void generateImageMarker(const Dictionary &dictionary, int id, int sidePixels, OutputArray _img, int borderBits) {
    dictionary.generateImageMarker(id, sidePixels, _img, borderBits);
}

}
}
