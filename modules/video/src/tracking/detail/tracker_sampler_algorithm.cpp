// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "../../precomp.hpp"
#include "opencv2/video/detail/tracking.private.hpp"

namespace cv {
namespace detail {
inline namespace tracking {

TrackerSamplerAlgorithm::~TrackerSamplerAlgorithm()
{
}

bool TrackerSamplerAlgorithm::sampling(const Mat& image, Rect boundingBox, std::vector<Mat>& sample)
{
    if (image.empty())
        return false;

    return samplingImpl(image, boundingBox, sample);
}

Ptr<TrackerSamplerAlgorithm> TrackerSamplerAlgorithm::create(const String& trackerSamplerType)
{
    if (trackerSamplerType.find("CSC") == 0)
    {
        return Ptr<TrackerSamplerCSC>(new TrackerSamplerCSC());
    }

    if (trackerSamplerType.find("CS") == 0)
    {
        return Ptr<TrackerSamplerCS>(new TrackerSamplerCS());
    }

    CV_Error(-1, "Tracker sampler algorithm type not supported");
}

String TrackerSamplerAlgorithm::getClassName() const
{
    return className;
}



TrackerSamplerCSC::Params::Params()
{
    initInRad = 3;
    initMaxNegNum = 65;
    searchWinSize = 25;
    trackInPosRad = 4;
    trackMaxNegNum = 65;
    trackMaxPosNum = 100000;
}

TrackerSamplerCSC::TrackerSamplerCSC(const TrackerSamplerCSC::Params& parameters)
    : params(parameters)
{
    className = "CSC";
    mode = MODE_INIT_POS;
    rng = theRNG();
}

TrackerSamplerCSC::~TrackerSamplerCSC()
{
}

bool TrackerSamplerCSC::samplingImpl(const Mat& image, Rect boundingBox, std::vector<Mat>& sample)
{
    float inrad = 0;
    float outrad = 0;
    int maxnum = 0;

    switch (mode)
    {
    case MODE_INIT_POS:
        inrad = params.initInRad;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad);
        break;
    case MODE_INIT_NEG:
        inrad = 2.0f * params.searchWinSize;
        outrad = 1.5f * params.initInRad;
        maxnum = params.initMaxNegNum;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum);
        break;
    case MODE_TRACK_POS:
        inrad = params.trackInPosRad;
        outrad = 0;
        maxnum = params.trackMaxPosNum;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum);
        break;
    case MODE_TRACK_NEG:
        inrad = 1.5f * params.searchWinSize;
        outrad = params.trackInPosRad + 5;
        maxnum = params.trackMaxNegNum;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad, outrad, maxnum);
        break;
    case MODE_DETECT:
        inrad = params.searchWinSize;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad);
        break;
    default:
        inrad = params.initInRad;
        sample = sampleImage(image, boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height, inrad);
        break;
    }
    return false;
}

void TrackerSamplerCSC::setMode(int samplingMode)
{
    mode = samplingMode;
}

std::vector<Mat> TrackerSamplerCSC::sampleImage(const Mat& img, int x, int y, int w, int h, float inrad, float outrad, int maxnum)
{
    int rowsz = img.rows - h - 1;
    int colsz = img.cols - w - 1;
    float inradsq = inrad * inrad;
    float outradsq = outrad * outrad;
    int dist;

    uint minrow = max(0, (int)y - (int)inrad);
    uint maxrow = min((int)rowsz - 1, (int)y + (int)inrad);
    uint mincol = max(0, (int)x - (int)inrad);
    uint maxcol = min((int)colsz - 1, (int)x + (int)inrad);

    //fprintf(stderr,"inrad=%f minrow=%d maxrow=%d mincol=%d maxcol=%d\n",inrad,minrow,maxrow,mincol,maxcol);

    std::vector<Mat> samples;
    samples.resize((maxrow - minrow + 1) * (maxcol - mincol + 1));
    int i = 0;

    float prob = ((float)(maxnum)) / samples.size();

    for (int r = minrow; r <= int(maxrow); r++)
        for (int c = mincol; c <= int(maxcol); c++)
        {
            dist = (y - r) * (y - r) + (x - c) * (x - c);
            if (float(rng.uniform(0.f, 1.f)) < prob && dist < inradsq && dist >= outradsq)
            {
                samples[i] = img(Rect(c, r, w, h));
                i++;
            }
        }

    samples.resize(min(i, maxnum));
    return samples;
};

/**
 * TrackerSamplerCS
 */
TrackerSamplerCS::Params::Params()
{
    overlap = 0.99f;
    searchFactor = 2;
}

TrackerSamplerCS::TrackerSamplerCS(const TrackerSamplerCS::Params& parameters)
    : params(parameters)
{
    className = "CS";
    mode = MODE_POSITIVE;
}

void TrackerSamplerCS::setMode(int samplingMode)
{
    mode = samplingMode;
}

TrackerSamplerCS::~TrackerSamplerCS()
{
}

bool TrackerSamplerCS::samplingImpl(const Mat& image, Rect boundingBox, std::vector<Mat>& sample)
{

    trackedPatch = boundingBox;
    Size imageSize(image.cols, image.rows);
    validROI = Rect(0, 0, imageSize.width, imageSize.height);

    Size trackedPatchSize(trackedPatch.width, trackedPatch.height);
    Rect trackingROI = getTrackingROI(params.searchFactor);

    sample = patchesRegularScan(image, trackingROI, trackedPatchSize);

    return true;
}

Rect TrackerSamplerCS::getTrackingROI(float searchFactor)
{
    Rect searchRegion;

    searchRegion = RectMultiply(trackedPatch, searchFactor);
    //check
    if (searchRegion.y + searchRegion.height > validROI.height)
        searchRegion.height = validROI.height - searchRegion.y;
    if (searchRegion.x + searchRegion.width > validROI.width)
        searchRegion.width = validROI.width - searchRegion.x;

    return searchRegion;
}

Rect TrackerSamplerCS::RectMultiply(const Rect& rect, float f)
{
    cv::Rect r_tmp;
    r_tmp.y = (int)(rect.y - ((float)rect.height * f - rect.height) / 2);
    if (r_tmp.y < 0)
        r_tmp.y = 0;
    r_tmp.x = (int)(rect.x - ((float)rect.width * f - rect.width) / 2);
    if (r_tmp.x < 0)
        r_tmp.x = 0;
    r_tmp.height = (int)(rect.height * f);
    r_tmp.width = (int)(rect.width * f);

    return r_tmp;
}

Rect TrackerSamplerCS::getROI() const
{
    return ROI;
}

void TrackerSamplerCS::setCheckedROI(Rect imageROI)
{
    int dCol, dRow;
    dCol = imageROI.x - validROI.x;
    dRow = imageROI.y - validROI.y;
    ROI.y = (dRow < 0) ? validROI.y : imageROI.y;
    ROI.x = (dCol < 0) ? validROI.x : imageROI.x;
    dCol = imageROI.x + imageROI.width - (validROI.x + validROI.width);
    dRow = imageROI.y + imageROI.height - (validROI.y + validROI.height);
    ROI.height = (dRow > 0) ? validROI.height + validROI.y - ROI.y : imageROI.height + imageROI.y - ROI.y;
    ROI.width = (dCol > 0) ? validROI.width + validROI.x - ROI.x : imageROI.width + imageROI.x - ROI.x;
}

std::vector<Mat> TrackerSamplerCS::patchesRegularScan(const Mat& image, Rect trackingROI, Size patchSize)
{
    std::vector<Mat> sample;
    if ((validROI == trackingROI))
        ROI = trackingROI;
    else
        setCheckedROI(trackingROI);

    if (mode == MODE_POSITIVE)
    {
        int num = 4;
        sample.resize(num);
        Mat singleSample = image(trackedPatch);
        for (int i = 0; i < num; i++)
            sample[i] = singleSample;
        return sample;
    }

    int stepCol = (int)floor((1.0f - params.overlap) * (float)patchSize.width + 0.5f);
    int stepRow = (int)floor((1.0f - params.overlap) * (float)patchSize.height + 0.5f);
    if (stepCol <= 0)
        stepCol = 1;
    if (stepRow <= 0)
        stepRow = 1;

    Size m_patchGrid;
    Rect m_rectUpperLeft;
    Rect m_rectUpperRight;
    Rect m_rectLowerLeft;
    Rect m_rectLowerRight;
    int num;

    m_patchGrid.height = ((int)((float)(ROI.height - patchSize.height) / stepRow) + 1);
    m_patchGrid.width = ((int)((float)(ROI.width - patchSize.width) / stepCol) + 1);

    num = m_patchGrid.width * m_patchGrid.height;
    sample.resize(num);
    int curPatch = 0;

    m_rectUpperLeft = m_rectUpperRight = m_rectLowerLeft = m_rectLowerRight = cv::Rect(0, 0, patchSize.width, patchSize.height);
    m_rectUpperLeft.y = ROI.y;
    m_rectUpperLeft.x = ROI.x;
    m_rectUpperRight.y = ROI.y;
    m_rectUpperRight.x = ROI.x + ROI.width - patchSize.width;
    m_rectLowerLeft.y = ROI.y + ROI.height - patchSize.height;
    m_rectLowerLeft.x = ROI.x;
    m_rectLowerRight.y = ROI.y + ROI.height - patchSize.height;
    m_rectLowerRight.x = ROI.x + ROI.width - patchSize.width;

    if (mode == MODE_NEGATIVE)
    {
        int numSamples = 4;
        sample.resize(numSamples);
        sample[0] = image(m_rectUpperLeft);
        sample[1] = image(m_rectUpperRight);
        sample[2] = image(m_rectLowerLeft);
        sample[3] = image(m_rectLowerRight);
        return sample;
    }

    int numPatchesX;
    int numPatchesY;

    numPatchesX = 0;
    numPatchesY = 0;
    for (int curRow = 0; curRow < ROI.height - patchSize.height + 1; curRow += stepRow)
    {
        numPatchesY++;

        for (int curCol = 0; curCol < ROI.width - patchSize.width + 1; curCol += stepCol)
        {
            if (curRow == 0)
                numPatchesX++;

            Mat singleSample = image(Rect(curCol + ROI.x, curRow + ROI.y, patchSize.width, patchSize.height));
            sample[curPatch] = singleSample;
            curPatch++;
        }
    }

    CV_Assert(curPatch == num);

    return sample;
}

#if 0
TrackerSamplerPF::Params::Params(){
    iterationNum=20;
    particlesNum=100;
    alpha=0.9;
    std=(Mat_<double>(1,4)<<15.0,15.0,15.0,15.0);
}
TrackerSamplerPF::TrackerSamplerPF(const Mat& chosenRect,const TrackerSamplerPF::Params &parameters):
    params( parameters ),_function(new TrackingFunctionPF(chosenRect)){
        className="PF";
        _solver=createPFSolver(_function,parameters.std,TermCriteria(TermCriteria::MAX_ITER,parameters.iterationNum,0.0),
        parameters.particlesNum,parameters.alpha);
}
bool TrackerSamplerPF::samplingImpl( const Mat& image, Rect boundingBox, std::vector<Mat>& sample ){
    Ptr<TrackerTargetState> ptr;
    Mat_<double> _last_guess=(Mat_<double>(1,4)<<(double)boundingBox.x,(double)boundingBox.y,
    (double)boundingBox.x+boundingBox.width,(double)boundingBox.y+boundingBox.height);
    PFSolver* promoted_solver=dynamic_cast<PFSolver*>(static_cast<MinProblemSolver*>(_solver));

    promoted_solver->setParamsSTD(params.std);
    promoted_solver->minimize(_last_guess);
    dynamic_cast<TrackingFunctionPF*>(static_cast<MinProblemSolver::Function*>(promoted_solver->getFunction()))->update(image);
    while(promoted_solver->iteration() <= promoted_solver->getTermCriteria().maxCount);
    promoted_solver->getOptParam(_last_guess);

    Rect res=Rect(Point_<int>((int)_last_guess(0,0),(int)_last_guess(0,1)),Point_<int>((int)_last_guess(0,2),(int)_last_guess(0,3)));
    sample.clear();
    sample.push_back(image(res));
    return true;
}
#endif

}}}  // namespace cv::detail::tracking
