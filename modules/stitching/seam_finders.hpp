#ifndef __OPENCV_SEAM_FINDERS_HPP__
#define __OPENCV_SEAM_FINDERS_HPP__

#include <vector>
#include <opencv2/core/core.hpp>

class SeamFinder
{
public:
    enum { NO, VORONOI, GRAPH_CUT };
    static cv::Ptr<SeamFinder> createDefault(int type);

    virtual ~SeamFinder() {}
    virtual void find(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners,
                      std::vector<cv::Mat> &masks) = 0;
};


class NoSeamFinder : public SeamFinder
{
public:
    void find(const std::vector<cv::Mat>&, const std::vector<cv::Point>&, std::vector<cv::Mat>&) {}
};


class PairwiseSeamFinder : public SeamFinder
{
public:
    void find(const std::vector<cv::Mat> &src, const std::vector<cv::Point> &corners,
              std::vector<cv::Mat> &masks);
protected:
    virtual void findInPair(const cv::Mat &img1, const cv::Mat &img2, cv::Point tl1, cv::Point tl2,
                            cv::Rect roi, cv::Mat &mask1, cv::Mat &mask2) = 0;
};


class VoronoiSeamFinder : public PairwiseSeamFinder
{
private:
    void findInPair(const cv::Mat &img1, const cv::Mat &img2, cv::Point tl1, cv::Point tl2,
                    cv::Rect roi, cv::Mat &mask1, cv::Mat &mask2);
};


class GraphCutSeamFinder : public PairwiseSeamFinder
{
public:
    // TODO add COST_COLOR_GRAD support
    enum { COST_COLOR };
    GraphCutSeamFinder(int cost_type = COST_COLOR, float terminal_cost = 10000.f,
                       float bad_region_penalty = 1000.f);

private:
    void findInPair(const cv::Mat &img1, const cv::Mat &img2, cv::Point tl1, cv::Point tl2,
                    cv::Rect roi, cv::Mat &mask1, cv::Mat &mask2);

    class Impl;
    cv::Ptr<Impl> impl_;
};

#endif // __OPENCV_SEAM_FINDERS_HPP__
