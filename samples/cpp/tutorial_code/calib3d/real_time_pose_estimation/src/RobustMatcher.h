/*
 * RobustMatcher.h
 *
 *  Created on: Jun 4, 2014
 *      Author: eriba
 */

#ifndef ROBUSTMATCHER_H_
#define ROBUSTMATCHER_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

class RobustMatcher {
public:
    RobustMatcher() : detector_(), extractor_(), matcher_(),
        ratio_(0.8f), training_img_(), img_matching_()
    {
        // ORB is the default feature
        detector_ = cv::ORB::create();
        extractor_ = cv::ORB::create();

        // BruteFroce matcher with Norm Hamming is the default matcher
        matcher_ = cv::makePtr<cv::BFMatcher>((int)cv::NORM_HAMMING, false);

    }
    virtual ~RobustMatcher();

    // Set the feature detector
    void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detect) {  detector_ = detect; }

    // Set the descriptor extractor
    void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor>& desc) { extractor_ = desc; }

    // Set the matcher
    void setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher>& match) {  matcher_ = match; }

    // Compute the keypoints of an image
    void computeKeyPoints( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

    // Compute the descriptors of an image given its keypoints
    void computeDescriptors( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    cv::Mat getImageMatching() const { return img_matching_; }

    // Set ratio parameter for the ratio test
    void setRatio( float rat) { ratio_ = rat; }

    void setTrainingImage(const cv::Mat &img) { training_img_ = img; }

    // Clear matches for which NN ratio is > than threshold
    // return the number of removed points
    // (corresponding entries being cleared,
    // i.e. size will be 0)
    int ratioTest(std::vector<std::vector<cv::DMatch> > &matches);

    // Insert symmetrical matches in symMatches vector
    void symmetryTest( const std::vector<std::vector<cv::DMatch> >& matches1,
                       const std::vector<std::vector<cv::DMatch> >& matches2,
                       std::vector<cv::DMatch>& symMatches );

    // Match feature points using ratio and symmetry test
    void robustMatch( const cv::Mat& frame, std::vector<cv::DMatch>& good_matches,
                      std::vector<cv::KeyPoint>& keypoints_frame,
                      const cv::Mat& descriptors_model,
                      const std::vector<cv::KeyPoint>& keypoints_model);

    // Match feature points using ratio test
    void fastRobustMatch( const cv::Mat& frame, std::vector<cv::DMatch>& good_matches,
                          std::vector<cv::KeyPoint>& keypoints_frame,
                          const cv::Mat& descriptors_model,
                          const std::vector<cv::KeyPoint>& keypoints_model);

private:
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector_;
    // pointer to the feature descriptor extractor object
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    // pointer to the matcher object
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    // max ratio between 1st and 2nd NN
    float ratio_;
    // training image
    cv::Mat training_img_;
    // matching image
    cv::Mat img_matching_;
};

#endif /* ROBUSTMATCHER_H_ */
