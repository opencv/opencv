#ifndef CVVISUAL_MATCH_CALL_HPP
#define CVVISUAL_MATCH_CALL_HPP

#include <vector>
#include <utility>
#include <type_traits>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "call.hpp"

namespace cvv
{
namespace impl
{

/**
 * Contains all the calldata (= location, images and their keypoints).
 */
class MatchCall : public Call
{
      public:
	/**
	 * @brief Constructs a MatchCall.
	 */
	MatchCall(cv::InputArray img1, std::vector<cv::KeyPoint> keypoints1,
	          cv::InputArray img2, std::vector<cv::KeyPoint> keypoints2,
	          std::vector<cv::DMatch> matches, impl::CallMetaData data,
	          QString type, QString description, QString requestedView,
	          bool useTrainDescriptor);

	size_t matrixCount() const override
	{
		return 2;
	}
	const cv::Mat &matrixAt(size_t index) const override;

	/**
	 * @brief Returns the first Mat.
	 */
	const cv::Mat &img1() const
	{
		return img1_;
	}

	/**
	 * @brief Returns the second Mat.
	 */
	const cv::Mat &img2() const
	{
		return img2_;
	}

	/**
	 * @brief Returns the keypoints for the first Mat.
	 */
	const std::vector<cv::KeyPoint> &keyPoints1() const
	{
		return keypoints1_;
	}

	/**
	 * @brief Returns the keypoints for the second Mat.
	 */
	const std::vector<cv::KeyPoint> &keyPoints2() const
	{
		return keypoints2_;
	}

	/**
	 * @brief Returns the matches.
	 */
	const std::vector<cv::DMatch> &matches() const
	{
		return matches_;
	}

	bool usesTrainDescriptor() const
	{
		return usesTrainDescriptor_;
	}

      private:
	cv::Mat img1_;
	std::vector<cv::KeyPoint> keypoints1_;
	cv::Mat img2_;
	std::vector<cv::KeyPoint> keypoints2_;
	std::vector<cv::DMatch> matches_;
	bool usesTrainDescriptor_;
};

/**
 * Constructs a MatchCall and adds it to the global data-controller.
 */
void debugMatchCall(cv::InputArray img1, std::vector<cv::KeyPoint> keypoints1,
                    cv::InputArray img2, std::vector<cv::KeyPoint> keypoints2,
                    std::vector<cv::DMatch> matches, const CallMetaData &data,
                    const char *description, const char *view,
                    bool useTrainDescriptor);
}
} // namespaces cvv::impl

#endif
