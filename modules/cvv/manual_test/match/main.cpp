#include <iostream>
#include <random>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "dmatch.hpp"
#include "final_show.hpp"

std::vector<cv::KeyPoint> makeRandomKeys(size_t x, size_t y, size_t n)
{
	static std::mt19937_64 gen{ std::random_device{}() };
	std::uniform_real_distribution<float> xdist{ 0.0f,
		                                     static_cast<float>(x) };
	std::uniform_real_distribution<float> ydist{ 0.0f,
		                                     static_cast<float>(y) };
	std::uniform_real_distribution<float> sdist{ 0.0f, 3.0f };
	std::vector<cv::KeyPoint> keypoints;
	for (size_t i = 0; i < n; ++i)
	{
		keypoints.emplace_back(xdist(gen), ydist(gen), sdist(gen));
	}
	return keypoints;
}
std::vector<cv::KeyPoint> scaleDown(const std::vector<cv::KeyPoint> &in,
                                    size_t x, size_t y, float factor)
{
	std::vector<cv::KeyPoint> points;
	points.reserve(in.size());

	for (const auto &point : in)
	{
		auto newX = (x * (1.0f - factor) / 2) + factor * point.pt.x;
		auto newY = (y * (1.0f - factor) / 2) + factor * point.pt.y;
		points.emplace_back(newX, newY, point.size);
	}

	return points;
}

/**
 * @brief This test creates random matches of random quality on all the provided images and scales
 * them down for a copy of the same image.
 * 
 * Expected behaviour:
 * * A Mainwindow should open that shows an overview-table containing matches from the first image to itself
 * * Upon klicking step multiple times or '>>' once, all further images should appear in the table.
 * * All calls can be opened in any existing window or in a new one. It is possible to select all the
 *   different match-visualisations for all of them.
 * * Closing calltabs should work. Closing the last tab of a window results in the closing of the window
 * * Clicking the Close-button results in the termination of the program with 0 as exit-status.
 */
int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << argv[0]
		          << " must recieve one or more files as arguments\n";
		return 1;
	}
	for (int i = 1; i < argc; ++i)
	{
		auto src = cv::imread(argv[i]);
		const size_t keypointCount = 500;
		auto keypoints1 =
		    makeRandomKeys(src.cols, src.rows, keypointCount);
		auto keypoints2 =
		    scaleDown(keypoints1, src.cols, src.rows, 0.8f);

		std::vector<cv::DMatch> match;
		std::mt19937_64 gen{ std::random_device{}() };
		std::uniform_real_distribution<float> dist{ 0.0f, 1.0f };
		for (size_t i = 0; i < keypointCount; i++)
		{
			match.emplace_back(i, i, dist(gen));
		}

		cvv::debugDMatch(src, keypoints1, src, keypoints2, match,
		                 CVVISUAL_LOCATION);
	}
	cvv::finalShow();
}
