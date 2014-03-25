#include "../../src/qtutil/util.hpp"

#include <iostream>
#include <typeinfo>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

template <int depth> void test(int w, int h, int threads)
{
	using c = std::chrono::high_resolution_clock;

	cv::Mat mat{ w, h, depth, cv::Scalar{ 0, 0, 0, 0 } };
	std::cout << "##test: \n";
	std::cout << "depth " << depth << "\t"; // typeid();
	std::cout << "pixel: " << ((w * h) / 1000000) << " M pixels\t";
	auto start = c::now();

	auto res = cvv::qtutil::convertMatToQImage(mat, false, threads);

	auto end = c::now();
	auto elapsed = end - start;

	std::cout << "success: "
		  << (res.first == cvv::qtutil::ImageConversionResult::SUCCESS)
		  << "\t";

	std::cout << "time: " << ((elapsed.count() * c::period::num) /
				  (c::period::den / 1000)) << "\n";
}

/**
 * @brief
 * - "qt allows images with: 1000 M pixels" will be printed if qt allows the craetion of
 *   1 G pixel images
 * - "# threads?:" will be printed and the user has to enter the number of threads to use
 * - the following text will be printed where XXXX is the time needed to execute (in ms)
	##test:
	depth 30	pixel: 1 M pixels	success: 1	time: XXXX
	##test:
	depth 28	pixel: 1 M pixels	success: 1	time: XXXX
	##test:
	depth 30	pixel: 10 M pixels	success: 1	time: XXXX
	##test:
	depth 28	pixel: 10 M pixels	success: 1	time: XXXX
	continue with 100M pixel test? will need about 3,2 gig mem (y)
 * - if the user enters anything other than y the programm will exit
 * - if y was entered the following text will be printed where XXXX is the
 *  time needed to execute (in ms)
	##test:
	depth 30	pixel: 100 M pixels	success: 1	time: XXXX
	##test:
	depth 28	pixel: 100 M pixels	success: 1	time: XXXX
 */
int main()
{
	try
	{
		{
			// test wheather qimage is allowed to be large
			int w = 100000;
			int h = 10000;
			QImage i{ w, h, QImage::Format_ARGB32 };
			i.fill(99);
			std::cout << "qt allows images with: "
				  << ((w * h) / 1000000) << " M pixels\n";
		}
		int threads;
		std::cout << "# threads?:\n";
		std::cin >> threads;
		// 1M
		test<CV_64FC4>(1000, 1000, threads);
		test<CV_32SC4>(1000, 1000, threads);
		// 10M
		test<CV_64FC4>(10000, 1000, threads);
		test<CV_32SC4>(10000, 1000, threads);

		// 100M
		char c;
		std::cout << "continue with 100M pixel test? will need about "
			     "3,2 gig mem (y)\n";
		std::cin >> c;
		if (c != 'y')
		{
			return 0;
		};
		test<CV_64FC4>(10000, 10000, threads);
		test<CV_32SC4>(10000, 10000, threads);
		// 1000M
		// test<CV_64FC4>(100000,10000,threads);
		// test<CV_32SC4>(100000,10000,threads);
		return 0;
	}
	catch (const std::bad_alloc &)
	{
		std::cout << "Out of mem\n";
	};
	return 0;
}
