#include "test_precomp.hpp"
#include "opencv2/photo.hpp"
#include <string>

using namespace cv;
using namespace std;

#ifdef DUMP_RESULTS
#  define DUMP(image, path) imwrite(path, image)
#else
#  define DUMP(image, path)
#endif


TEST(Photo_Decolor, regression)
{
        string folder = string(cvtest::TS::ptr()->get_data_path()) + "decolor/";
        string original_path = folder + "color_image_1.png";
        string expected_path1 = folder + "grayscale_image_1.png";
        string expected_path2 = folder + "color_boost_image_1.png";

        Mat original = imread(original_path, IMREAD_COLOR);
        Mat expected1 = imread(expected_path1, IMREAD_GRAYSCALE);
        Mat expected2 = imread(expected_path2, IMREAD_COLOR);

        ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
        ASSERT_FALSE(expected1.empty()) << "Could not load reference image " << expected_path1;
        ASSERT_FALSE(expected2.empty()) << "Could not load reference image " << expected_path2;

        Mat grayscale, color_boost;
        decolor(original, grayscale, color_boost);

        DUMP(grayscale, expected_path1 + ".grayscale.png");
        DUMP(color_boost, expected_path2 + ".color_boost.png");

        ASSERT_EQ(0, norm(grayscale != expected1));
        ASSERT_EQ(0, norm(color_boost != expected2));
}

