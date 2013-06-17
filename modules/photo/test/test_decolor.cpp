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
        string expected_path = folder + "grayscale_image_1.png";

        Mat original = imread(original_path, IMREAD_COLOR);
        Mat expected = imread(expected_path, IMREAD_GRAYSCALE);

        ASSERT_FALSE(original.empty()) << "Could not load input image " << original_path;
        ASSERT_FALSE(expected.empty()) << "Could not load reference image " << expected_path;

        Mat result;
        decolor(original, result);

        DUMP(result, expected_path + ".res.png");

        ASSERT_EQ(0, norm(result != expected));
}

