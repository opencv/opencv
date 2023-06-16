// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/barcode.hpp"
#include <set>

using namespace std;

namespace opencv_test{namespace{

typedef std::set<string> StringSet;

// Convert ';'-separated strings to a set
inline static StringSet toSet(const string &line)
{
    StringSet res;
    string::size_type it = 0, ti;
    while (true)
    {
        ti = line.find(';', it);
        if (ti == string::npos)
        {
            res.insert(string(line, it, line.size() - it));
            break;
        }
        res.insert(string(line, it, ti - it));
        it = ti + 1;
    }
    return res;
}

// Convert vector of strings to a set
inline static StringSet toSet(const vector<string> &lines)
{
    StringSet res;
    for (const string & line : lines)
        res.insert(line);
    return res;
}

// Get all keys of a map in a vector
template<typename T, typename V>
inline static vector<T> getKeys(const map<T, V> &m)
{
    vector<T> res;
    for (const auto & it : m)
        res.push_back(it.first);
    return res;
}

struct BarcodeResult
{
    string type;
    string data;
};

map<string, BarcodeResult> testResults {
    { "single/book.jpg", {"EAN_13", "9787115279460"} },
    { "single/bottle_1.jpg", {"EAN_13", "6922255451427"} },
    { "single/bottle_2.jpg", {"EAN_13", "6921168509256"} },
    { "multiple/4_barcodes.jpg", {"EAN_13;EAN_13;EAN_13;EAN_13", "9787564350840;9783319200064;9787118081473;9787122276124"} }
};

typedef testing::TestWithParam< string > BarcodeDetector_main;

TEST_P(BarcodeDetector_main, interface)
{
    const string fname = GetParam();
    const string image_path = findDataFile(string("barcode/") + fname);
    const StringSet expected_lines = toSet(testResults[fname].data);
    const StringSet expected_types = toSet(testResults[fname].type);
    const size_t expected_count = expected_lines.size(); // assume codes are unique
    // TODO: verify points location

    Mat img = imread(image_path);
    ASSERT_FALSE(img.empty()) << "Can't read image: " << image_path;

    barcode::BarcodeDetector det;
    vector<Point2f> points;
    vector<string> types;
    vector<string> lines;

    // common interface (single)
    {
        bool res = det.detect(img, points);
        ASSERT_TRUE(res);
        EXPECT_EQ(expected_count * 4, points.size());
    }

    {
        string res = det.decode(img, points);
        ASSERT_FALSE(res.empty());
        EXPECT_EQ(1u, expected_lines.count(res));
    }

    // common interface (multi)
    {
        bool res = det.detectMulti(img, points);
        ASSERT_TRUE(res);
        EXPECT_EQ(expected_count * 4, points.size());
    }

    {
        bool res = det.decodeMulti(img, points, lines);
        ASSERT_TRUE(res);
        EXPECT_EQ(expected_lines, toSet(lines));
    }

    // specific interface
    {
        bool res = det.decodeWithType(img, points, lines, types);
        ASSERT_TRUE(res);
        EXPECT_EQ(expected_types, toSet(types));
        EXPECT_EQ(expected_lines, toSet(lines));
    }

    {
        bool res = det.detectAndDecodeWithType(img, lines, types, points);
        ASSERT_TRUE(res);
        EXPECT_EQ(expected_types, toSet(types));
        EXPECT_EQ(expected_lines, toSet(lines));
    }
}

INSTANTIATE_TEST_CASE_P(/**/, BarcodeDetector_main, testing::ValuesIn(getKeys(testResults)));

TEST(BarcodeDetector_base, invalid)
{
    auto bardet = barcode::BarcodeDetector();
    std::vector<Point> corners;
    vector<cv::String> decoded_info;
    Mat zero_image = Mat::zeros(256, 256, CV_8UC1);
    EXPECT_FALSE(bardet.detectMulti(zero_image, corners));
    corners = std::vector<Point>(4);
    EXPECT_ANY_THROW(bardet.decodeMulti(zero_image, corners, decoded_info));
}

}} // opencv_test::<anonymous>::
