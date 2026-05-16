// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/aruco2.hpp"
#include "opencv2/imgproc.hpp"

namespace opencv_test { namespace {

using namespace cv;
using namespace cv::aruco2;

TEST(Objdetect_Aruco2, Generation) {
    Mat img;
    int id = 42;
    unsigned int bitSize = 10;
    DictionaryType dict = DICT_ARUCO_MIP_36h12;
    
    // 6x6 bits + 2 border bits = 8 bits per side
    // If externalBorder is true, it adds another bit? Let's check the code or just see the result.
    // Standard aruco markers have 1 bit border.
    // Let's see what generateMarkerImage does.
    generateMarkerImage(img, dict, id, bitSize, true);
    
    ASSERT_FALSE(img.empty());
    ASSERT_EQ(img.type(), CV_8UC1);
    
    // 6x6 bits + 2 internal border bits + 2 external border bits = 10 bits per side
    int expectedSize = (6 + 4) * bitSize; 
    ASSERT_EQ(img.rows, expectedSize);
    ASSERT_EQ(img.cols, expectedSize);
}

TEST(Objdetect_Aruco2, SimpleDetection) {
    DictionaryType dict = DICT_ARUCO_MIP_36h12;
    int id = 100;
    Mat marker;
    generateMarkerImage(marker, dict, id, 20, false);
    
    // Create a larger canvas
    Mat canvas(marker.rows * 2, marker.cols * 2, CV_8UC1, Scalar(255));
    Rect roi(marker.cols / 2, marker.rows / 2, marker.cols, marker.rows);
    marker.copyTo(canvas(roi));
    
    std::vector<Marker> markers = detectMarkers(canvas, dict);
    
    ASSERT_EQ(markers.size(), 1u);
    EXPECT_EQ(markers[0].id, id);
    EXPECT_EQ(markers[0].dict, dict);
    
    // Check corners
    // Top-left should be (roi.x, roi.y)
    EXPECT_NEAR(markers[0].corners[0].x, (float)roi.x, 1.0f);
    EXPECT_NEAR(markers[0].corners[0].y, (float)roi.y, 1.0f);
}

TEST(Objdetect_Aruco2, Rotation) {
    DictionaryType dict = DICT_ARUCO_MIP_36h12;
    int id = 50;
    Mat marker;
    generateMarkerImage(marker, dict, id, 20, false);
    
    Mat canvas(marker.rows * 2, marker.cols * 2, CV_8UC1, Scalar(255));
    Rect roi(marker.cols / 2, marker.rows / 2, marker.cols, marker.rows);
    marker.copyTo(canvas(roi));
    
    Point2f center(canvas.cols / 2.0f, canvas.rows / 2.0f);
    std::vector<Point2f> originalCorners = {
        Point2f((float)roi.x, (float)roi.y),
        Point2f((float)roi.x + roi.width, (float)roi.y),
        Point2f((float)roi.x + roi.width, (float)roi.y + roi.height),
        Point2f((float)roi.x, (float)roi.y + roi.height)
    };
    
    for (double angle : {90.0, 180.0, 270.0}) {
        Mat rot = getRotationMatrix2D(center, angle, 1.0);
        Mat rotated;
        warpAffine(canvas, rotated, rot, canvas.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
        
        std::vector<Marker> markers = detectMarkers(rotated, dict);
        
        ASSERT_EQ(markers.size(), 1u) << "Failed for angle " << angle;
        EXPECT_EQ(markers[0].id, id);
        
        std::vector<Point2f> expectedCorners;
        transform(originalCorners, expectedCorners, rot);
        
        for (int i = 0; i < 4; i++) {
            // Note: rotation might shift pixels slightly, 1.5px tolerance is safe for non-subpixel
            EXPECT_NEAR(markers[0].corners[i].x, expectedCorners[i].x, 1.5f) << "Angle: " << angle << " Corner: " << i;
            EXPECT_NEAR(markers[0].corners[i].y, expectedCorners[i].y, 1.5f) << "Angle: " << angle << " Corner: " << i;
        }
    }
}

TEST(Objdetect_Aruco2, Perspective) {
    DictionaryType dict = DICT_ARUCO_MIP_36h12;
    int id = 25;
    Mat marker;
    generateMarkerImage(marker, dict, id, 20, false);
    
    Size imgSize(500, 500);
    Mat canvas(imgSize, CV_8UC1, Scalar(255));
    
    std::vector<Point2f> srcPoints = {
        Point2f(0, 0),
        Point2f((float)marker.cols, 0),
        Point2f((float)marker.cols, (float)marker.rows),
        Point2f(0, (float)marker.rows)
    };
    
    std::vector<Point2f> dstPoints = {
        Point2f(150, 150),
        Point2f(350, 180),
        Point2f(320, 380),
        Point2f(120, 350)
    };
    
    Mat M = getPerspectiveTransform(srcPoints, dstPoints);
    warpPerspective(marker, canvas, M, imgSize, INTER_LINEAR, BORDER_TRANSPARENT);
    
    std::vector<Marker> markers = detectMarkers(canvas, dict);
    
    ASSERT_EQ(markers.size(), 1u);
    EXPECT_EQ(markers[0].id, id);
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(markers[0].corners[i].x, dstPoints[i].x, 2.0f);
        EXPECT_NEAR(markers[0].corners[i].y, dstPoints[i].y, 2.0f);
    }
}

TEST(Objdetect_Aruco2, Inverted) {
    DictionaryType dict = DICT_ARUCO_MIP_36h12;
    int id = 10;
    Mat marker;
    generateMarkerImage(marker, dict, id, 20, false);
    
    Mat inverted = 255 - marker;
    
    Mat canvas(inverted.rows * 2, inverted.cols * 2, CV_8UC1, Scalar(0));
    Rect roi(inverted.cols / 2, inverted.rows / 2, inverted.cols, inverted.rows);
    inverted.copyTo(canvas(roi));
    
    DetectorParameters params;
    params.detectInvertedMarker = true;
    
    std::vector<Marker> markers = detectMarkers(canvas, dict, params);
    
    ASSERT_EQ(markers.size(), 1u);
    EXPECT_EQ(markers[0].id, id);
    EXPECT_NEAR(markers[0].corners[0].x, (float)roi.x, 1.0f);
    EXPECT_NEAR(markers[0].corners[0].y, (float)roi.y, 1.0f);
}

TEST(Objdetect_Aruco2, MultiMarker) {
    DictionaryType dict = DICT_ARUCO_MIP_36h12;
    std::vector<int> ids = {10, 20, 30};
    Mat canvas(800, 800, CV_8UC1, Scalar(255));
    std::vector<Rect> rois;
    
    for (int i = 0; i < (int)ids.size(); ++i) {
        Mat marker;
        generateMarkerImage(marker, dict, ids[i], 20, false);
        // Space them out: 200 pixels apart horizontally
        Rect roi(50 + i * 250, 50, marker.cols, marker.rows);
        marker.copyTo(canvas(roi));
        rois.push_back(roi);
    }
    
    std::vector<Marker> markers = detectMarkers(canvas, dict);
    
    ASSERT_EQ(markers.size(), ids.size());
    
    for (const auto& m : markers) {
        auto it = std::find(ids.begin(), ids.end(), m.id);
        ASSERT_NE(it, ids.end()) << "Detected unknown ID: " << m.id;
        int idx = (int)std::distance(ids.begin(), it);
        
        EXPECT_NEAR(m.corners[0].x, (float)rois[idx].x, 1.0f);
        EXPECT_NEAR(m.corners[0].y, (float)rois[idx].y, 1.0f);
    }
}

TEST(Objdetect_Aruco2, MultiDictionary) {
    DictionaryType dict1 = DICT_4X4_50;
    DictionaryType dict2 = DICT_ARUCO_MIP_36h12;
    int id1 = 5;
    int id2 = 10;
    
    Mat marker1, marker2;
    generateMarkerImage(marker1, dict1, id1, 20, false);
    generateMarkerImage(marker2, dict2, id2, 20, false);
    
    Mat canvas(600, 600, CV_8UC1, Scalar(255));
    marker1.copyTo(canvas(Rect(100, 100, marker1.cols, marker1.rows)));
    marker2.copyTo(canvas(Rect(300, 300, marker2.cols, marker2.rows)));
    
    std::vector<DictionaryType> dicts = {dict1, dict2};
    std::vector<Marker> markers = detectMarkers(canvas, dicts);
    
    ASSERT_EQ(markers.size(), 2u);
    
    bool found1 = false, found2 = false;
    for (const auto& m : markers) {
        if (m.id == id1 && m.dict == dict1) found1 = true;
        if (m.id == id2 && m.dict == dict2) found2 = true;
    }
    
    EXPECT_TRUE(found1);
    EXPECT_TRUE(found2);
}

}} // namespace
