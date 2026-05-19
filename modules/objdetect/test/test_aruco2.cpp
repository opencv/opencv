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
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;

    // 6x6 bits + 2 border bits = 8 bits per side
    // If externalBorder is true, it adds another bit? Let's check the code or just see the result.
    // Standard aruco FiducialMarkers have 1 bit border.
    // Let's see what generateFiducialMarkerImage does.
    getFiducialMarker(img, dictionary, id, bitSize, true);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(img.type(), CV_8UC1);

    // 6x6 bits + 2 internal border bits + 2 external border bits = 10 bits per side
    int expectedSize = (6 + 4) * bitSize;
    ASSERT_EQ(img.rows, expectedSize);
    ASSERT_EQ(img.cols, expectedSize);
}

TEST(Objdetect_Aruco2, SimpleDetection) {
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    int id = 100;
    Mat markerImg;
    getFiducialMarker(markerImg, dictionary, id, 20, false);

    // Create a larger canvas
    Mat canvas(markerImg.rows * 2, markerImg.cols * 2, CV_8UC1, Scalar(255));
    Rect roi(markerImg.cols / 2, markerImg.rows / 2, markerImg.cols, markerImg.rows);
    markerImg.copyTo(canvas(roi));

    auto markers = detectFiducialMarkers(canvas, dictionary);

    ASSERT_EQ(markers.size(), 1u);
    EXPECT_EQ(markers[0].id, id);
    EXPECT_EQ(markers[0].dictionary, dictionary);

    // Test getSolvePnpPoints
    Mat objPoints, imgPoints;
    getSolvePnpPoints(markers[0], objPoints, imgPoints, 0.1f);
    ASSERT_EQ(objPoints.total(), 4u);
    ASSERT_EQ(imgPoints.total(), 4u);
    ASSERT_EQ(objPoints.type(), CV_32FC3);
    ASSERT_EQ(imgPoints.type(), CV_32FC2);

    // Check that imgPoints match corners
    for(int i=0; i<4; i++) {
        EXPECT_NEAR(imgPoints.at<Vec2f>(i)[0], markers[0].corners[i].x, 1e-5);
        EXPECT_NEAR(imgPoints.at<Vec2f>(i)[1], markers[0].corners[i].y, 1e-5);
    }

    // Check corners
    // Top-left should be (roi.x, roi.y)
    EXPECT_NEAR(markers[0].corners[0].x, (float)roi.x, 1.0f);
    EXPECT_NEAR(markers[0].corners[0].y, (float)roi.y, 1.0f);
}

TEST(Objdetect_Aruco2, Rotation) {
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    int id = 50;
    Mat markerImg;
    getFiducialMarker(markerImg, dictionary, id, 20, false);

    Mat canvas(markerImg.rows * 2, markerImg.cols * 2, CV_8UC1, Scalar(255));
    Rect roi(markerImg.cols / 2, markerImg.rows / 2, markerImg.cols, markerImg.rows);
    markerImg.copyTo(canvas(roi));

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

        auto markers = detectFiducialMarkers(rotated, dictionary);

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
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    int id = 25;
    Mat markerImg;
    getFiducialMarker(markerImg, dictionary, id, 20, false);

    Size imgSize(500, 500);
    Mat canvas(imgSize, CV_8UC1, Scalar(255));

    std::vector<Point2f> srcPoints = {
        Point2f(0, 0),
        Point2f((float)markerImg.cols, 0),
        Point2f((float)markerImg.cols, (float)markerImg.rows),
        Point2f(0, (float)markerImg.rows)
    };

    std::vector<Point2f> dstPoints = {
        Point2f(150, 150),
        Point2f(350, 180),
        Point2f(320, 380),
        Point2f(120, 350)
    };

    Mat M = getPerspectiveTransform(srcPoints, dstPoints);
    warpPerspective(markerImg, canvas, M, imgSize, INTER_LINEAR, BORDER_TRANSPARENT);

    std::vector<FiducialMarker> markers = detectFiducialMarkers(canvas, dictionary);

    ASSERT_EQ(markers.size(), 1u);
    EXPECT_EQ(markers[0].id, id);

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(markers[0].corners[i].x, dstPoints[i].x, 2.0f);
        EXPECT_NEAR(markers[0].corners[i].y, dstPoints[i].y, 2.0f);
    }
}

TEST(Objdetect_Aruco2, Inverted) {
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    int id = 10;
    Mat markerImg;
    getFiducialMarker(markerImg, dictionary, id, 20, false);

    Mat inverted = 255 - markerImg;

    Mat canvas(inverted.rows * 2, inverted.cols * 2, CV_8UC1, Scalar(0));
    Rect roi(inverted.cols / 2, inverted.rows / 2, inverted.cols, inverted.rows);
    inverted.copyTo(canvas(roi));

    DetectionParameters params;
    params.detectInvertedMarker = true;

    std::vector<FiducialMarker> markers = detectFiducialMarkers(canvas, dictionary, params);

    ASSERT_EQ(markers.size(), 1u);
    EXPECT_EQ(markers[0].id, id);
    EXPECT_NEAR(markers[0].corners[0].x, (float)roi.x, 1.0f);
    EXPECT_NEAR(markers[0].corners[0].y, (float)roi.y, 1.0f);
}

TEST(Objdetect_Aruco2, MultiFiducialMarker) {
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    std::vector<int> ids = {10, 20, 30};
    Mat canvas(800, 800, CV_8UC1, Scalar(255));
    std::vector<Rect> rois;

    for (int i = 0; i < (int)ids.size(); ++i) {
        Mat markerImg;
        getFiducialMarker(markerImg, dictionary, ids[i], 20, false);
        // Space them out: 200 pixels apart horizontally
        Rect roi(50 + i * 250, 50, markerImg.cols, markerImg.rows);
        markerImg.copyTo(canvas(roi));
        rois.push_back(roi);
    }

    std::vector<FiducialMarker> markers = detectFiducialMarkers(canvas, dictionary);

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
    DictionaryType dictionary1 = DICT_4X4_50;
    DictionaryType dictionary2 = DICT_ARUCO_MIP_36h12;
    int id1 = 5;
    int id2 = 10;

    Mat markerImg1, markerImg2;
    getFiducialMarker(markerImg1, dictionary1, id1, 20, false);
    getFiducialMarker(markerImg2, dictionary2, id2, 20, false);

    Mat canvas(600, 600, CV_8UC1, Scalar(255));
    markerImg1.copyTo(canvas(Rect(100, 100, markerImg1.cols, markerImg1.rows)));
    markerImg2.copyTo(canvas(Rect(300, 300, markerImg2.cols, markerImg2.rows)));

    std::vector<DictionaryType> dictionaries = {dictionary1, dictionary2};
    std::vector<FiducialMarker> markers = detectFiducialMarkers(canvas, dictionaries);

    ASSERT_EQ(markers.size(), 2u);

    bool found1 = false, found2 = false;
    for (const auto& m : markers) {
        if (m.id == id1 && m.dictionary == dictionary1) found1 = true;
        if (m.id == id2 && m.dictionary == dictionary2) found2 = true;
    }
    EXPECT_TRUE(found1);
    EXPECT_TRUE(found2);
}

TEST(Objdetect_Aruco2, BoardGeneration) {
    Size gridSize(4, 3);
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    Mat img;
    getGridBoard(img, gridSize, dictionary, 20);

    ASSERT_FALSE(img.empty());
    ASSERT_EQ(img.type(), CV_8UC1);

    // Each FiducialMarker is 6x6 bits.
    // FiducialMarkerSizePix = 6 * 20 = 120
    // border = 120 / 4 = 30
    // imgWidth = 4 * 120 + 2 * 30 = 480 + 60 = 540
    // imgHeight = 3 * 120 + 2 * 30 = 360 + 60 = 420
    EXPECT_EQ(img.cols, 540);
    EXPECT_EQ(img.rows, 420);
}

TEST(Objdetect_Aruco2, BoardDetection) {
    Size gridSize(3, 2);
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    Mat boardImg;
    getGridBoard(boardImg, gridSize, dictionary, 20);

    Mat canvas(boardImg.rows + 100, boardImg.cols + 100, CV_8UC1, Scalar(255));
    Rect roi(50, 50, boardImg.cols, boardImg.rows);
    boardImg.copyTo(canvas(roi));

    GridBoard board;
    bool found = detectGridBoard(canvas, gridSize, dictionary, board);

    ASSERT_TRUE(found);
    EXPECT_EQ(board.gridSize, gridSize);
    EXPECT_EQ(board.dictionary, dictionary);
    EXPECT_EQ(board.markers.size(), 6u);

    // Test getSolvePnpPoints
    Mat objPoints, imgPoints;
    getSolvePnpPoints(board, objPoints, imgPoints, 0.05f);
    // 3x2 board has (3+1)x(2+1) = 12 intersection corners
    ASSERT_EQ(objPoints.total(), 12u);
    ASSERT_EQ(imgPoints.total(), 12u);
    ASSERT_EQ(objPoints.type(), CV_32FC3);
    ASSERT_EQ(imgPoints.type(), CV_32FC2);
}

TEST(Objdetect_Aruco2, BoardRotation) {
    Size gridSize(3, 2);
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    Mat boardImg;
    getGridBoard(boardImg, gridSize, dictionary, 20);

    Mat canvas(800, 800, CV_8UC1, Scalar(255));
    Rect roi((canvas.cols - boardImg.cols) / 2, (canvas.rows - boardImg.rows) / 2, boardImg.cols, boardImg.rows);
    boardImg.copyTo(canvas(roi));

    Point2f center(canvas.cols / 2.0f, canvas.rows / 2.0f);

    for (double angle : {90.0, 180.0, 270.0}) {
        Mat rot = getRotationMatrix2D(center, angle, 1.0);
        Mat rotated;
        warpAffine(canvas, rotated, rot, canvas.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));

        GridBoard board;
        bool found = detectGridBoard(rotated, gridSize, dictionary, board);

        ASSERT_TRUE(found) << "Failed for angle " << angle;
        EXPECT_EQ(board.markers.size(), 6u) << "Failed for angle " << angle;
    }
}

TEST(Objdetect_Aruco2, DiamondGeneration) {
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    Vec4i ids(1, 2, 3, 4);
    Mat img;
    getDiamondImage(img, dictionary, ids, 20);

    ASSERT_FALSE(img.empty());
    // Diamond is a 2x2 board.
    // FiducialMarkerSizePix = 6 * 20 = 120
    // border = 120 / 4 = 30
    // imgSize = 2 * 120 + 2 * 30 = 240 + 60 = 300
    EXPECT_EQ(img.cols, 300);
    EXPECT_EQ(img.rows, 300);
}

TEST(Objdetect_Aruco2, DiamondDetection) {
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    Vec4i ids(5, 10, 15, 20);
    Mat diamondImg;
    getDiamondImage(diamondImg, dictionary, ids, 20);

    Mat canvas(diamondImg.rows + 100, diamondImg.cols + 100, CV_8UC1, Scalar(255));
    Rect roi(50, 50, diamondImg.cols, diamondImg.rows);
    diamondImg.copyTo(canvas(roi));

    std::vector<Diamond> diamonds = detectDiamonds(canvas, dictionary);

    ASSERT_EQ(diamonds.size(), 1u);
    EXPECT_EQ(diamonds[0].id, ids);
    EXPECT_EQ(diamonds[0].dictionary, dictionary);
    EXPECT_EQ(diamonds[0].markers.size(), 4u);

    // Test getSolvePnpPoints
    Mat objPoints, imgPoints;
    getSolvePnpPoints(diamonds[0], objPoints, imgPoints, 0.1f);
    // Diamond returns a 3x3 grid of 9 points
    ASSERT_EQ(objPoints.total(), 9u);
    ASSERT_EQ(imgPoints.total(), 9u);
    ASSERT_EQ(objPoints.type(), CV_32FC3);
    ASSERT_EQ(imgPoints.type(), CV_32FC2);
}

TEST(Objdetect_Aruco2, DiamondRotation) {
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    Vec4i ids(1, 2, 3, 4);
    Mat diamondImg;
    getDiamondImage(diamondImg, dictionary, ids, 20);

    Mat canvas(800, 800, CV_8UC1, Scalar(255));
    Rect roi((canvas.cols - diamondImg.cols) / 2, (canvas.rows - diamondImg.rows) / 2, diamondImg.cols, diamondImg.rows);
    diamondImg.copyTo(canvas(roi));

    Point2f center(canvas.cols / 2.0f, canvas.rows / 2.0f);

    for (double angle : {90.0, 180.0, 270.0}) {
        Mat rot = getRotationMatrix2D(center, angle, 1.0);
        Mat rotated;
        warpAffine(canvas, rotated, rot, canvas.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));

        std::vector<Diamond> diamonds = detectDiamonds(rotated, dictionary);

        ASSERT_EQ(diamonds.size(), 1u) << "Failed for angle " << angle;
        // Diamond ID is Vec4i of the 4 constituent FiducialMarkers.
        // The detector should return them in consistent order regardless of rotation.
        EXPECT_EQ(diamonds[0].id, ids) << "Failed for angle " << angle;
    }
}

TEST(Objdetect_Aruco2, DiamondPerspective) {
    DictionaryType dictionary = DICT_ARUCO_MIP_36h12;
    Vec4i ids(10, 20, 30, 40);
    Mat diamondImg;
    getDiamondImage(diamondImg, dictionary, ids, 20);

    Size imgSize(800, 800);
    Mat canvas(imgSize, CV_8UC1, Scalar(255));

    std::vector<Point2f> srcPoints = {
        Point2f(0, 0),
        Point2f((float)diamondImg.cols, 0),
        Point2f((float)diamondImg.cols, (float)diamondImg.rows),
        Point2f(0, (float)diamondImg.rows)
    };

    std::vector<Point2f> dstPoints = {
        Point2f(200, 200),
        Point2f(600, 250),
        Point2f(550, 650),
        Point2f(150, 600)
    };

    Mat M = getPerspectiveTransform(srcPoints, dstPoints);
    warpPerspective(diamondImg, canvas, M, imgSize, INTER_LINEAR, BORDER_TRANSPARENT);

    std::vector<Diamond> diamonds = detectDiamonds(canvas, dictionary);

    ASSERT_EQ(diamonds.size(), 1u);
    EXPECT_EQ(diamonds[0].id, ids);
}

TEST(Objdetect_Aruco2, FractalGeneration) {
    FractalType ftype = FRACTAL_2L_6;
    Mat img;
    getFractalImage(img, ftype, 20);

    ASSERT_FALSE(img.empty());
    // Fractal size depends on nesting, let's just check it's non-empty and square.
    EXPECT_EQ(img.cols, img.rows);
    EXPECT_GT(img.cols, 0);
}

TEST(Objdetect_Aruco2, FractalDetection) {
    FractalType ftype = FRACTAL_2L_6;
    Mat fractalImg;
    getFractalImage(fractalImg, ftype, 40); // Larger for better detection

    Mat canvas(fractalImg.rows + 100, fractalImg.cols + 100, CV_8UC1, Scalar(255));
    Rect roi(50, 50, fractalImg.cols, fractalImg.rows);
    fractalImg.copyTo(canvas(roi));

    std::vector<FractalMarker> fractals = detectFractals(canvas, ftype);

    ASSERT_EQ(fractals.size(), 1u);
    EXPECT_EQ(fractals[0].type, ftype);

    // Test getSolvePnpPoints
    Mat objPoints, imgPoints;
    getSolvePnpPoints(fractals[0], objPoints, imgPoints, 0.2f);
    // Fractal FiducialMarker should have at least the 4 outer corners, and likely more inner ones.
    ASSERT_GE(objPoints.total(), 4u);
    ASSERT_EQ(objPoints.total(), imgPoints.total());
    ASSERT_EQ(objPoints.type(), CV_32FC3);
    ASSERT_EQ(imgPoints.type(), CV_32FC2);
}

TEST(Objdetect_Aruco2, FractalRotation) {
    FractalType ftype = FRACTAL_2L_6;
    Mat fractalImg;
    getFractalImage(fractalImg, ftype, 20); // Smaller bitSize (20 instead of 40)

    Mat canvas(1200, 1200, CV_8UC1, Scalar(255)); // Larger canvas
    int posX = (canvas.cols - fractalImg.cols) / 2;
    int posY = (canvas.rows - fractalImg.rows) / 2;
    // Ensure the FiducialMarker fits in the canvas
    ASSERT_GT(posX, 0);
    ASSERT_GT(posY, 0);
    Rect roi(posX, posY, fractalImg.cols, fractalImg.rows);
    fractalImg.copyTo(canvas(roi));

    Point2f center(canvas.cols / 2.0f, canvas.rows / 2.0f);

    for (double angle : {90.0, 180.0, 270.0}) {
        Mat rot = getRotationMatrix2D(center, angle, 1.0);
        Mat rotated;
        warpAffine(canvas, rotated, rot, canvas.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));

        std::vector<FractalMarker> fractals = detectFractals(rotated, ftype);

        ASSERT_EQ(fractals.size(), 1u) << "Failed for angle " << angle;
    }
}

TEST(Objdetect_Aruco2, FractalPerspective) {
    FractalType ftype = FRACTAL_3L_6; // Use more levels for perspective robustness check
    Mat fractalImg;
    getFractalImage(fractalImg, ftype, 40);

    Size imgSize(800, 800);
    Mat canvas(imgSize, CV_8UC1, Scalar(255));

    std::vector<Point2f> srcPoints = {
        Point2f(0, 0),
        Point2f((float)fractalImg.cols, 0),
        Point2f((float)fractalImg.cols, (float)fractalImg.rows),
        Point2f(0, (float)fractalImg.rows)
    };

    std::vector<Point2f> dstPoints = {
        Point2f(200, 200),
        Point2f(600, 250),
        Point2f(550, 650),
        Point2f(150, 600)
    };

    Mat M = getPerspectiveTransform(srcPoints, dstPoints);
    warpPerspective(fractalImg, canvas, M, imgSize, INTER_LINEAR, BORDER_TRANSPARENT);

    std::vector<FractalMarker> fractals = detectFractals(canvas, ftype);

    ASSERT_EQ(fractals.size(), 1u);
}


}} // namespace
