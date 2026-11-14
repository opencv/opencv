// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "test_aruco_utils.hpp"

namespace opencv_test { namespace {

/**
 * @brief Check Charuco2 detection accuracy using synthetic images
 */
class CV_Charuco2Detection : public cvtest::BaseTest {
    public:
    CV_Charuco2Detection() {}
    void run(int) override;
};

void CV_Charuco2Detection::run(int) {
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500);
    int squaresX = 5;
    int squaresY = 5;
    float squareLength = 0.04f;
    aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
    aruco::CharucoBoard board(Size(squaresX, squaresY), squareLength, squareLength, dictionary, noArray(), aruco::CHARUCO_2);

    cameraMatrix.at<double>(0, 0) = cameraMatrix.at<double>(1, 1) = 600;
    cameraMatrix.at<double>(0, 2) = imgSize.width / 2.0;
    cameraMatrix.at<double>(1, 2) = imgSize.height / 2.0;

    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));

    // for different perspectives
    for(double distance : {0.3, 0.5}) {
        for(int yaw = -45; yaw <= 45; yaw += 30) {
            for(int pitch = -45; pitch <= 45; pitch += 30) {
                Mat rvec, tvec;
                getSyntheticRT(deg2rad(yaw), deg2rad(pitch), distance, rvec, tvec);

                // Generate board image
                Mat boardImg;
                board.generateImage(Size(1000, 1000), boardImg);

                // Project board image corners to find the transformation
                float b = 0.25f * squareLength;
                vector<Point3f> boardCorners3d;
                boardCorners3d.push_back(Point3f(-b, -b, 0));
                boardCorners3d.push_back(Point3f(squaresX * squareLength + b, -b, 0));
                boardCorners3d.push_back(Point3f(squaresX * squareLength + b, squaresY * squareLength + b, 0));
                boardCorners3d.push_back(Point3f(-b, squaresY * squareLength + b, 0));

                // Center the board for projection (matching projectMarker/projectBoard behavior)
                Point3f boardCenter(squaresX * squareLength / 2.f, squaresY * squareLength / 2.f, 0.f);
                vector<Point3f> centeredBoardCorners3d = boardCorners3d;
                for(auto& p : centeredBoardCorners3d) {
                    p -= boardCenter;
                }

                vector<Point2f> boardCorners2d;
                projectPoints(centeredBoardCorners3d, rvec, tvec, cameraMatrix, distCoeffs, boardCorners2d);

                vector<Point2f> srcCorners;
                srcCorners.push_back(Point2f(0, 0));
                srcCorners.push_back(Point2f((float)boardImg.cols, 0));
                srcCorners.push_back(Point2f((float)boardImg.cols, (float)boardImg.rows));
                srcCorners.push_back(Point2f(0, (float)boardImg.rows));

                Mat H = getPerspectiveTransform(srcCorners, boardCorners2d);
                Mat img(imgSize, CV_8UC1, Scalar(255));
                warpPerspective(boardImg, img, H, imgSize, INTER_LINEAR, BORDER_CONSTANT, Scalar(255));

                // Detect
                aruco::CharucoDetector detector(board);
                vector<Point2f> charucoCorners;
                vector<int> charucoIds;
                detector.detectBoard(img, charucoCorners, charucoIds);

                // Check results against projected ground truth corners
                vector<Point3f> allCorners3d = board.getChessboardCorners();
                vector<Point3f> centeredAllCorners3d = allCorners3d;
                for(auto& p : centeredAllCorners3d) {
                    p -= boardCenter;
                }
                vector<Point2f> allCorners2d;
                projectPoints(centeredAllCorners3d, rvec, tvec, cameraMatrix, distCoeffs, allCorners2d);

                ASSERT_GT(charucoIds.size(), (size_t)0) << "No Charuco corners detected at yaw=" << yaw << " pitch=" << pitch << " dist=" << distance;
                for(size_t i = 0; i < charucoIds.size(); i++) {
                    int id = charucoIds[i];
                    ASSERT_LT(id, (int)allCorners2d.size()) << "Invalid Charuco corner id";
                    double err = cv::norm(charucoCorners[i] - allCorners2d[id]);

                    ASSERT_LT(err, 5.0) << "Charuco corner reprojection error too high at corner " << id;
                }
            }
        }
    }
}

TEST(CV_Charuco2Detection, accuracy) {
    CV_Charuco2Detection test;
    test.run(0);
}

/**
 * @brief Check CharucoBoard creation for CHARUCO_2
 */
TEST(CV_CharucoBoardCreation, charuco2) {
    Size boardSize(5, 5);
    float squareLength = 0.04f;
    aruco::Dictionary dict = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);

    // Using the main constructor with CHARUCO_2 type
    aruco::CharucoBoard board1(boardSize, squareLength, squareLength, dict, noArray(), aruco::CHARUCO_2);
    ASSERT_EQ(board1.getChessboardSize(), boardSize);
    ASSERT_EQ(board1.getSquareLength(), squareLength);
    ASSERT_EQ(board1.getMarkerLength(), squareLength);
    ASSERT_EQ(board1.getChessboardCorners().size(), (size_t)((boardSize.width + 1) * (boardSize.height + 1)));

    // Using the convenience constructor for CHARUCO_2
    aruco::CharucoBoard board2(boardSize, squareLength, dict);
    ASSERT_EQ(board2.getChessboardSize(), boardSize);
    ASSERT_EQ(board2.getSquareLength(), squareLength);
    ASSERT_EQ(board2.getMarkerLength(), squareLength);
    ASSERT_EQ(board2.getChessboardCorners().size(), (size_t)((boardSize.width + 1) * (boardSize.height + 1)));

    // getObjPoints should return W*H marker corners (4 per marker)
    ASSERT_EQ(board1.getObjPoints().size(), (size_t)(boardSize.width * boardSize.height));
    for (const auto& mc : board1.getObjPoints())
        ASSERT_EQ(mc.size(), 4u);

    // getRightBottomCorner should be at (W*sq, H*sq, 0)
    Point3f rbc = board1.getRightBottomCorner();
    EXPECT_NEAR(rbc.x, boardSize.width  * squareLength, 1e-6f);
    EXPECT_NEAR(rbc.y, boardSize.height * squareLength, 1e-6f);
    EXPECT_NEAR(rbc.z, 0.f, 1e-6f);

    // getNearestMarkerIdx/Corners return empty (not throw) for CHARUCO_2
    ASSERT_TRUE(board1.getNearestMarkerIdx().empty());
    ASSERT_TRUE(board1.getNearestMarkerCorners().empty());
}

/**
 * @brief Check Charuco2 matchImagePoints
 */
TEST(CV_Charuco2MatchImagePoints, accuracy) {
    Size boardSize(3, 3);
    float squareLength = 0.1f;
    aruco::Dictionary dict = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
    aruco::CharucoBoard board(boardSize, squareLength, dict);

    // Corners for CHARUCO_2 are (W+1)*(H+1)
    int nCorners = (boardSize.width + 1) * (boardSize.height + 1);
    vector<Point2f> detectedCorners;
    vector<int> detectedIds;

    // Simulate detection of all corners with some arbitrary image coordinates
    for(int i = 0; i < nCorners; i++) {
        detectedCorners.push_back(Point2f((float)i * 10, (float)i * 10));
        detectedIds.push_back(i);
    }

    Mat objPoints, imgPoints;
    board.matchImagePoints(detectedCorners, detectedIds, objPoints, imgPoints);

    ASSERT_EQ(objPoints.total(), (size_t)nCorners);
    ASSERT_EQ(imgPoints.total(), (size_t)nCorners);

    vector<Point3f> boardObjPoints = board.getChessboardCorners();
    for(int i = 0; i < nCorners; i++) {
        Point3f p3d = objPoints.at<Point3f>(i);
        EXPECT_NEAR(p3d.x, boardObjPoints[i].x, 1e-5);
        EXPECT_NEAR(p3d.y, boardObjPoints[i].y, 1e-5);
        EXPECT_NEAR(p3d.z, boardObjPoints[i].z, 1e-5);

        Point2f p2d = imgPoints.at<Point2f>(i);
        EXPECT_NEAR(p2d.x, (float)i * 10, 1e-5);
        EXPECT_NEAR(p2d.y, (float)i * 10, 1e-5);
    }
}

}} // namespace
