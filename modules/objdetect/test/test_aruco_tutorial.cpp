// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"

namespace opencv_test { namespace {


TEST(CV_ArucoTutorial, can_find_singlemarkersoriginal)
{
    string img_path = cvtest::findDataFile("aruco/singlemarkersoriginal.jpg");
    Mat image = imread(img_path);
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250));

    vector<int> ids;
    vector<vector<Point2f> > corners, rejected;
    const size_t N = 6ull;
    // corners of ArUco markers with indices goldCornersIds
    const int goldCorners[N][8] = { {359,310, 404,310, 410,350, 362,350}, {427,255, 469,256, 477,289, 434,288},
                                    {233,273, 190,273, 196,241, 237,241}, {298,185, 334,186, 335,212, 297,211},
                                    {425,163, 430,186, 394,186, 390,162}, {195,155, 230,155, 227,178, 190,178} };
    const int goldCornersIds[N] = { 40, 98, 62, 23, 124, 203};
    map<int, const int*> mapGoldCorners;
    for (size_t i = 0; i < N; i++)
        mapGoldCorners[goldCornersIds[i]] = goldCorners[i];

    detector.detectMarkers(image, corners, ids, rejected);

    ASSERT_EQ(N, ids.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        ASSERT_EQ(4ull, corners[i].size());
        ASSERT_TRUE(mapGoldCorners.find(arucoId) != mapGoldCorners.end());
        for (int j = 0; j < 4; j++)
        {
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2]), corners[i][j].x, 1.f);
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2 + 1]), corners[i][j].y, 1.f);
        }
    }
}

TEST(CV_ArucoTutorial, can_find_gboriginal)
{
    string imgPath = cvtest::findDataFile("aruco/gboriginal.jpg");
    Mat image = imread(imgPath);
    string dictPath = cvtest::findDataFile("aruco/tutorial_dict.yml");
    aruco::Dictionary dictionary;

    FileStorage fs(dictPath, FileStorage::READ);
    dictionary.aruco::Dictionary::readDictionary(fs.root()); // set marker from tutorial_dict.yml
    aruco::DetectorParameters detectorParams;

    aruco::ArucoDetector detector(dictionary, detectorParams);

    vector<int> ids;
    vector<vector<Point2f> > corners, rejected;
    const size_t N = 35ull;
    // corners of ArUco markers with indices 0, 1, ..., 34
    const int goldCorners[N][8] = { {252,74, 286,81, 274,102, 238,95},    {295,82, 330,89, 319,111, 282,104},
                                    {338,91, 375,99, 365,121, 327,113},   {383,100, 421,107, 412,130, 374,123},
                                    {429,109, 468,116, 461,139, 421,132}, {235,100, 270,108, 257,130, 220,122},
                                    {279,109, 316,117, 304,140, 266,133}, {324,119, 362,126, 352,150, 313,143},
                                    {371,128, 410,136, 400,161, 360,152}, {418,139, 459,145, 451,170, 410,163},
                                    {216,128, 253,136, 239,161, 200,152}, {262,138, 300,146, 287,172, 248,164},
                                    {309,148, 349,156, 337,183, 296,174}, {358,158, 398,167, 388,194, 346,185},
                                    {407,169, 449,176, 440,205, 397,196}, {196,158, 235,168, 218,195, 179,185},
                                    {243,170, 283,178, 269,206, 228,197}, {293,180, 334,190, 321,218, 279,209},
                                    {343,192, 385,200, 374,230, 330,220}, {395,203, 438,211, 429,241, 384,233},
                                    {174,192, 215,201, 197,231, 156,221}, {223,204, 265,213, 249,244, 207,234},
                                    {275,215, 317,225, 303,257, 259,246}, {327,227, 371,238, 359,270, 313,259},
                                    {381,240, 426,249, 416,282, 369,273}, {151,228, 193,238, 173,271, 130,260},
                                    {202,241, 245,251, 228,285, 183,274}, {255,254, 300,264, 284,299, 238,288},
                                    {310,267, 355,278, 342,314, 295,302}, {366,281, 413,290, 402,327, 353,317},
                                    {125,267, 168,278, 147,314, 102,303}, {178,281, 223,293, 204,330, 157,317},
                                    {233,296, 280,307, 263,346, 214,333}, {291,310, 338,322, 323,363, 274,349},
                                    {349,325, 399,336, 386,378, 335,366} };
    map<int, const int*> mapGoldCorners;
    for (int i = 0; i < static_cast<int>(N); i++)
        mapGoldCorners[i] = goldCorners[i];

    detector.detectMarkers(image, corners, ids, rejected);

    ASSERT_EQ(N, ids.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        ASSERT_EQ(4ull, corners[i].size());
        ASSERT_TRUE(mapGoldCorners.find(arucoId) != mapGoldCorners.end());
        for (int j = 0; j < 4; j++)
        {
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j*2]), corners[i][j].x, 1.f);
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j*2+1]), corners[i][j].y, 1.f);
        }
    }
}

TEST(CV_ArucoTutorial, can_find_choriginal)
{
    string imgPath = cvtest::findDataFile("aruco/choriginal.jpg");
    Mat image = imread(imgPath);
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250));

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    const size_t N = 17ull;
    // corners of aruco markers with indices goldCornersIds
    const int goldCorners[N][8] = { {268,77,  290,80,  286,97,  263,94},  {360,90,  382,93,  379,111, 357,108},
                                    {211,106, 233,109, 228,127, 205,123}, {306,120, 328,124, 325,142, 302,138},
                                    {402,135, 425,139, 423,157, 400,154}, {247,152, 271,155, 267,174, 242,171},
                                    {347,167, 371,171, 369,191, 344,187}, {185,185, 209,189, 203,210, 178,206},
                                    {288,201, 313,206, 309,227, 284,223}, {393,218, 418,222, 416,245, 391,241},
                                    {223,240, 250,244, 244,268, 217,263}, {333,258, 359,262, 356,286, 329,282},
                                    {152,281, 179,285, 171,312, 143,307}, {267,300, 294,305, 289,331, 261,327},
                                    {383,319, 410,324, 408,351, 380,347}, {194,347, 223,352, 216,382, 186,377},
                                    {315,368, 345,373, 341,403, 310,398} };
    map<int, const int*> mapGoldCorners;
    for (int i = 0; i < static_cast<int>(N); i++)
        mapGoldCorners[i] = goldCorners[i];

    detector.detectMarkers(image, corners, ids, rejected);

    ASSERT_EQ(N, ids.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        ASSERT_EQ(4ull, corners[i].size());
        ASSERT_TRUE(mapGoldCorners.find(arucoId) != mapGoldCorners.end());
        for (int j = 0; j < 4; j++)
        {
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2]), corners[i][j].x, 1.f);
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2 + 1]), corners[i][j].y, 1.f);
        }
    }
}

TEST(CV_ArucoTutorial, can_find_chocclusion)
{
    string imgPath = cvtest::findDataFile("aruco/chocclusion_original.jpg");
    Mat image = imread(imgPath);
    aruco::ArucoDetector detector(aruco::getPredefinedDictionary(aruco::DICT_6X6_250));

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    const size_t N = 13ull;
    // corners of aruco markers with indices goldCornersIds
    const int goldCorners[N][8] = { {301,57, 322,62, 317,79, 295,73}, {391,80, 413,85, 408,103, 386,97},
                                    {242,79, 264,85, 256,102, 234,96}, {334,103, 357,109, 352,126, 329,121},
                                    {428,129, 451,134, 448,152, 425,146}, {274,128, 296,134, 290,153, 266,147},
                                    {371,154, 394,160, 390,180, 366,174}, {208,155, 232,161, 223,181, 199,175},
                                    {309,182, 333,188, 327,209, 302,203}, {411,210, 436,216, 432,238, 407,231},
                                    {241,212, 267,219, 258,242, 232,235}, {167,244, 194,252, 183,277, 156,269},
                                    {202,314, 230,322, 220,349, 191,341} };
    map<int, const int*> mapGoldCorners;
    const int goldCornersIds[N] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15};
    for (int i = 0; i < static_cast<int>(N); i++)
        mapGoldCorners[goldCornersIds[i]] = goldCorners[i];

    detector.detectMarkers(image, corners, ids, rejected);

    ASSERT_EQ(N, ids.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        ASSERT_EQ(4ull, corners[i].size());
        ASSERT_TRUE(mapGoldCorners.find(arucoId) != mapGoldCorners.end());
        for (int j = 0; j < 4; j++)
        {
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2]), corners[i][j].x, 1.f);
            EXPECT_NEAR(static_cast<float>(mapGoldCorners[arucoId][j * 2 + 1]), corners[i][j].y, 1.f);
        }
    }
}

TEST(CV_ArucoTutorial, can_find_diamondmarkers)
{
    string imgPath = cvtest::findDataFile("aruco/diamondmarkers.jpg");
    Mat image = imread(imgPath);

    string dictPath = cvtest::findDataFile("aruco/tutorial_dict.yml");
    aruco::Dictionary dictionary;
    FileStorage fs(dictPath, FileStorage::READ);
    dictionary.aruco::Dictionary::readDictionary(fs.root()); // set marker from tutorial_dict.yml

    string detectorPath = cvtest::findDataFile("aruco/detector_params.yml");
    fs = FileStorage(detectorPath, FileStorage::READ);
    aruco::DetectorParameters detectorParams;
    detectorParams.readDetectorParameters(fs.root());
    detectorParams.cornerRefinementMethod = aruco::CORNER_REFINE_APRILTAG;

    aruco::CharucoBoard charucoBoard(Size(3, 3), 0.4f, 0.25f, dictionary);
    aruco::CharucoDetector detector(charucoBoard, aruco::CharucoParameters(), detectorParams);

    vector<int> ids;
    vector<vector<Point2f> > corners, diamondCorners;
    vector<Vec4i> diamondIds;
    const size_t N = 12ull;
    // corner indices of ArUco markers
    const int goldCornersIds[N] = { 4, 12, 11, 3, 12, 10, 12, 10, 10, 11, 2, 11 };
    map<int, int> counterGoldCornersIds;
    for (int i = 0; i < static_cast<int>(N); i++)
        counterGoldCornersIds[goldCornersIds[i]]++;

    const size_t diamondsN = 3;
    // corners of diamonds with Vec4i indices
    const float goldDiamondCorners[diamondsN][8] = {{195.6f,150.9f, 213.5f,201.2f, 136.4f,215.3f, 122.4f,163.5f},
                                            {501.1f,171.3f, 501.9f,208.5f, 446.2f,199.8f, 447.8f,163.3f},
                                            {343.4f,361.2f, 359.7f,328.7f, 400.8f,344.6f, 385.7f,378.4f}};
    auto comp = [](const Vec4i& a, const Vec4i& b) {
        for (int i = 0; i < 3; i++)
            if (a[i] != b[i]) return a[i] < b[i];
        return a[3] < b[3];
    };
    map<Vec4i, const float*, decltype(comp)> goldDiamonds(comp);
    goldDiamonds[Vec4i(10, 4, 11, 12)] = goldDiamondCorners[0];
    goldDiamonds[Vec4i(10, 3, 11, 12)] = goldDiamondCorners[1];
    goldDiamonds[Vec4i(10, 2, 11, 12)] = goldDiamondCorners[2];

    detector.detectDiamonds(image, diamondCorners, diamondIds, corners, ids);
    map<int, int> counterRes;

    ASSERT_EQ(N, ids.size());
    for (size_t i = 0; i < N; i++)
    {
        int arucoId = ids[i];
        counterRes[arucoId]++;
    }

    ASSERT_EQ(counterGoldCornersIds, counterRes); // check the number of ArUco markers
    ASSERT_EQ(goldDiamonds.size(), diamondIds.size()); // check the number of diamonds

    for (size_t i = 0; i < goldDiamonds.size(); i++)
    {
        Vec4i diamondId = diamondIds[i];
        ASSERT_TRUE(goldDiamonds.find(diamondId) != goldDiamonds.end());
        for (int j = 0; j < 4; j++)
        {
            EXPECT_NEAR(goldDiamonds[diamondId][j * 2], diamondCorners[i][j].x, 0.5f);
            EXPECT_NEAR(goldDiamonds[diamondId][j * 2 + 1], diamondCorners[i][j].y, 0.5f);
        }
    }
}

}} // namespace
