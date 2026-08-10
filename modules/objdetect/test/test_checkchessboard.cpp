// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// cv::checkChessboard() is public API, and findChessboardCorners() falls back to it under
// CALIB_CB_FAST_CHECK (see calibinit.cpp): when it returns false there, detection stops
// immediately, so a false negative here silently costs a detection.
//
// Its only coverage used to be an assertion buried inside
// TEST(Calib3d_ChessboardDetector, timing), which stopped at the first mismatch via a goto
// and sat among per-image timing printf output that nothing collected. Extracted here so
// the check is named for what it verifies and cannot be dropped along with the timing.
//
// checkChessboard() is deliberately not required to agree with findChessboardCorners():
// it is a permissive pre-filter, and the two legitimately differ on
// chessboard-artificial2.png, where checkChessboard() is the one that gets it right.
TEST(Objdetect_CheckChessboard, accuracy)
{
    const string folder = string(cvtest::TS::ptr()->get_data_path()) + "cameracalibration/";
    const string listname = folder + "chessboard_timing_list.dat";

    FileStorage fs(listname, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened()) << "Could not read " << listname;

    FileNode boards = fs["boards"];
    ASSERT_TRUE(boards.isSeq()) << listname << " does not contain a 'boards' sequence";
    ASSERT_EQ(size_t(0), boards.size() % 4) << listname << " is malformed";

    const int count = (int)boards.size() / 4;
    ASSERT_GT(count, 0) << listname << " lists no images";

    FileNodeIterator it = boards.begin();
    for (int i = 0; i < count; i++)
    {
        string imgname;
        int isChessboard = 0;
        Size patternSize;
        read(*it++, imgname, "dummy.txt");
        read(*it++, isChessboard, 0);
        read(*it++, patternSize.width, -1);
        read(*it++, patternSize.height, -1);

        SCOPED_TRACE(cv::format("image %d/%d: %s", i + 1, count, imgname.c_str()));

        Mat img = imread(folder + imgname);
        ASSERT_FALSE(img.empty()) << "Could not read " << folder << imgname;
        ASSERT_GT(patternSize.width, 0);
        ASSERT_GT(patternSize.height, 0);

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);

        // EXPECT, not ASSERT: every image is reported, rather than stopping at the first
        // mismatch as the original loop did.
        EXPECT_EQ(isChessboard != 0, checkChessboard(gray, patternSize));
    }
}

}} // namespace
