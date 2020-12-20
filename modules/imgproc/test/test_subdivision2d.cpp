/*M///////////////////////////////////////////////////////////////////////////////////////
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//M*/
#include "test_precomp.hpp"

namespace opencv_test { namespace {
TEST(Imgproc_Subdiv2D_getTriangleList, regression_5788)
{
    const float points[65][2] = {
        { 390,  802}, { 397,  883}, { 414,  963 }, { 439, 1042 }, { 472, 1113},
        { 521, 1181}, { 591, 1238}, { 678, 1284 }, { 771, 1292 }, { 853, 1281},
        { 921, 1243}, { 982, 1191}, {1030, 1121 }, {1059, 1038 }, {1072,  945},
        {1081,  849}, {1082,  749}, { 459,  734 }, { 502,  704 }, { 554,  696},
        { 609,  698}, { 660,  707}, { 818,  688 }, { 874,  661 }, { 929,  646},
        { 982,  653}, {1026,  682}, { 740,  771 }, { 748,  834 }, { 756,  897},
        { 762,  960}, { 700,  998}, { 733, 1006 }, { 766, 1011 }, { 797,  999},
        { 825,  987}, { 528,  796}, { 566,  766 }, { 617,  763 }, { 659,  794},
        { 619,  808}, { 569,  812}, { 834,  777 }, { 870,  735 }, { 918,  729},
        { 958,  750}, { 929,  773}, { 882,  780 }, { 652, 1102 }, { 701, 1079},
        { 743, 1063}, { 774, 1068}, { 807, 1057 }, { 852, 1065 }, { 896, 1077},
        { 860, 1117}, { 820, 1135}, { 783, 1141 }, { 751, 1140 }, { 706, 1130},
        { 675, 1102}, { 743, 1094}, { 774, 1094 }, { 809, 1088 }, { 878, 1082}
    };
    std::vector<cv::Point2f> pts;
    cv::Rect rect(0, 0, 1500, 2000);
    cv::Subdiv2D subdiv(rect);
    for( int i = 0; i < 65; i++ )
    {
        cv::Point2f pt(points[i][0], points[i][1]);
        pts.push_back(pt);
    }

    subdiv.insert(pts);

    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);

    int trig_cnt = 0;
    for( std::vector<cv::Vec6f>::const_iterator it = triangles.begin(); it != triangles.end(); it++, trig_cnt++ )
    {
        EXPECT_TRUE( (0 <= triangles.at(trig_cnt).val[0] && triangles.at(trig_cnt).val[0] < 1500) &&
                     (0 <= triangles.at(trig_cnt).val[1] && triangles.at(trig_cnt).val[1] < 2000) &&
                     (0 <= triangles.at(trig_cnt).val[2] && triangles.at(trig_cnt).val[2] < 1500) &&
                     (0 <= triangles.at(trig_cnt).val[3] && triangles.at(trig_cnt).val[3] < 2000) &&
                     (0 <= triangles.at(trig_cnt).val[4] && triangles.at(trig_cnt).val[4] < 1500) &&
                     (0 <= triangles.at(trig_cnt).val[5] && triangles.at(trig_cnt).val[5] < 2000) );
    }
    EXPECT_EQ(trig_cnt, 105);
}

TEST(Imgproc_Subdiv2D_getTriangleList, regression_16763) {

    const auto less_than = [](const cv::Vec6f &lhs, const cv::Vec6f &rhs) -> bool {
        return (lhs[0] < rhs[0]) ||
               (lhs[0] == rhs[0] && lhs[1] < rhs[1]) ||
               (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] < rhs[2]) ||
               (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] && lhs[3] < rhs[3]) ||
               (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] && lhs[3] == rhs[3] && lhs[4] < rhs[4]) ||
               (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2] && lhs[3] == rhs[3] && lhs[4] == rhs[4] && lhs[5] < rhs[5]);
    };

    struct {
        std::vector<cv::Point2f> points;
        cv::Rect rect;
        std::vector<cv::Vec6f> triangles;
    } suits[] = {
        {
            { {623, 1000},{620, 1000}, {496, 770}, {473, 671} },
            { 472, 670, 625 - 472, 1002 - 670 },
            { { 620, 1000, 496, 770, 623, 1000 }, { 623, 1000, 496, 770, 473, 671 } }
        },
        {
            { {400, 400}, {400, 300}, {400, 200}, {395, 400} },
            { 0, 0, 500, 500 },
            { { 395, 400, 400, 200, 400, 300 }, { 400, 300, 400, 400, 395, 400} }
        }
    };

    for (auto& suit : suits) {
        cv::Subdiv2D subdivision(suit.rect);
        subdivision.insert(suit.points);

        std::sort(suit.triangles.begin(), suit.triangles.end(), less_than);

        std::vector<cv::Vec6f> output;
        subdivision.getTriangleList(output);

        sort(output.begin(), output.end(), less_than);

        EXPECT_EQ(output.size(), suit.triangles.size());
        for (size_t i = 0; i < output.size(); i++) {
            EXPECT_TRUE(!less_than(output[i], suit.triangles[i]) && !less_than(suit.triangles[i], output[i]));
        }
    }
}

}};
