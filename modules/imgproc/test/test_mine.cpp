//
// Created by amir on 06.12.20.
//

#include "test_precomp.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace opencv_test { namespace {

    void print_extra(Rect &rect) {
        float big_coord = 3.f * MAX( rect.width, rect.height );
        float rx = (float)rect.x;
        float ry = (float)rect.y;

        auto topLeft = Point2f( rx, ry );
        auto bottomRight = Point2f( rx + rect.width, ry + rect.height );

        printf("x = [%.0f, %.0f, %.0f, %.0f, %0.f];\n", topLeft.x, topLeft.x, bottomRight.x, bottomRight.x, topLeft.x);
        printf("y = [%.0f, %.0f, %.0f, %.0f, %0.f];\n", topLeft.y, bottomRight.y, bottomRight.y, topLeft.y, topLeft.y);
        printf("plot(x, y, '-c');\n");
        printf("hold on;\n");

        Point2f ppA( rx + big_coord, ry );
        Point2f ppB( rx, ry + big_coord );
        Point2f ppC( rx - big_coord, ry - big_coord );

        printf("plot(%.0f, %.0f, 'o-k');\n", ppA.x, ppA.y);
        printf("hold on;\n");
        printf("plot(%.0f, %.0f, 'o-k');\n", ppB.x, ppB.y);
        printf("hold on;\n");
        printf("plot(%.0f, %.0f, 'o-k');\n", ppC.x, ppC.y);
        printf("hold on;\n");
    }

    void print_voronoi(Subdiv2D &subdiv) {
        std::vector<std::vector<Point2f>> facetList;
        std::vector<Point2f> facetCenters;
        subdiv.getVoronoiFacetList({}, facetList, facetCenters);
        for (size_t i = 0; i < facetList.size(); ++i) {
            printf("x = [");
            for (size_t j = 0; j < facetList[i].size(); ++j) {
                printf("%.0f ", facetList[i][j].x);
            }
            printf("%.0f];\n", facetList[i][0].x);
            printf("y = [");
            for (size_t j = 0; j < facetList[i].size(); ++j) {
                printf("%.0f ", facetList[i][j].y);
            }
            printf("%.0f];\n", facetList[i][0].y);
            printf("plot(x, y, '-k');\n");
            printf("hold on;\n");
            printf("plot(%.0f, %.0f, 'o-k');\n", facetCenters[i].x, facetCenters[i].y);
            printf("hold on;\n");
        }

    }

TEST(Imgproc_Mine, accuracy) {

//    std::vector<cv::Point2f> points{
//            {623, 1000}, {620, 1000}, {496, 770}, {473, 671}
//    };
//    cv::Rect subdivRect{
//        cv::Point{472, 670}, cv::Point{625, 1002}
//    };

    std::vector<cv::Point2f> points {
            {400, 400}, {400, 300}, {400, 200}, {390, 400}
    };		// 3 out of 4 points are in a straight line
    cv::Rect subdivRect{
            cv::Point{0, 500}, cv::Point{500, 0}
    };	// Square with coordinates: (0,0), (0,500), (500,500), (500,0)

    cv::Subdiv2D subdiv{subdivRect};

    printf("clf;\n");
    printf("axis('square');\n");
    printf("hold on;\n");
    print_extra(subdivRect);

    for (size_t i = 0; i < points.size() - 1; ++i) {
        subdiv.insert(points[i]);
    }

//    print_voronoi(subdiv);
    std::vector<cv::Vec6f> out;
    subdiv.getTriangleList(out);

    subdiv.insert(points[points.size() - 1]);

    print_voronoi(subdiv);
    subdiv.getTriangleList(out);

    std::cout << "triangles: " << out.size() << std::endl;
//    cv::Vec6f outTr = out[0];
    // clang-format off
//    std::cout << '[' << outTr[0] << ", " << outTr[1] << "], "
//              << '[' << outTr[2] << ", " << outTr[3] << "], "
//              << '[' << outTr[4] << ", " << outTr[5] << "]" << std::endl;
    // clang-format on
}

}}
