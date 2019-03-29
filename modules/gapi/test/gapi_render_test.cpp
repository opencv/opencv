// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include "test_precomp.hpp"
#include "opencv2/gapi/render.hpp"

namespace opencv_test
{

//TEST(Render, DrawRectangle)
//{
    //auto out_mat = cv::imread("/home/atalaman/Downloads/people.jpg");
    ////cv::Mat out_mat(100, 100, CV_8UC3);
    ////cv::randu(out_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    //cv::Mat ref_mat;
    //out_mat.copyTo(ref_mat);
    //cv::Point2f p1(50, 50), p2(75, 75);
    //cv::Scalar color(0, 255, 0);
    //
    //auto r = cv::RenderCreator::create(cv::RenderCreator::BackendType::OCV);

    //cv::rectangle(ref_mat, p1, p2, color);

    //r->rectangle(p1, p2, color);
    //r->run(out_mat);

    //EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
//}

//TEST(Render, DrawText)
//{
    //auto out_mat = cv::imread("/home/atalaman/Downloads/people.jpg");
    ////cv::Mat out_mat(100, 100, CV_8UC3);
    ////cv::randu(out_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    //cv::Mat ref_mat;
    //out_mat.copyTo(ref_mat);
    //cv::Point2f p1(50, 50);
    //cv::Scalar color(0, 255, 0);
    //std::string text = "People";
    ////std::string text = "Россия - самая большая страна в мире";

    //auto r = cv::RenderCreator::create(cv::RenderCreator::BackendType::OCV);

    //cv::putText(ref_mat, text, p1, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);

    //r->putText(text, p1, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
    //r->run(out_mat);

    //cv::imshow("window", out_mat);
    //cv::waitKey(0);

    //EXPECT_EQ(0, cv::countNonZero(out_mat != ref_mat));
//}

TEST(Render, OpenCVVersion)
{
    auto frame = cv::Mat(320, 480, CV_8UC3, cv::Scalar::all(0));
    int w     = 30;
    int h     = 40;
    int thick = 2;
    int fs    = 1;
    int ff    = cv::FONT_HERSHEY_SIMPLEX;

    cv::Scalar color(0, 255, 0);
    std::string text = "Some text";

    for (int i = 0; i < 5; ++i)
    {
        int pos_x = 30 + i * 60;
        int pos_y = 40 + i * 50;

        cv::rectangle(frame, cv::Rect(pos_x, pos_y, w, h), color, thick);
        cv::putText(frame, text, cv::Point(pos_x, pos_y), ff, fs, color, thick);
    }

    // ...

    cv::imshow("opencv", frame);
    cv::waitKey(0);
}

TEST(Render, RenderGAPIVersion)
{
    auto frame = cv::Mat(320, 480, CV_8UC3, cv::Scalar::all(0));
    int w     = 30;
    int h     = 40;
    int thick = 2;
    int fs    = 1;
    int ff    = cv::FONT_HERSHEY_SIMPLEX;

    cv::Scalar color(0, 255, 0);
    std::string text = "Some text";

    std::vector<cv::gapi::DrawEvent> events;

    for (int i = 0; i < 5; ++i)
    {
        int pos_x = 30 + i * 60;
        int pos_y = 40 + i * 50;

        events.emplace_back(cv::gapi::RectEvent{pos_x, pos_y, w, h, color, thick});
        events.emplace_back(cv::gapi::TextEvent{text, pos_x, pos_y, ff, fs, color, thick});
    }

    cv::gapi::render(frame, events);

    // ...

    cv::imwrite("/home/atalaman/Pictures/render.png", frame);
    cv::waitKey(0);
}

TEST(Render, RednerText2Points)
{
    auto frame = cv::Mat(600, 1200, CV_8UC3, cv::Scalar::all(0));
    int fs = 3;
    int ff = cv::FONT_HERSHEY_SIMPLEX;
    int thick = 2;
    cv::Scalar color(0, 255, 0);
    std::string text = "TheQuickBrownFoxJumps";

    auto pts = text2Points(text, cv::Point(30, 100), ff, fs, false);
    cv::putText(frame, text, cv::Point(30, 100), ff, fs, color, thick);

    int total_pts = 0;
    for (int i = 0; i < pts.size(); ++i)
    {
        total_pts += pts[i].size();
        for (int j = 0; j < pts[i].size(); ++j)
        {
            auto p0 = pts[i][j];

            p0.x = (p0.x + ((1<<16)>>1)) >> 16;
            p0.y = (p0.y + ((1<<16)>>1)) >> 16;

            cv::circle(frame, p0, 1, cv::Scalar(0, 0, 255), 2);
        }
    }

    std::cout << "total pts = " << total_pts << std::endl;

    cv::imshow("frame", frame);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
}

}
