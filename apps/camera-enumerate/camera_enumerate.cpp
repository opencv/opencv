// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <iostream>

#include <opencv2/videoio.hpp>
#include <opencv2/videoio/registry.hpp>

static void dumpCameraDevices()
{
    const std::vector<cv::VideoDeviceInfo> devices = cv::videoio_registry::enumerateDevices();
    std::cout << "Camera devices available via cv::VideoCapture(index):" << std::endl;
    for (size_t i = 0; i < devices.size(); i++)
    {
        std::cout << "    index=" << devices[i].cam_idx
                  << " (" << cv::videoio_registry::getBackendName(devices[i].backend) << ")"
                  << " -> " << devices[i].cam_name << std::endl;
    }
    std::cout << "Total available: " << devices.size() << std::endl;
}

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser(argc, argv,
        "{ help h usage ? | | show this help message }"
    );

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    dumpCameraDevices();
    return 0;
}
