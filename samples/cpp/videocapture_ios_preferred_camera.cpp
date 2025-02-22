/**
* @file videocapture_ios_preferred_camera.cpp
* @brief A starter sample for using OpenCV VideoCapture with capture devices, video files, or image sequences.
*
* Modified to automatically detect and open a specific camera on macOS.
*/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <sstream>

using namespace cv;
using namespace std;

// Function to list connected cameras on macOS
vector<string> getUSBCameras() {
    vector<string> cameraNames;
    FILE* pipe = popen("system_profiler SPUSBDataType", "r");
    if (!pipe) {
        cerr << "Error: Could not run system_profiler command." << endl;
        return cameraNames;
    }

    char buffer[256];
    string currentCamera;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        string line(buffer);
        size_t pos = line.find("Camera");
        if (pos != string::npos) {
            currentCamera = line.substr(0, line.find(" ("));  // Extract name before '('
            cameraNames.push_back(currentCamera);
        }
    }
    pclose(pipe);

    return cameraNames;
}

// Function to select a camera by name
string selectCamera(const vector<string>& cameras, const string& preferredCamera) {
    for (const string& camera : cameras) {
        if (camera.find(preferredCamera) != string::npos) {
            return camera;
        }
    }
    return cameras.empty() ? "" : cameras[0];  // Default to the first camera if no match
}

// Process video frames
int process(VideoCapture& capture) {
    int n = 0;
    char filename[200];
    string window_name = "video | q or esc to quit";
    cout << "Press space to save a picture. q or esc to quit" << endl;
    namedWindow(window_name, WINDOW_KEEPRATIO);
    Mat frame;

    while (true) {
        capture >> frame;
        if (frame.empty())
            break;

        imshow(window_name, frame);
        char key = (char)waitKey(30);

        switch (key) {
            case 'q':
            case 'Q':
            case 27:
                return 0;
            case ' ':
                snprintf(filename, sizeof(filename), "filename%.3d.jpg", n++);
                imwrite(filename, frame);
                cout << "Saved " << filename << endl;
                break;
            default:
                break;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, "{help h||}{@input||}");
    if (parser.has("help")) {
        cout << "Usage:\n" << argv[0] << " <video file, image sequence, or camera device>\n";
        return 0;
    }

    string input = parser.get<string>("@input");

    // If no input is given, auto-detect cameras
    if (input.empty()) {
        vector<string> cameras = getUSBCameras();
        if (cameras.empty()) {
            cerr << "No cameras found!" << endl;
            return 1;
        }

        // Select camera (change the name if needed)
        input = selectCamera(cameras, "FaceTime HD Camera");
        cout << "Using detected camera: " << input << endl;
    }

    // Open video capture using AVFoundation
    VideoCapture capture(input, cv::CAP_AVFOUNDATION);
    if (!capture.isOpened()) {
        cerr << "Failed to open the video device, video file, or image sequence!\n" << endl;
        return 1;
    }

    return process(capture);
}
