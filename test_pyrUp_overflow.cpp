#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    try {
        // Create a small test image
        Mat image(100, 100, CV_8UC3, Scalar(128, 128, 128));
        
        cout << "Starting pyrUp sequence test..." << endl;
        cout << "Initial image size: " << image.cols << "x" << image.rows << endl;
        
        Mat current = image.clone();
        
        // Keep calling pyrUp until memory overflow
        for (int i = 0; i < 20; i++) {
            auto start = chrono::high_resolution_clock::now();
            
            Mat next;
            pyrUp(current, next);
            
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            
            cout << "Iteration " << i+1 << ": " 
                 << current.cols << "x" << current.rows 
                 << " -> " << next.cols << "x" << next.rows
                 << " (Memory: " << (size_t)next.cols * next.rows * next.channels() << " bytes)"
                 << " (Time: " << duration.count() << "ms)" << endl;
            
            current = next;
            
            // Stop if image gets too large (before crash)
            if (current.cols > 25600 || current.rows > 25600) {
                cout << "Stopping before potential crash..." << endl;
                break;
            }
        }
        
        cout << "Final image size: " << current.cols << "x" << current.rows << endl;
        cout << "Test completed successfully!" << endl;
        
    } catch (const cv::Exception& e) {
        cout << "OpenCV Error: " << e.what() << endl;
        return -1;
    } catch (const std::exception& e) {
        cout << "Standard Error: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}
