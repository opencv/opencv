// WARNING: this sample is under construction! Use it on your own risk.

#include <opencv2/contrib/contrib.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>
#include <iomanip>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::gpu;

void help()
{
    cout << "Usage: ./cascadeclassifier <cascade_file> <image_or_video_or_cameraid>\n"               
            "Using OpenCV version " << CV_VERSION << endl << endl;
}

void DetectAndDraw(Mat& img, CascadeClassifier_GPU& cascade);

String cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
String nestedCascadeName = "../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";



template<class T> void convertAndReseize(const T& src, T& gray, T& resized, double scale = 2.0)
{
    if (src.channels() == 3)
        cvtColor( src, gray, CV_BGR2GRAY );
    else
        gray = src;

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));
    if (scale != 1)
        resize(gray, resized, sz);
    else
        resized = gray;
}



int main( int argc, const char** argv )
{        
    if (argc != 3)
        return help(), -1;

    if (cv::gpu::getCudaEnabledDeviceCount() == 0)
        return cerr << "No GPU found or the library is compiled without GPU support" << endl, -1;

    VideoCapture capture;
     
    string cascadeName = argv[1];
    string inputName = argv[2];

    cv::gpu::CascadeClassifier_GPU cascade_gpu;
    if( !cascade_gpu.load( cascadeName ) )
        return cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"" << endl, help(), -1;

    cv::CascadeClassifier cascade_cpu;
    if( !cascade_cpu.load( cascadeName ) )
        return cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"" << endl, help(), -1;
    
    Mat image = imread( inputName);
    if( image.empty() )
        if (!capture.open(inputName))
        {
            int camid = 0;
            sscanf(inputName.c_str(), "%d", &camid);
            if(!capture.open(camid))
                cout << "Can't open source" << endl;
        }
    
    namedWindow( "result", 1 );        
    Size fontSz = cv::getTextSize("T[]", FONT_HERSHEY_SIMPLEX, 1.0, 2, 0);

    Mat frame, frame_cpu, gray_cpu, resized_cpu, faces_downloaded, frameDisp;
    vector<Rect> facesBuf_cpu;

    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;    
    
    /* parameters */
    bool useGPU = true;
    double scale_factor = 1;

    bool visualizeInPlace = false;   
    bool findLargestObject = false;    

    printf("\t<space> - toggle GPU/CPU\n");
    printf("\tL       - toggle lagest faces\n");
    printf("\tV       - toggle visualisation in-place (for GPU only)\n");
    printf("\t1/q     - inc/dec scale\n");
        
    int detections_num;
    for(;;)
    {               
        if( capture.isOpened() )
        {
            capture >> frame;                            
            if( frame.empty())
                break;
        }

        (image.empty() ? frame : image).copyTo(frame_cpu);
        frame_gpu.upload( image.empty() ? frame : image);
        
        convertAndReseize(frame_gpu, gray_gpu, resized_gpu, scale_factor);
        convertAndReseize(frame_cpu, gray_cpu, resized_cpu, scale_factor);

        cv::TickMeter tm;
        tm.start();      

        if (useGPU)
        {
            cascade_gpu.visualizeInPlace = visualizeInPlace;   
            cascade_gpu.findLargestObject = findLargestObject;    

            detections_num = cascade_gpu.detectMultiScale( resized_gpu, facesBuf_gpu ); 
            facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
        
        }
        else /* so use CPU */
        {   
            Size minSize = cascade_gpu.getClassifierSize();
            if (findLargestObject)
            {                
                float ratio = (float)std::min(frame.cols / minSize.width, frame.rows / minSize.height);
                ratio = std::max(ratio / 2.5f, 1.f);
                minSize = Size(cvRound(minSize.width * ratio), cvRound(minSize.height * ratio));                
            }
            
            cascade_cpu.detectMultiScale(resized_cpu, facesBuf_cpu, 1.2, 4, (findLargestObject ? CV_HAAR_FIND_BIGGEST_OBJECT : 0) | CV_HAAR_SCALE_IMAGE, minSize);                            
            detections_num = (int)facesBuf_cpu.size();
        }

        tm.stop();
        printf( "detection time = %g ms\n", tm.getTimeMilli() );

        if (useGPU)
            resized_gpu.download(resized_cpu);

        if (!visualizeInPlace || !useGPU)
            if (detections_num)
            {
                Rect* faces = useGPU ? faces_downloaded.ptr<Rect>() : &facesBuf_cpu[0];                
                for(int i = 0; i < detections_num; ++i)                
                    cv::rectangle(resized_cpu, faces[i], Scalar(255));            
            }
        
        Point text_pos(5, 25);
        int offs = fontSz.height + 5;
        Scalar color = CV_RGB(255, 0, 0);


        cv::cvtColor(resized_cpu, frameDisp, CV_GRAY2BGR);

        char buf[4096];
        sprintf(buf, "%s, FPS = %0.3g", useGPU ? "GPU" : "CPU", 1.0/tm.getTimeSec());                       
        putText(frameDisp, buf, text_pos, FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
        sprintf(buf, "scale = %0.3g, [%d*scale x %d*scale]", scale_factor, frame.cols, frame.rows);                       
        putText(frameDisp, buf, text_pos+=Point(0,offs), FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
        putText(frameDisp, "Hotkeys: space, 1, Q, L, V, Esc", text_pos+=Point(0,offs), FONT_HERSHEY_SIMPLEX, 1.0, color, 2);

        if (findLargestObject)
            putText(frameDisp, "FindLargestObject", text_pos+=Point(0,offs), FONT_HERSHEY_SIMPLEX, 1.0, color, 2);

        if (visualizeInPlace && useGPU)
            putText(frameDisp, "VisualizeInPlace", text_pos+Point(0,offs), FONT_HERSHEY_SIMPLEX, 1.0, color, 2);

        cv::imshow( "result", frameDisp);

        int key = waitKey( 5 );
        if( key == 27)
            break;

        switch (key)
        {
        case (int)' ':  useGPU = !useGPU;  printf("Using %s\n", useGPU ? "GPU" : "CPU");break;
        case (int)'v':  case (int)'V': visualizeInPlace = !visualizeInPlace; printf("VisualizeInPlace = %d\n", visualizeInPlace); break;
        case (int)'l':  case (int)'L': findLargestObject = !findLargestObject;  printf("FindLargestObject = %d\n", findLargestObject); break;
        case (int)'1':  scale_factor*=1.05; printf("Scale factor = %g\n", scale_factor); break;
        case (int)'q':  case (int)'Q':scale_factor/=1.05; printf("Scale factor = %g\n", scale_factor); break;
        }
       
    }    
    return 0;
}



