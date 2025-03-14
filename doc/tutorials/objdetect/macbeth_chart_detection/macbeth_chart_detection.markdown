Detecting colorcheckers{#tutorial_macbeth_chart_detection}
===========================

In this tutorial you will learn how to use the 'mcc' module to detect colorcharts in a image.
Here we will only use the basic detection algorithm and an improved version that enhances accuracy using a neural network.

Source Code of the sample
-----------

```
run
<path_of_your_opencv_build_directory>/bin/example_cpp_macbeth_chart_detection -t=<type_of_chart> -v=<optional_path_to_video_if_not_provided_webcam_will_be_used.mp4> --ci=<optional_camera_id_needed_only_if_video_not_provided> --nc=<optional_maximum_number_of_charts_to_look_for>
```

* -t=#  is the chart type where 0 (Standard), 1 (DigitalSG), 2 (Vinyl)
* --ci=#  is the camera ID where 0 (default is the main camera), 1 (secondary camera) etc
* --nc=#  By default its values is 1 which means only the best chart will be detected

Examples:

```
Run a movie on a standard macbeth chart:
/home/opencv/build/bin/example_cpp_macbeth_chart_detection -t=0 -v=mcc24.mp4

Or run on a vinyl macbeth chart from camera 0:
/home/opencv/build/bin/example_cpp_macbeth_chart_detection -t=2 --ci=0

Or run on a vinyl macbeth chart, detecting the best 5 charts(Detections can be less than 5 but never more):
/home/opencv/build/bin/example_cpp_macbeth_chart_detection -t=2 --ci=0 --nc=5

```

```
Simple run on CPU with neural network (GPU wont be used)
/home/opencv/build/bin/example_cpp_macbeth_chart_detection -t=0 -m=/home/model.pb --pb=/home/model.pbtxt -v=mcc24.mp4
```

```
To run on GPU with neural network
/home/opencv/build/bin/example_cpp_macbeth_chart_detection -t=0 -m=/home/model.pb --pb=/home/model.pbtxt -v=mcc24.mp4 --use_gpu

To run on GPU with neural network and detect the best 5 charts (Detections can be less than 5 but not more than 5)
/home/opencv/build/bin/example_cpp_macbeth_chart_detection -t=0 -m=/home/model.pb --pb=/home/model.pbtxt -v=mcc24.mp4 --use_gpu --nc=5
```

@includelineno samples/cpp/macbeth_chart_detection.cpp

Explanation
-----------

-#  **Set header and namespaces**
    @code{.cpp}
    #include <opencv2/mcc.hpp>
    using namespace std;
    using namespace cv;
    using namespace mcc;
    @endcode

    If you want you can set the namespace like the code above.
-#  **Create the detector object**
    @code{.cpp}
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    @endcode
-#  **Or create the detector object with neural network**
    @code{.cpp}
    Ptr<CCheckerDetector> detector = CCheckerDetector::create(net);
    @endcode

    This is just to create the object.
-#  **Run the detector**
    @code{.cpp}
    detector->process(image, chartType);
    @endcode

    If the detector successfully detects atleast one chart, it return true otherwise it returns false. In the above given code we print a failure message if no chart were detected. Otherwise if it were successful, the list of colorcharts is stored inside the detector itself, we will see in the next step on how to extract it. By default it will detect atmost one chart, but you can tune the third parameter, nc(maximum number of charts), for detecting more charts.
-#  **Get List of ColorCheckers**
    @code{.cpp}
    std::vector<cv::Ptr<mcc::CChecker>> checkers;
    detector->getListColorChecker(checkers);
    @endcode

    All the colorcheckers that were detected are now stored in the 'checkers' vector.

-#  **Draw the colorcheckers back to the image**
    @code{.cpp}

    detector->draw(checkers, image);
    @endcode
