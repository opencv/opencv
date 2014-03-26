.. _Visual_Debugging_Introduction:

Interactive Visual Debugging of Computer Vision applications
************************************************************

What is the most common way to debug computer vision applications?
Usually the answer is temporary, hacked together, custom code that must be removed from the code for release compilation.

In this tutorial we will show how to use the visual debugging features of the **cvv** module (*opencv2/cvv/cvv.hpp*) instead.


Goals
======

In this tutorial you will learn how to:

* Add cvv debug calls to your application
* Use the visual debug GUI
* Enable and disable the visual debug features during compilation (with zero runtime overhead when disabled)


Code
=====

The example code

* captures images (*highgui*), e.g. from a webcam,
* applies some filters to each image (*imgproc*),
* detects image features and matches them to the previous image (*features2d*).

If the program is compiled without visual debugging (see CMakeLists.txt below) the only result is some information printed to the command line.
We want to demonstrate how much debugging or development functionality is added by just a few lines of *cvv* commands.

.. code-block:: cpp

    // system includes
    #include <getopt.h>
    #include <iostream>
    
    // library includes
    #include <opencv2/highgui/highgui.hpp>
    #include <opencv2/imgproc/imgproc.hpp>
    #include <opencv2/features2d/features2d.hpp>
    
    // Visual debugging
    #include <opencv2/cvv/cvv.hpp>
    
    
    // helper function to convert objects that support operator<<() to std::string
    template<class T> std::string toString(const T& p_arg)
    {
      std::stringstream ss;
    
      ss << p_arg;
    
      return ss.str();
    }
    
    
    void
    usage()
    {
      printf("usage: cvvt [-r WxH]\n");
      printf("-h       print this help\n");
      printf("-r WxH   change resolution to width W and height H\n");
    }
    
    
    int
    main(int argc, char** argv)
    {
    #ifdef CVVISUAL_DEBUGMODE
      std::cout << "Visual debugging is ENABLED" << std::endl;
    #else
      std::cout << "Visual debugging is DISABLED" << std::endl;
    #endif
    
      cv::Size* resolution = nullptr;
    
      // parse options
      const char* optstring = "hr:";
      int opt;
      while ((opt = getopt(argc, argv, optstring)) != -1) {
        switch (opt) {
        case 'h':
          usage();
          return 0;
          break;
        case 'r':
          {
            char dummych;
            resolution = new cv::Size();
            if (sscanf(optarg, "%d%c%d", &resolution->width, &dummych, &resolution->height) != 3) {
              printf("%s not a valid resolution\n", optarg);
              return 1;
            }
          }
          break;
        default:
          usage();
          return 2;
        }
      }
    
      // setup video capture
      cv::VideoCapture capture(0);
      if (!capture.isOpened()) {
        std::cout << "Could not open VideoCapture" << std::endl;
        return 3;
      }
    
      if (resolution) {
        printf("Setting resolution to %dx%d\n", resolution->width, resolution->height);
        capture.set(CV_CAP_PROP_FRAME_WIDTH, resolution->width);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, resolution->height);
      }
    
    
      cv::Mat prevImgGray;
      std::vector<cv::KeyPoint> prevKeypoints;
      cv::Mat prevDescriptors;
    
      int maxFeatureCount = 500;
      cv::ORB detector(maxFeatureCount);
    
      cv::BFMatcher matcher(cv::NORM_HAMMING);
    
      for (int imgId = 0; imgId < 10; imgId++) {
        // capture a frame
        cv::Mat imgRead;
        capture >> imgRead;
        printf("%d: image captured\n", imgId);
    
        std::string imgIdString{"imgRead"};
        imgIdString += toString(imgId);
        cvv::showImage(imgRead, CVVISUAL_LOCATION, imgIdString.c_str());
    
        // convert to grayscale
        cv::Mat imgGray;
        cv::cvtColor(imgRead, imgGray, CV_BGR2GRAY);
        cvv::debugFilter(imgRead, imgGray, CVVISUAL_LOCATION, "to gray");
    
        // filter edges using Canny on smoothed image
        cv::Mat imgGraySmooth;
        cv::GaussianBlur(imgGray, imgGraySmooth, cv::Size(9, 9), 2, 2);
        cvv::debugFilter(imgGray, imgGraySmooth, CVVISUAL_LOCATION, "smoothed");
        cv::Mat imgEdges;
        cv::Canny(imgGraySmooth, imgEdges, 50, 150);
        cvv::showImage(imgEdges, CVVISUAL_LOCATION, "edges");
    
        // dilate edges
        cv::Mat imgEdgesDilated;
        cv::dilate(imgEdges, imgEdgesDilated, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7), cv::Point(3, 3)));
        cvv::debugFilter(imgEdges, imgEdgesDilated, CVVISUAL_LOCATION, "dilated edges");
    
        // detect ORB features
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector(imgGray, cv::noArray(), keypoints, descriptors);
        printf("%d: detected %zd keypoints\n", imgId, keypoints.size());
    
        // match them to previous image (if available)
        if (!prevImgGray.empty()) {
          std::vector<cv::DMatch> matches;
          matcher.match(prevDescriptors, descriptors, matches);
          printf("%d: all matches size=%zd\n", imgId, matches.size());
          std::string allMatchIdString{"all matches "};
          allMatchIdString += toString(imgId-1) + "<->" + toString(imgId);
          cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, allMatchIdString.c_str());
    
          // remove worst (as defined by match distance) bestRatio quantile
          double bestRatio = 0.8;
          std::sort(matches.begin(), matches.end());
          matches.resize(int(bestRatio * matches.size()));
          printf("%d: best matches size=%zd\n", imgId, matches.size());
          std::string bestMatchIdString{"best " + toString(bestRatio) + " matches "};
          bestMatchIdString += toString(imgId-1) + "<->" + toString(imgId);
          cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, bestMatchIdString.c_str());
        }
    
        prevImgGray = imgGray;
        prevKeypoints = keypoints;
        prevDescriptors = descriptors;
      }
    
      cvv::finalShow();
    
      return 0;
    }


.. code-block:: cmake

    cmake_minimum_required(VERSION 2.8)
    
    project(cvvisual_test)
    
    SET(CMAKE_PREFIX_PATH ~/software/opencv/install)
    
    SET(CMAKE_CXX_COMPILER "g++-4.8")
    SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -pthread -Wall -Werror")
    
    # (un)set: cmake -DCVV_DEBUG_MODE=OFF ..
    OPTION(CVV_DEBUG_MODE "cvvisual-debug-mode" ON)
    if(CVV_DEBUG_MODE MATCHES ON)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCVVISUAL_DEBUGMODE")
    endif()
    
    
    FIND_PACKAGE(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    
    add_executable(cvvt main.cpp)
    target_link_libraries(cvvt
      opencv_core opencv_highgui opencv_imgproc opencv_features2d
      opencv_cvv
    )


Explanation
============

#. We compile the program either using the above CmakeLists.txt with Option *CVV_DEBUG_MODE=ON* (*cmake -DCVV_DEBUG_MODE=ON*) or by adding the corresponding define *CVVISUAL_DEBUGMODE* to our compiler (e.g. *g++ -DCVVISUAL_DEBUGMODE*).

#. The first cvv call simply shows the image (similar to *imshow*) with the imgIdString as comment.

   .. code-block:: cpp

     cvv::showImage(imgRead, CVVISUAL_LOCATION, imgIdString.c_str());

   The image is added to the overview tab in the visual debug GUI and the cvv call blocks.
   

   .. image:: images/01_overview_single.jpg
      :alt: Overview with image of first cvv call
      :align: center

   The image can then be selected and viewed

   .. image:: images/02_single_image_view.jpg
      :alt: Display image added through cvv::showImage
      :align: center

   Whenever you want to continue in the code, i.e. unblock the cvv call, you can
   either continue until the next cvv call (*Step*), continue until the last cvv
   call (*>>*) or run the application until it exists (*Close*).

   We decide to press the green *Step* button.


#. The next cvv calls are used to debug all kinds of filter operations, i.e. operations that take a picture as input and return a picture as output.

   .. code-block:: cpp

       cvv::debugFilter(imgRead, imgGray, CVVISUAL_LOCATION, "to gray");

   As with every cvv call, you first end up in the overview.

   .. image:: images/03_overview_two.jpg
      :alt: Overview with two cvv calls after pressing Step
      :align: center

   We decide not to care about the conversion to gray scale and press *Step*.

   .. code-block:: cpp

       cvv::debugFilter(imgGray, imgGraySmooth, CVVISUAL_LOCATION, "smoothed");

   If you open the filter call, you will end up in the so called "DefaultFilterView".
   Both images are shown next to each other and you can (synchronized) zoom into them.

   .. image:: images/04_default_filter_view.jpg
      :alt: Default filter view displaying a gray scale image and its corresponding GaussianBlur filtered one
      :align: center

   When you go to very high zoom levels, each pixel is annotated with its numeric values.

   .. image:: images/05_default_filter_view_high_zoom.jpg
      :alt: Default filter view at very high zoom levels
      :align: center

   We press *Step* twice and have a look at the dilated image.

   .. code-block:: cpp

       cvv::debugFilter(imgEdges, imgEdgesDilated, CVVISUAL_LOCATION, "dilated edges");

   The DefaultFilterView showing both images

   .. image:: images/06_default_filter_view_edges.jpg
      :alt: Default filter view showing an edge image and the image after dilate()
      :align: center

   Now we use the *View* selector in the top right and select the "DualFilterView".
   We select "Changed Pixels" as filter and apply it (middle image).

   .. image:: images/07_dual_filter_view_edges.jpg
      :alt: Dual filter view showing an edge image and the image after dilate()
      :align: center

   After we had a close look at these images, perhaps using different views, filters or other GUI features, we decide to let the program run through. Therefore we press the yellow *>>* button.

   The program will block at

   .. code-block:: cpp

      cvv::finalShow();

   and display the overview with everything that was passed to cvv in the meantime.

   .. image:: images/08_overview_all.jpg
      :alt: Overview displaying all cvv calls up to finalShow()
      :align: center

#. The cvv debugDMatch call is used in a situation where there are two images each with a set of descriptors that are matched to each other.

   We pass both images, both sets of keypoints and their matching to the visual debug module.

   .. code-block:: cpp

       cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, allMatchIdString.c_str());

   Since we want to have a look at matches, we use the filter capabilities (*#type match*) in the overview to only show match calls.

   .. image:: images/09_overview_filtered_type_match.jpg
      :alt: Overview displaying only match calls
      :align: center

   We want to have a closer look at one of them, e.g. to tune our parameters that use the matching.
   The view has various settings how to display keypoints and matches.
   Furthermore, there is a mouseover tooltip.

   .. image:: images/10_line_match_view.jpg
      :alt: Line match view
      :align: center

   We see (visual debugging!) that there are many bad matches.
   We decide that only 70% of the matches should be shown - those 70% with the lowest match distance.

   .. image:: images/11_line_match_view_portion_selector.jpg
      :alt: Line match view showing the best 70% matches, i.e. lowest match distance
      :align: center

   Having successfully reduced the visual distraction, we want to see more clearly what changed between the two images.
   We select the "TranslationMatchView" that shows to where the keypoint was matched in a different way.

   .. image:: images/12_translation_match_view_portion_selector.jpg
      :alt: Translation match view
      :align: center

   It is easy to see that the cup was moved to the left during the two images.

   Although, cvv is all about interactively *seeing* the computer vision bugs, this is complemented by a "RawView" that allows to have a look at the underlying numeric data.

   .. image:: images/13_raw_view.jpg
      :alt: Raw view of matches
      :align: center

#. There are many more useful features contained in the cvv GUI. For instance, one can group the overview tab.

   .. image:: images/14_overview_group_by_line.jpg
      :alt: Overview grouped by call line
      :align: center


Result
=======

* By adding a view expressive lines to our computer vision program we can interactively debug it through different visualizations.
* Once we are done developing/debugging we do not have to remove those lines. We simply disable cvv debugging (*cmake -DCVV_DEBUG_MODE=OFF* or g++ without *-DCVVISUAL_DEBUGMODE*) and our programs runs without any debug overhead.

Enjoy computer vision!
