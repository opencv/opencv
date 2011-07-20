.. _hough_lines:

Hough Line Transform
*********************

Goal
=====

In this tutorial you will learn how to:

* Use the OpenCV functions :hough_lines:`HoughLines <>` and :hough_lines_p:`HoughLinesP <>` to detect lines in an image.
  
Theory
=======

.. note::
   The explanation below belongs to the book **Learning OpenCV** by Bradski and Kaehler.

Hough Line Transform
---------------------
#. The Hough Line Transform is a transform used to detect straight lines. 
#. To apply the Transform, first an edge detection pre-processing is desirable.

How does it work?
^^^^^^^^^^^^^^^^^^

#. As you know, a line in the image space can be expressed with two variables. For example:
 
   a. In the **Cartesian coordinate system:**  Parameters: :math:`(m,b)`.
   b. In the **Polar coordinate system:** Parameters: :math:`(r,\theta)`

   .. image:: images/Hough_Lines_Tutorial_Theory_0.jpg
      :alt: Line variables
      :align: center 

   For Hough Transforms, we will express lines in the *Polar system*. Hence, a line equation can be written as: 

   .. math::

      y = \left ( -\dfrac{\cos \theta}{\sin \theta} \right ) x + \left ( \dfrac{r}{\sin \theta} \right ) 

  Arranging the terms: :math:`r = x \cos \theta + y \sin \theta`

#. In general for each point :math:`(x_{0}, y_{0})`, we can define the family of lines that goes through that point as:

   .. math::
   
      r_{\theta} = x_{0} \cdot \cos \theta  + y_{0} \cdot \sin \theta

   Meaning that each pair :math:`(r_{\theta},\theta)` represents each line that passes by :math:`(x_{0}, y_{0})`. 

#. If for a given :math:`(x_{0}, y_{0})` we plot the family of lines that goes through it, we get a sinusoid. For instance, for :math:`x_{0} = 8` and :math:`y_{0} = 6` we get the following plot (in a plane :math:`\theta` - :math:`r`):

   .. image:: images/Hough_Lines_Tutorial_Theory_1.jpg
      :alt: Polar plot of a the family of lines of a point
      :align: center 

   We consider only points such that :math:`r > 0` and :math:`0< \theta < 2 \pi`. 

#. We can do the same operation above for all the points in an image. If the curves of two different points intersect in the plane :math:`\theta` - :math:`r`, that means that both points belong to a same line. For instance, following with the example above and drawing the plot for two more points: :math:`x_{1} = 9`, :math:`y_{1} = 4` and :math:`x_{2} = 12`, :math:`y_{2} = 3`, we get:

   .. image:: images/Hough_Lines_Tutorial_Theory_2.jpg
      :alt: Polar plot of the family of lines for three points
      :align: center 

   The three plots intersect in one single point :math:`(0.925, 9.6)`, these coordinates are the parameters (:math:`\theta, r`) or the line in which :math:`(x_{0}, y_{0})`, :math:`(x_{1}, y_{1})` and :math:`(x_{2}, y_{2})` lay. 

#. What does all the stuff above mean? It means that in general, a line can be *detected* by finding the number of intersections between curves.The more curves intersecting means that the line represented by that intersection have more points. In general, we can define a *threshold* of the minimum number of intersections needed to *detect* a line.
 
#. This is what the Hough Line Transform does. It keeps track of the intersection between curves of every point in the image. If the number of intersections is above some *threshold*, then it declares it as a line with the parameters :math:`(\theta, r_{\theta})` of the intersection point.

Standard and Probabilistic Hough Line Transform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OpenCV implements two kind of Hough Line Transforms: 

a. **The Standard Hough Transform**

  * It consists in pretty much what we just explained in the previous section. It gives you as result a vector of couples :math:`(\theta, r_{\theta})`

  * In OpenCV it is implemented with the function :hough_lines:`HoughLines <>`

b. **The Probabilistic Hough Line Transform**

  * A more efficient implementation of the Hough Line Transform. It gives as output the extremes of the detected lines :math:`(x_{0}, y_{0}, x_{1}, y_{1})`

  * In OpenCV it is implemented with the function :hough_lines_p:`HoughLinesP <>`

Code
======

.. |TutorialHoughLinesSimpleDownload| replace:: here
.. _TutorialHoughLinesSimpleDownload: https://code.ros.org/svn/opencv/trunk/opencv/samples/cpp/houghlines.cpp
.. |TutorialHoughLinesFancyDownload| replace:: here
.. _TutorialHoughLinesFancyDownload: https://code.ros.org/svn/opencv/trunk/opencv/samples/cpp/tutorial_code/ImgTrans/HoughLines_Demo.cpp


#. **What does this program do?**
 
   * Loads an image
   * Applies either a *Standard Hough Line Transform* or a *Probabilistic Line Transform*. 
   * Display the original image and the detected line in two windows.

#. The sample code that we will explain can be downloaded from  |TutorialHoughLinesSimpleDownload|_. A slightly fancier version (which shows both Hough standard and probabilistic with trackbars for changing the threshold values) can be found  |TutorialHoughLinesFancyDownload|_.

.. code-block:: cpp 

   #include "opencv2/highgui/highgui.hpp"
   #include "opencv2/imgproc/imgproc.hpp"

   #include <iostream>

   using namespace cv;
   using namespace std;

   void help()
   {
    cout << "\nThis program demonstrates line finding with the Hough transform.\n"
            "Usage:\n"
            "./houghlines <image_name>, Default is pic1.jpg\n" << endl;
   }

   int main(int argc, char** argv)
   {
    const char* filename = argc >= 2 ? argv[1] : "pic1.jpg";

    Mat src = imread(filename, 0);
    if(src.empty())
    {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }

    Mat dst, cdst;
    Canny(src, dst, 50, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);

    #if 0
     vector<Vec2f> lines;
     HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

     for( size_t i = 0; i < lines.size(); i++ )
     {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
     }
    #else
     vector<Vec4i> lines;
     HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
     for( size_t i = 0; i < lines.size(); i++ )
     {
       Vec4i l = lines[i];
       line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
     }
    #endif
    imshow("source", src);
    imshow("detected lines", cdst);

    waitKey();

    return 0;
   }

Explanation
=============

#. Load an image

   .. code-block:: cpp

      Mat src = imread(filename, 0);
      if(src.empty())
      {
        help();
        cout << "can not open " << filename << endl;
        return -1;
      }

#. Detect the edges of the image by using a Canny detector

   .. code-block:: cpp

      Canny(src, dst, 50, 200, 3);

   Now we will apply the Hough Line Transform. We will explain how to use both OpenCV functions available for this purpose:

#. **Standard Hough Line Transform**

   a. First, you apply the Transform:

      .. code-block:: cpp

         vector<Vec2f> lines;
         HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

      with the following arguments:

      * *dst*: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
      * *lines*: A vector that will store the parameters :math:`(r,\theta)` of the detected lines
      * *rho* : The resolution of the parameter :math:`r` in pixels. We use **1** pixel.
      * *theta*: The resolution of the parameter :math:`\theta` in radians. We use **1 degree** (CV_PI/180)
      * *threshold*: The minimum number of intersections to "*detect*" a line
      * *srn* and *stn*: Default parameters to zero. Check OpenCV reference for more info. 

   b. And then you display the result by drawing the lines. 

      .. code-block:: cpp

         for( size_t i = 0; i < lines.size(); i++ )
         {
           float rho = lines[i][0], theta = lines[i][1];
           Point pt1, pt2;
           double a = cos(theta), b = sin(theta);
           double x0 = a*rho, y0 = b*rho;
           pt1.x = cvRound(x0 + 1000*(-b));
           pt1.y = cvRound(y0 + 1000*(a));
           pt2.x = cvRound(x0 - 1000*(-b));
           pt2.y = cvRound(y0 - 1000*(a));
           line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
         }

#. **Probabilistic Hough Line Transform**

   a. First you apply the transform:

      .. code-block:: cpp

         vector<Vec4i> lines;
         HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );

      with the arguments:
 
      * *dst*: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
      * *lines*: A vector that will store the parameters :math:`(x_{start}, y_{start}, x_{end}, y_{end})` of the detected lines
      * *rho* : The resolution of the parameter :math:`r` in pixels. We use **1** pixel.
      * *theta*: The resolution of the parameter :math:`\theta` in radians. We use **1 degree** (CV_PI/180)
      * *threshold*: The minimum number of intersections to "*detect*" a line
      * *minLinLength*: The minimum number of points that can form a line. Lines with less than this number of points are disregarded. 
      * *maxLineGap*: The maximum gap between two points to be considered in the same line. 

   b. And then you display the result by drawing the lines.

      .. code-block:: cpp

         for( size_t i = 0; i < lines.size(); i++ )
         {
           Vec4i l = lines[i];
           line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
         }


#. Display the original image and the detected lines: 

   .. code-block:: cpp

      imshow("source", src);
      imshow("detected lines", cdst);

#. Wait until the user exits the program

   .. code-block:: cpp

      waitKey();


Result
=======

.. note::
  
   The results below are obtained using the slightly fancier version we mentioned in the *Code* section. It still implements the same stuff as above, only adding the Trackbar for the Threshold.

Using an input image such as:

.. image:: images/Hough_Lines_Tutorial_Original_Image.jpg
   :alt: Result of detecting lines with Hough Transform
   :align: center 
 
We get the following result by using the Probabilistic Hough Line Transform:

.. image:: images/Hough_Lines_Tutorial_Result.jpg
   :alt: Result of detecting lines with Hough Transform
   :align: center 

You may observe that the number of lines detected vary while you change the *threshold*. The explanation is sort of evident: If you establish a higher threshold, fewer lines will be detected (since you will need more points to declare a line detected). 

