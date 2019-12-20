Frequently Asked Questions {#faq}
==========================

-   **What is InputArray?**

        It can be seen that almost all OpenCV functions receive InputArray type.
        What is it, and how can I understand the actual input types of parameters?


    This is the proxy class for passing read-only input arrays into OpenCV functions.

    Inside a function you should use cv::_InputArray::getMat() method to construct
a matrix header for the array (without copying data). cv::_InputArray::kind() can be used to distinguish Mat from vector<> etc.
but normally it is not needed.

   for more information see cv::_InputArray

-   **Which is more efficient, use contourArea() or count number of ROI non-zero pixels?**

        In a case where you only want relative areas, which one is faster to compute:
        calculate a contour area or count the number of ROI non-zero pixels?

    cv::contourArea() uses Green formula (http://en.wikipedia.org/wiki/Green's_theorem) to compute the area, therefore its complexity is O(contour_number_of_vertices). Counting non-zero pixels in the ROI is O(roi_width*roi_height) algorithm, i.e. much slower. Note, however, that because of finite, and quite low, resolution of the raster grid, the two algorithms will give noticeably different results. For large and square-like contours the error will be minimal. For small and/or oblong contours the error can be quite large.
