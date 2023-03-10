// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>

using namespace cv;

namespace
{
class ROISelector
{
  public:
    Rect select(const String &windowName, Mat img, bool showCrossair = true, bool fromCenter = true, bool printNotice = true)
    {
        if(printNotice)
        {
            // show notice to user
            printf("Select a ROI and then press SPACE or ENTER button!\n");
            printf("Cancel the selection process by pressing c button!\n");
        }

        key = 0;
        imageSize = img.size();

        // set the drawing mode
        selectorParams.drawFromCenter = fromCenter;

        // show the image and give feedback to user
        imshow(windowName, img);

        // copy the data, rectangle should be drawn in the fresh image
        selectorParams.image = img.clone();

        // select the object
        setMouseCallback(windowName, mouseHandler, (void*)this);

        // end selection process on SPACE (32) ESC (27) or ENTER (13)
        while (!(key == 32 || key == 27 || key == 13))
        {
            // draw the selected object
            rectangle(selectorParams.image, selectorParams.box, Scalar(255, 0, 0), 2, 1);

            // draw cross air in the middle of bounding box
            if (showCrossair)
            {
                // horizontal line
                line(selectorParams.image,
                     Point((int)selectorParams.box.x,
                           (int)(selectorParams.box.y + selectorParams.box.height / 2)),
                     Point((int)(selectorParams.box.x + selectorParams.box.width),
                           (int)(selectorParams.box.y + selectorParams.box.height / 2)),
                     Scalar(255, 0, 0), 2, 1);

                // vertical line
                line(selectorParams.image,
                     Point((int)(selectorParams.box.x + selectorParams.box.width / 2),
                           (int)selectorParams.box.y),
                     Point((int)(selectorParams.box.x + selectorParams.box.width / 2),
                           (int)(selectorParams.box.y + selectorParams.box.height)),
                     Scalar(255, 0, 0), 2, 1);
            }

            // show the image bounding box
            imshow(windowName, selectorParams.image);

            // reset the image
            selectorParams.image = img.clone();

            // get keyboard event
            key = waitKey(30);

            if (key == 'c' || key == 'C')//cancel selection
            {
                selectorParams.box = Rect();
                break;
            }
        }

        //cleanup callback
        setMouseCallback(windowName, emptyMouseHandler, NULL);

        return selectorParams.box;
    }

    void select(const String &windowName, Mat img, std::vector<Rect> &boundingBoxes,
                bool showCrosshair = true, bool fromCenter = true, bool printNotice = true)
    {
        if(printNotice)
        {
            printf("Finish the selection process by pressing ESC button!\n");
        }
        boundingBoxes.clear();
        key = 0;

        // while key is not ESC (27)
        for (;;)
        {
            Rect temp = select(windowName, img, showCrosshair, fromCenter, printNotice);
            if (key == 27)
                break;
            if (temp.width > 0 && temp.height > 0)
                boundingBoxes.push_back(temp);
        }
    }

    struct handlerT
    {
        // basic parameters
        bool isDrawing;
        Rect2d box;
        Mat image;
        Point2f startPos;

        // parameters for drawing from the center
        bool drawFromCenter;

        // initializer list
        handlerT() : isDrawing(false), drawFromCenter(true){};
    } selectorParams;

  private:
    static void emptyMouseHandler(int, int, int, int, void*)
    {
    }

    static void mouseHandler(int event, int x, int y, int flags, void *param)
    {
        ROISelector *self = static_cast<ROISelector *>(param);
        self->opencv_mouse_callback(event, x, y, flags);
    }

    void opencv_mouse_callback(int event, int x, int y, int)
    {
        switch (event)
        {
        // update the selected bounding box
        case EVENT_MOUSEMOVE:
            if (selectorParams.isDrawing)
            {
                if (selectorParams.drawFromCenter)
                {
                    // limit half extends to imageSize
                    float halfWidth = std::min(std::min(
                            std::abs(x - selectorParams.startPos.x),
                            selectorParams.startPos.x),
                            imageSize.width - selectorParams.startPos.x);
                    float halfHeight = std::min(std::min(
                            std::abs(y - selectorParams.startPos.y),
                            selectorParams.startPos.y),
                            imageSize.height - selectorParams.startPos.y);

                    selectorParams.box.width = halfWidth * 2;
                    selectorParams.box.height = halfHeight * 2;
                    selectorParams.box.x = selectorParams.startPos.x - halfWidth;
                    selectorParams.box.y = selectorParams.startPos.y - halfHeight;

                }
                else
                {
                    // limit x and y to imageSize
                    int lx = std::min(std::max(x, 0), imageSize.width);
                    int by = std::min(std::max(y, 0), imageSize.height);
                    selectorParams.box.width = std::abs(lx - selectorParams.startPos.x);
                    selectorParams.box.height = std::abs(by - selectorParams.startPos.y);
                    selectorParams.box.x = std::min((float)lx, selectorParams.startPos.x);
                    selectorParams.box.y = std::min((float)by, selectorParams.startPos.y);
                }
            }
            break;

        // start to select the bounding box
        case EVENT_LBUTTONDOWN:
            selectorParams.isDrawing = true;
            selectorParams.box = Rect2d(x, y, 0, 0);
            selectorParams.startPos = Point2f((float)x, (float)y);
            break;

        // cleaning up the selected bounding box
        case EVENT_LBUTTONUP:
            selectorParams.isDrawing = false;
            if (selectorParams.box.width < 0)
            {
                selectorParams.box.x += selectorParams.box.width;
                selectorParams.box.width *= -1;
            }
            if (selectorParams.box.height < 0)
            {
                selectorParams.box.y += selectorParams.box.height;
                selectorParams.box.height *= -1;
            }
            break;
        }
    }

    // save the keypressed character
    int key;
    Size imageSize;
};
}

Rect cv::selectROI(InputArray img, bool showCrosshair, bool fromCenter, bool printNotice)
{
    ROISelector selector;
    return selector.select("ROI selector", img.getMat(), showCrosshair, fromCenter, printNotice);
}

Rect cv::selectROI(const String& windowName, InputArray img, bool showCrosshair, bool fromCenter, bool printNotice)
{
    ROISelector selector;
    return selector.select(windowName, img.getMat(), showCrosshair, fromCenter, printNotice);
}

void cv::selectROIs(const String& windowName, InputArray img,
                             std::vector<Rect>& boundingBox, bool showCrosshair, bool fromCenter, bool printNotice)
{
    ROISelector selector;
    selector.select(windowName, img.getMat(), boundingBox, showCrosshair, fromCenter, printNotice);
}
