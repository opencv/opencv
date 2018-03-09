package org.opencv.highgui;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;

/**
 * This class was designed to create and manipulate
 * the Windows to be used by the HighGui class.
 */
public final class ImageWindow {

    public final static int WINDOW_NORMAL = 0;
    public final static int WINDOW_AUTOSIZE = 1;

    public String name;
    public Mat img = null;
    public Boolean alreadyUsed = false;
    public Boolean imgToBeResized = false;
    public Boolean windowToBeResized = false;
    public Boolean positionToBeChanged = false;
    public JFrame frame = null;
    public JLabel lbl = null;
    public int flag;
    public int x = -1;
    public int y = -1;
    public int width = -1;
    public int height = -1;

    public ImageWindow(String name, Mat img) {
        this.name = name;
        this.img = img;
        this.flag = WINDOW_NORMAL;
    }

    public ImageWindow(String name, int flag) {
        this.name = name;
        this.flag = flag;
    }

    public static Size keepAspectRatioSize(int original_width, int original_height, int bound_width, int bound_height) {

        int new_width = original_width;
        int new_height = original_height;

        if (original_width > bound_width) {
            new_width = bound_width;
            new_height = (new_width * original_height) / original_width;
        }

        if (new_height > bound_height) {
            new_height = bound_height;
            new_width = (new_height * original_width) / original_height;
        }

        return new Size(new_width, new_height);
    }

    public void setMat(Mat img) {

        this.img = img;
        this.alreadyUsed = false;

        if (imgToBeResized) {
            resizeImage();
            imgToBeResized = false;
        }

    }

    public void setFrameLabelVisible(JFrame frame, JLabel lbl) {
        this.frame = frame;
        this.lbl = lbl;

        if (windowToBeResized) {
            lbl.setPreferredSize(new Dimension(width, height));
            windowToBeResized = false;
        }

        if (positionToBeChanged) {
            frame.setLocation(x, y);
            positionToBeChanged = false;
        }

        frame.add(lbl);
        frame.pack();
        frame.setVisible(true);
    }

    public void setNewDimension(int width, int height) {

        if (this.width != width || this.height != height) {
            this.width = width;
            this.height = height;

            if (img != null) {
                resizeImage();
            } else {
                imgToBeResized = true;
            }

            if (lbl != null) {
                lbl.setPreferredSize(new Dimension(width, height));
            } else {
                windowToBeResized = true;
            }
        }
    }

    public void setNewPosition(int x, int y) {
        if (this.x != x || this.y != y) {
            this.x = x;
            this.y = y;

            if (frame != null) {
                frame.setLocation(x, y);
            } else {
                positionToBeChanged = true;
            }
        }
    }

    private void resizeImage() {
        if (flag == WINDOW_NORMAL) {
            Size tmpSize = keepAspectRatioSize(img.width(), img.height(), width, height);
            Imgproc.resize(img, img, tmpSize, 0, 0, Imgproc.INTER_LINEAR_EXACT);
        }
    }
}
