Histograms - 2: Histogram Equalization {#tutorial_js_histogram_equalization}
======================================

Goal
----

-   We will learn the concepts of histogram equalization and use it to improve the contrast of our
    images.

Theory
------

Consider an image whose pixel values are confined to some specific range of values only. For eg,
brighter image will have all pixels confined to high values. But a good image will have pixels from
all regions of the image. So you need to stretch this histogram to either ends (as given in below
image, from wikipedia) and that is what Histogram Equalization does (in simple words). This normally
improves the contrast of the image.

![image](images/histogram_equalization.png)

I would recommend you to read the wikipedia page on [Histogram
Equalization](http://en.wikipedia.org/wiki/Histogram_equalization) for more details about it. It has
a very good explanation with worked out examples, so that you would understand almost everything
after reading that.

Histograms Equalization in OpenCV
---------------------------------

We use the function: **cv.equalizeHist (src, dst)**

@param src      source 8-bit single channel image.
@param dst      destination image of the same size and type as src.

Try it
------

\htmlonly
<iframe src="../../js_histogram_equalization_equalizeHist.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly

CLAHE (Contrast Limited Adaptive Histogram Equalization)
--------------------------------------------------------

In **adaptive histogram equalization**, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region
(unless there is noise). If noise is there, it will be amplified. To avoid this, **contrast limiting** is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.

We use the class: **cv.CLAHE (clipLimit = 40, tileGridSize = new cv.Size(8, 8))**

@param clipLimit      threshold for contrast limiting.
@param tileGridSize   size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles. tileGridSize defines the number of tiles in row and column.

@note Don't forget to delete CLAHE!

Try it
------

\htmlonly
<iframe src="../../js_histogram_equalization_createCLAHE.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly