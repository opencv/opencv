Foreground Extraction using GrabCut Algorithm {#tutorial_js_grabcut}
=========================================================

Goal
----

-   We will learn GrabCut algorithm to extract foreground in images

Theory
------

GrabCut algorithm was designed by Carsten Rother, Vladimir Kolmogorov & Andrew Blake from Microsoft
Research Cambridge, UK. in their paper, ["GrabCut": interactive foreground extraction using iterated
graph cuts](http://dl.acm.org/citation.cfm?id=1015720) . An algorithm was needed for foreground
extraction with minimal user interaction, and the result was GrabCut.

How it works from user point of view ? Initially user draws a rectangle around the foreground region
(foreground region should be completely inside the rectangle). Then algorithm segments it
iteratively to get the best result. Done. But in some cases, the segmentation won't be fine, like,
it may have marked some foreground region as background and vice versa. In that case, user need to
do fine touch-ups. Just give some strokes on the images where some faulty results are there. Strokes
basically says *"Hey, this region should be foreground, you marked it background, correct it in next
iteration"* or its opposite for background. Then in the next iteration, you get better results.

What happens in background ?

-   User inputs the rectangle. Everything outside this rectangle will be taken as sure background
    (That is the reason it is mentioned before that your rectangle should include all the
    objects). Everything inside rectangle is unknown. Similarly any user input specifying
    foreground and background are considered as hard-labelling which means they won't change in
    the process.
-   Computer does an initial labelling depeding on the data we gave. It labels the foreground and
    background pixels (or it hard-labels)
-   Now a Gaussian Mixture Model(GMM) is used to model the foreground and background.
-   Depending on the data we gave, GMM learns and create new pixel distribution. That is, the
    unknown pixels are labelled either probable foreground or probable background depending on its
    relation with the other hard-labelled pixels in terms of color statistics (It is just like
    clustering).
-   A graph is built from this pixel distribution. Nodes in the graphs are pixels. Additional two
    nodes are added, **Source node** and **Sink node**. Every foreground pixel is connected to
    Source node and every background pixel is connected to Sink node.
-   The weights of edges connecting pixels to source node/end node are defined by the probability
    of a pixel being foreground/background. The weights between the pixels are defined by the edge
    information or pixel similarity. If there is a large difference in pixel color, the edge
    between them will get a low weight.
-   Then a mincut algorithm is used to segment the graph. It cuts the graph into two separating
    source node and sink node with minimum cost function. The cost function is the sum of all
    weights of the edges that are cut. After the cut, all the pixels connected to Source node
    become foreground and those connected to Sink node become background.
-   The process is continued until the classification converges.

It is illustrated in below image (Image Courtesy: <http://www.cs.ru.ac.za/research/g02m1682/>)

![image](images/grabcut_scheme.jpg)

Demo
----

We use the function: **cv.grabCut (image, mask, rect, bgdModel, fgdModel, iterCount, mode = cv.GC_EVAL)**

@param image      input 8-bit 3-channel image.
@param mask       input/output 8-bit single-channel mask. The mask is initialized by the function when mode is set to GC_INIT_WITH_RECT. Its elements may have one of the cv.rabCutClasses.
@param rect       ROI containing a segmented object. The pixels outside of the ROI are marked as "obvious background". The parameter is only used when mode==GC_INIT_WITH_RECT.
@param bgdModel   temporary array for the background model. Do not modify it while you are processing the same image.
@param fgdModel   temporary arrays for the foreground model. Do not modify it while you are processing the same image.
@param iterCount  number of iterations the algorithm should make before returning the result. Note that the result can be refined with further calls with mode==GC_INIT_WITH_MASK or mode==GC_EVAL .
@param mode       operation mode that could be one of the cv::GrabCutModes

Try it
------

\htmlonly
<iframe src="../../js_grabcut_grabCut.html" width="100%"
        onload="this.style.height=this.contentDocument.body.scrollHeight +'px';">
</iframe>
\endhtmlonly