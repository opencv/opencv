#CVVisual Example
CVVisual is a debug visualization for OpenCV; thus, its main purpose is to offer different ways to visualize 
the results of OpenCV functions to make it possible to see whether they are what the programmer had in mind;
and also to offer some functionality to try other operations on the images right in the debug window.
This text wants to illustrate the use of CVVisual on a code example.

Image we want to debug this program:

[code_example/main.cpp](https://github.com/CVVisualPSETeam/CVVisual/tree/master/doc/code_example/main.cpp)

Note the includes for CVVisual:

	10 #include <opencv2/debug_mode.hpp>
	11 #include <opencv2/show_image.hpp>
	12 #include <opencv2/filter.hpp>
	13#include <opencv2/dmatch.hpp>
	14 #include <opencv2/final_show.hpp>

It takes 10 snapshots with the webcam.
With each, it first shows the image alone in the debug window,

	97 cvv::showImage(imgRead, CVVISUAL_LOCATION, imgIdString.c_str());

then converts it to grayscale and calls CVVisual with the original and resulting image, 

	101 cv::cvtColor(imgRead, imgGray, CV_BGR2GRAY);
	102	cvv::debugFilter(imgRead, imgGray, CVVISUAL_LOCATION, "to gray");

detects the grayscale image's ORB features

	107 detector(imgGray, cv::noArray(), keypoints, descriptors);

and matches them to those of the previous image, if available. It calls cvv::debugDMatch() with the results.

	113 matcher.match(prevDescriptors, descriptors, matches);
	...
	117 cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, allMatchIdString.c_str());

Finally, it removes the worst (as defined by match distance) 0.8 quantile of matches and calls cvv::debugDMatch() again.

	121 std::sort(matches.begin(), matches.end());
	122 matches.resize(int(bestRatio * matches.size()));
	...
	126 cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, bestMatchIdString.c_str());

After we started the program, the CVVisual Main Window opens with one _Call_, that is, the first image that a `cvv::showImage()` was called with (the program execution was halted at this call).

![](../images_example/overview_single_call.png)

The image is shown as a small thumbnail in the _Overview table_, together with additional information on it, like the line of the call and the description passed as a parameter.
We double-click it, and a tab opens, where the image is shown bigger. It looks like the webcam worked, so we press `Step` and go to the _Overview_.

![](../images_example/single_image_tab.png)

The window shows up again, this time with the first _Call_ to `cvv::debugFilter()` added.

![](../images_example/overview_two_calls.png)

We open its tab, too, because, say, the grayscale image does not exactly look like what we wanted.

![](../images_example/filter_tab_default.png)

After switching to _SingleFilterView_, which will be more useful to us here, we select to not show the right two images - the grayscale image and the one below, where results of filter operations in this tab are depicted.

![](../images_example/single_filter_right_two_imgs_unselected.png)

In `Select a filter`, a gray filter can be applied with different parameters.

![](../images_example/single_filter_gray.png)

This looks more like what we wanted. 
Rechecking `Show image` for the unselected result image of the actual filter operation and zooming (`Ctrl` + `Mouse wheel`) into all images synchronously deeper than 60% shows the different values of the pixels.

![](../images_example/single_filter_deep_zoom.png)

Sadly, we can't do anything about this situation in this session, though, so we just continue.
As stepping through each single _Call_ seems quite tedious, we use the _fast-forward_ button, `>>`.
The program runs until it reaches `finalShow()`, taking images with the webcam along the way.
This saved us some clicking; on the downside, we now have quite an amount of _Calls_ in the table.

![](../images_example/overview_all.png)

Using the [filter query language](http://cvv.mostlynerdless.de/ref/filters-ref.html), the _Calls_ to `debugDMatch()` can be filtered out as they have the specific type "match".

![](../images_example/overview_matches_filtered.png)

We open the tab of the last such _Call_, and find ourselves greeted with a dense bundle of lines across both images, which represent the matches between the two.

![](../images_example/match_tab_line.png)

It is a bit unclear where there actually are matches in this case, so we switch to _TranslationMatchView_, which is a little bit better (especially after scrolling a bit to the right in the left image).

![](../images_example/match_translations.png)

_TranslationMatchView_ shows how the matching _KeyPoints_ are moved in the respective other image.
It seems more fitting for this debug session than the _LineMatchView_, thus,  we `Set`it `as default`.
Still, there are too many matches for our taste.
Back in the _Overview_, we open the _Call_ before the last, which is the one where the upper 80% of matches were not yet filtered out.

![](../images_example/match_tab_translations_2.png)

Here, the best 70% of matches can be chosen. The result looks more acceptable, and we take a mental note to change the threshold to 0.7.

![](../images_example/match_translations_2_70percent.png)

The matches can also be shown in a table, the so called _RawView_:

![](../images_example/raw_view.png)

Here, you could copy a selection of them as CSV, JSON, Ruby or Python to the clipboard.
We don't need that in the moment, though; we just close the window, and the program finishes.
We now know what we might want to change in the program.


Finally, a little note on the `cvv::finalShow()` function:

It needs to be there in every program using CVVisual, after the last call to another CVVisual function, er else, the program will crash in the end.

Hopefully, this example shed some light on how CVVisual can be used.
If you want to learn more, refer to the [API](http://cvv.mostlynerdless.de/api) or other documentation on the [web page](http://cvv.mostlynerdless.de/).

Credit, and special thanks, goes to Andreas Bihlmaier, supervisor of the project, who provided the example code.
