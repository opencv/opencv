Customising and Debugging the detection system{#tutorial_mcc_debugging_the_system}
===========================

There are many hyperparameters that are involved in the detection of a chart.The default values are chosen to maximize the detections in the average case. But these might not be best for your use case.These values can be configured to improve the accuracy for a particular use case. To do this, you would need to create an
instance of `DetectorParameters`.

```
    mcc::Ptr<DetectorParameters> params = mcc::DetectorParameters::create();
```
* `mcc::` is important.

It contains a lot of values, the complete list can be found in the documentation for `DetectorParameters`. For this tutorial we will be playing with the value of `maxError`. The other values can be configured similarly.

`maxError` controls how much error is allowed in detection. Like if some chart cell is occluded. It will increase the error. The default value allows some level of tolerance to occlusions, increasing(or decreasing) `maxError`, will increase(or decrease) this tolerance.

You can change its value simply like this.

```
    params.maxError = 0.5;
```

To use this in the detection system, you would need to pass it to the process function.

```
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    detector->process(image, chartType, params = params);
```

Thats how easy is it to play with the values. But there is a catch, there are a lot of parts in the detection pipeline. If you simply run it like this you would not be able to see the effect of this change in isolation. It is possible that the preceding parts detected no possible colorchecker candidates, and so changing the value of `maxError` will have no effect. Luckily OpenCV provides a solution for this. You can make the code output a multiple images, each one showing the effect of one part of the pipeling. This is disabled by default.

* This can only be used if you are compiling from sources. If you can't build from souces, and still need this feature,try raising as issue in the OpenCV repo.

To do this : Open the file `opencv/modules/objdetect/include/opencv2/objdetect/mcc_checker_detector.hpp`, near the top there is this line

```
// #define MCC_DEBUG
```

Uncomment this line and rebuild opencv. After this whenever you run the detector, It will show you multiple images, each corresponding to a part of the pipeline. Also you might see some repetetions like first you will see `Thresholding Output`, then some more images, and again `Thresholding Output` corresponding to same image, but slightly different from previous one, it is because internally the image is thesholded multiple times, with different parameters to adjust for different possible sizes of the colorchecker.
