Java Memory Management {#tutorial_java_mem_mgt}
======================

Starting with OpenCV 3.0.0 [Pull request 4014](https://github.com/Itseez/opencv/pull/4014) there is
a different memory management scheme that does not rely on Java's Object.finalize(). This guide will
demonstrate how to do proper memory management that's unique to the OpenCV Java bindings.

What we'll do in this guide
---------------------------

In this guide, we will:

-   Describe the old method of freeing native memory
-   Learn how to detect native memory leaks
-   Describe the new method of freeing native memory

The old way to free native memory
---------------------------------

OpenCV uses a Python code generator `modules/java/generator/gen_java.py` to generate the Java bindings
during the build process. The code to free memory looks like:

@code{.java}
@Override
protected void finalize() throws Throwable {
    delete(nativeObj);
}
@endcode

It is generally considered bad practice to use finalize since there no guarantee if or when it will be
called. When you build applications that process a lot of new Mats you will end up with a lot of
finalizers in the Finalizer queue and it eventually will get backed up. Another side effect that is
demonstrated by new Mat() is that it will consume large amounts of memory since n_delete is never called.
Instead there should be some type of public clean up method that deallocates native memory and not
rely on finalize(). For more reading on the subject see:

-   [Debugging to understand Finalizers](https://plumbr.eu/blog/garbage-collection/debugging-to-understand-finalizer)
-   [Why would you ever implement finalize()?](http://stackoverflow.com/questions/158174/why-would-you-ever-implement-finalize)

How to detect native memory leaks
---------------------------------

Since the OpenCV Java bindings wrap OpenCV's C++ libraries there are opportunities for native memory to
leak without being able to detect it from Java ([jmap](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/jmap.html)/[jhat](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/jhat.html)).
Some of the bindings create new local Mat objects and subclasses of Mat without calling Mat.release().
This will cause native memory leaks! These are the steps required to analyze a Java program using
OpenCV (or any JNI based app). For this example we are going to use Ubuntu.

### Install OpenCV with Java bindings

I have a simple install script aptly named [Install OpenCV](https://github.com/sgjava/install-opencv)
that does all the dirty work. You could also build it by hand as long as you generate the Java bindings
successfully.

### Install Valgrind and the Valkyrie GUI

@code{.bash}
sudo apt-get install valgrind valkyrie
@endcode

### Profile application

For this example we'll assume you used my script to install OpenCV. If not, just adjust the paths and
class you wish to execute.

@code{.bash}
cd /home/<username>/workspace/install-opencv/opencv-java
@endcode

@code{.bash}
valgrind --trace-children=yes --leak-check=full --num-callers=15 --xml=yes --xml-file=/home/<username>/canny.xml java -Djava.compiler=NONE -Djava.library.path=/home/<username>/opencv-3.0.0/build/lib -cp /home/<username>/opencv-3.0.0/build/bin/opencv-300.jar:bin com.codeferm.opencv.Canny
@endcode

### Examine Valgrind output

-   `valkyrie`
-   Open canny.xml (or whatever file name you used)
-   Scroll down to bottom
-   Look for OpenCV classes which are wrapped by Java such as `0x1FDD0BFE: Java_org_opencv_imgproc_Imgproc_findContours_11 (in /home/<username>/opencv-3.0.x/build/lib/libopencv_java30x.so)`
    This will give you a hint which Java class is leaking memory. There's always a chance it could be
    a memory leak in the C++ code which would require fixing the C++ source.

In the above example I found a memory leak in Imgproc.findContours. It creates a local Mat and
does not call Mat.release(). My install script patches this class after the build process
completes and rebuilds the Java bindings. This patched code looks like:

@code{.java}
@Override
public static void findContours(Mat image, List<MatOfPoint> contours, Mat hierarchy, int mode, int method)
{
    Mat contours_mat = new Mat();
    findContours_1(image.nativeObj, contours_mat.nativeObj, hierarchy.nativeObj, mode, method);
    Converters.Mat_to_vector_vector_Point(contours_mat, contours);
    contours_mat.release();
    return;
}
@endcode

The new way to free native memory
---------------------------------

The changes I made to `modules/java/generator/gen_java.py` will now generate a public delete() method to
replace finalize() as follows:

@code{.java}
public void delete() {
    delete(nativeObj);
}
@endcode

I also added n_delete() to Mat.release() and removed the finalize() method to prevent native memory leaks
in Mat and all Mat subclasses.

@code{.java}
public void release()
{

    n_release(nativeObj);
    n_delete(nativeObj);

    return;
}
@endcode

In essence, this means you have to explicitly manage memory (not normal in Java) instead of relying on
implicit finalize() method which does not work. You can see an example of this:
[Memory Leak from iterating Opencv frames](http://stackoverflow.com/questions/21050499/memory-leak-from-iterating-opencv-frames)
The way to explicitly free native memory is to call Mat.release() and for other classes call delete().
To find out which classes need delete() called go to the OpenCV build dir `opencv/build/src` and:

@code{.bash}
cd /home/<username>/opencv/build/src
grep -R -i "delete()" .
@endcode

Which will display:
```
./org/opencv/video/KalmanFilter.java:    public void delete() {
./org/opencv/video/BackgroundSubtractorMOG2.java:    public void delete() {
./org/opencv/video/BackgroundSubtractor.java:    public void delete() {
./org/opencv/video/DenseOpticalFlow.java:    public void delete() {
./org/opencv/video/BackgroundSubtractorKNN.java:    public void delete() {
./org/opencv/video/DualTVL1OpticalFlow.java:    public void delete() {
./org/opencv/core/Algorithm.java:    public void delete() {
./org/opencv/calib3d/StereoMatcher.java:    public void delete() {
./org/opencv/calib3d/StereoBM.java:    public void delete() {
./org/opencv/calib3d/StereoSGBM.java:    public void delete() {
./org/opencv/features2d/FeatureDetector.java:    public void delete() {
./org/opencv/features2d/DescriptorMatcher.java:    public void delete() {
./org/opencv/features2d/DescriptorExtractor.java:    public void delete() {
./org/opencv/videoio/VideoCapture.java:    public void delete() {
./org/opencv/videoio/VideoWriter.java:    public void delete() {
./org/opencv/ml/StatModel.java:    public void delete() {
./org/opencv/ml/LogisticRegression.java:    public void delete() {
./org/opencv/ml/Boost.java:    public void delete() {
./org/opencv/ml/ANN_MLP.java:    public void delete() {
./org/opencv/ml/RTrees.java:    public void delete() {
./org/opencv/ml/DTrees.java:    public void delete() {
./org/opencv/ml/TrainData.java:    public void delete() {
./org/opencv/ml/SVM.java:    public void delete() {
./org/opencv/ml/KNearest.java:    public void delete() {
./org/opencv/ml/EM.java:    public void delete() {
./org/opencv/ml/NormalBayesClassifier.java:    public void delete() {
./org/opencv/imgproc/Subdiv2D.java:    public void delete() {
./org/opencv/imgproc/CLAHE.java:    public void delete() {
./org/opencv/imgproc/LineSegmentDetector.java:    public void delete() {
./org/opencv/objdetect/CascadeClassifier.java:    public void delete() {
./org/opencv/objdetect/HOGDescriptor.java:    public void delete() {
./org/opencv/objdetect/BaseCascadeClassifier.java:    public void delete() {
./org/opencv/photo/TonemapDrago.java:    public void delete() {
./org/opencv/photo/CalibrateRobertson.java:    public void delete() {
./org/opencv/photo/CalibrateCRF.java:    public void delete() {
./org/opencv/photo/CalibrateDebevec.java:    public void delete() {
./org/opencv/photo/Tonemap.java:    public void delete() {
./org/opencv/photo/AlignMTB.java:    public void delete() {
./org/opencv/photo/MergeDebevec.java:    public void delete() {
./org/opencv/photo/MergeExposures.java:    public void delete() {
./org/opencv/photo/MergeMertens.java:    public void delete() {
./org/opencv/photo/MergeRobertson.java:    public void delete() {
./org/opencv/photo/TonemapReinhard.java:    public void delete() {
./org/opencv/photo/TonemapDurand.java:    public void delete() {
./org/opencv/photo/TonemapMantiuk.java:    public void delete() {
./org/opencv/photo/AlignExposures.java:    public void delete() {
```

If in doubt just use your IDE's code completion and try Object.delete(). For Mat and subclasses
simply call Mat.release(). To wrap things up let's review:

-   Do not use Object.finalize()
-   Use Valgrind and Valkyrie to look for native memory leaks
-   Always call Mat.release() and delete() where appropriate
