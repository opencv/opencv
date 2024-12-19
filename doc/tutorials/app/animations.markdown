Working with Animated Files {#tutorial_animations}
===========================

@tableofcontents

|    |    |
| -: | :- |
| Original author | Suleyman Turkmen (with help of ChatGPT) |
| Compatibility | OpenCV >= 4.11 |

Goal
----
In this tutorial, you will learn how to:

- Use cv::imreadanimation to load frames from animated files.
- Understand the structure and parameters of the cv::Animation structure.
- Display individual frames from an animation.
- Use cv::imwriteanimation to write cv::Animation to file.

Source Code
-----------

@add_toggle_cpp
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/imgcodecs/animations.cpp)

-   **Code at glance:**
    @include samples/cpp/tutorial_code/imgcodecs/animations.cpp
@end_toggle

@add_toggle_python
-   **Downloadable code**: Click
    [here](https://github.com/opencv/opencv/tree/4.x/samples/python/tutorial_code/imgcodecs/animations.py)

-   **Code at glance:**
    @include samples/python/tutorial_code/imgcodecs/animations.py
@end_toggle

### Function Overview:

The cv::imreadanimation function is used to load multiple frames from animated files (such as AVIF, APNG, or WebP) into an cv::Animation structure.
This function is especially useful for handling animated images, where each frame is a separate image that can be processed individually.

#### Function Signature

```cpp
CV_EXPORTS_W bool imreadanimation(const String& filename, CV_OUT Animation& animation, int start = 0, int count = INT16_MAX);
```

### Parameters

- filename: The path to the animated file, which must be in a supported animated format (e.g., AVIF, APNG, WebP).
- animation: A reference to an cv::Animation structure where the frames will be stored.
- start: The index of the first frame to load (optional, defaults to 0).
- count: The number of frames to load (optional, defaults to 32767).

Explanation
-----------

1. **Setting up the Animation Structure**

   Initialize an cv::Animation structure to hold the frames from the animated file.

   ```cpp
   Animation animation;
   ```

2. **Loading Frames**

   Use cv::imreadanimation to load frames from the specified file. Here, we load all frames from an animated WebP image.

@add_toggle_cpp
@snippet cpp/tutorial_code/imgcodecs/animations.cpp read_animation
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgcodecs/animations.py read_animation
@end_toggle

3. **Displaying Frames**

   Each frame in the `animation.frames` vector can be displayed as a standalone image. This loop iterates through each frame, displaying it in a window with a short delay to simulate the animation.

@add_toggle_cpp
@snippet cpp/tutorial_code/imgcodecs/animations.cpp show_animation
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgcodecs/animations.py show_animation
@end_toggle

4. **Saving Animation**

@add_toggle_cpp
@snippet cpp/tutorial_code/imgcodecs/animations.cpp write_animation
@end_toggle

@add_toggle_python
@snippet python/tutorial_code/imgcodecs/animations.py write_animation
@end_toggle

## Summary

The cv::imreadanimation and cv::imwriteanimation functions make it easy to work with animated images by loading frames into an cv::Animation structure, allowing frame-by-frame processing.
With these functions, you can load, process, and save frames from animated formats like AVIF, APNG, and WebP.
