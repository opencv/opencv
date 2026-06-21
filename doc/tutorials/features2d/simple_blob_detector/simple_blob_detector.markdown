SimpleBlobDetector Tutorial {#tutorial_simple_blob_detector}
===========================

@tableofcontents

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Understand what a **blob** is in the context of image processing.
-   Use @ref cv::SimpleBlobDetector to detect blobs in an image.
-   Control detection using the available filter parameters.

What is a Blob?
---------------

A **blob** is a region of an image where pixels share a similar property — such as colour,
brightness, or size. In simple terms, a blob is a connected group of pixels that stands out from
its surroundings.

```
Original grayscale image          Detected blobs (marked with circles)

  . . . . . . . . . . .             . . . . . . . . . . .
  . . . ██████ . . . .              . . . ╔══════╗ . . .
  . . . ██████ . . . .              . . . ║  ●   ║ . . .
  . . . ██████ . . . .              . . . ╚══════╝ . . .
  . . . . . . . . . . .             . . . . . . . . . . .
  . . . . . ████ . . .              . . . . . ╔════╗ . .
  . . . . . ████ . . .              . . . . . ║ ●  ║ . .
  . . . . . . . . . . .             . . . . . ╚════╝ . .

  (dark blobs on light background)  (each blob circled and centred)
```

`SimpleBlobDetector` finds these regions, filters them by the properties you specify, and returns
their centre coordinates and approximate sizes as @ref cv::KeyPoint objects.

How SimpleBlobDetector Works
-----------------------------

The detector runs the following pipeline internally for each image:

```
┌───────────────────────────────────────┐
│            Input Image                │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│  Step 1: Threshold                    │
│  Convert to binary using multiple     │
│  thresholds between minThreshold and  │
│  maxThreshold (step = thresholdStep)  │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│  Step 2: Group Connected Components   │
│  Find connected white (or black)      │
│  regions in each binary image         │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│  Step 3: Merge Centres                │
│  Group centres across threshold       │
│  levels that are close together       │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│  Step 4: Filter                       │
│  Apply enabled filters (see below)    │
│  — reject blobs that do not pass      │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│  Output: vector<KeyPoint>             │
│  centre (x, y) + size per blob        │
└───────────────────────────────────────┘
```

Filter Parameters
-----------------

`SimpleBlobDetector::Params` controls which blobs are accepted. Every filter can be switched on or
off independently with its `filterBy*` flag.

### Area

Controls the acceptable blob size in pixels.

| Parameter       | Default | Meaning |
| :-------------- | :------ | :------ |
| `filterByArea`  | `true`  | Enable area filtering |
| `minArea`       | `25`    | Minimum blob area (px²) |
| `maxArea`       | `5000`  | Maximum blob area (px²) |

```
Too small         Accepted range          Too large
(rejected)        ✓ minArea … maxArea ✓   (rejected)

  ·                  ██                  ████████████
                     ██                  ████████████
                                         ████████████
```

### Circularity

Measures how close a blob's shape is to a perfect circle.
`circularity = 4π × area / perimeter²` — a perfect circle scores **1.0**.

| Parameter            | Default | Meaning |
| :------------------- | :------ | :------ |
| `filterByCircularity`| `false` | Enable circularity filtering |
| `minCircularity`     | `0.8`   | Minimum score |
| `maxCircularity`     | `1e37`  | Maximum score (no upper limit) |

```
Circularity ≈ 0.1        Circularity ≈ 0.5       Circularity ≈ 1.0
(thin / elongated)       (roughly square)         (perfect circle)

  ████                      ████                      ●
  █                         ████
  █                         ████
  █
```

### Convexity

Ratio of blob area to its convex hull area. A convex shape scores **1.0**; a star or U-shape
scores lower.

| Parameter          | Default | Meaning |
| :----------------- | :------ | :------ |
| `filterByConvexity`| `true`  | Enable convexity filtering |
| `minConvexity`     | `0.95`  | Minimum convexity |
| `maxConvexity`     | `1e37`  | Maximum convexity |

### Inertia Ratio

Describes how elongated a blob is. A circle has ratio **1.0**; a line has ratio **0.0**.

| Parameter            | Default | Meaning |
| :------------------- | :------ | :------ |
| `filterByInertia`    | `true`  | Enable inertia filtering |
| `minInertiaRatio`    | `0.1`   | Minimum ratio |
| `maxInertiaRatio`    | `1e37`  | Maximum ratio |

### Colour

Matches blobs by the pixel intensity at their centre.

| Parameter       | Default | Meaning |
| :-------------- | :------ | :------ |
| `filterByColor` | `true`  | Enable colour filtering |
| `blobColor`     | `0`     | `0` = dark blobs, `255` = light blobs |

Code
----

@add_toggle_cpp
```cpp
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat image = cv::imread("blobs.jpg", cv::IMREAD_GRAYSCALE);

    // Configure parameters
    cv::SimpleBlobDetector::Params params;
    params.filterByArea    = true;
    params.minArea         = 100;
    params.maxArea         = 5000;
    params.filterByCircularity = true;
    params.minCircularity  = 0.8f;
    params.filterByColor   = true;
    params.blobColor       = 0;   // detect dark blobs

    cv::Ptr<cv::SimpleBlobDetector> detector =
        cv::SimpleBlobDetector::create(params);

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(image, keypoints);

    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output,
                      cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("Blobs", output);
    cv::waitKey(0);
    return 0;
}
```
@end_toggle

@add_toggle_python
```python
import cv2

image = cv2.imread("blobs.jpg", cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
params.maxArea = 5000
params.filterByCircularity = True
params.minCircularity = 0.8
params.filterByColor = True
params.blobColor = 0  # detect dark blobs

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(image)

output = cv2.drawKeypoints(
    image, keypoints, None,
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
cv2.imshow("Blobs", output)
cv2.waitKey(0)
```
@end_toggle

Output
------

Each detected blob is returned as a @ref cv::KeyPoint:

| Field        | Meaning                        |
| :----------- | :----------------------------- |
| `pt.x/pt.y`  | Centre coordinates of the blob |
| `size`       | Diameter of the blob           |
| `response`   | Confidence score               |

Use @ref cv::drawKeypoints with the `DRAW_RICH_KEYPOINTS` flag to visualise both the centre and
the approximate size of each detected blob as a circle.

Common Pitfalls
---------------

-   **No blobs detected** — check `blobColor`. If your blobs are bright on a dark background,
    set `blobColor = 255`.
-   **Too many false positives** — tighten `minCircularity` or reduce `maxArea`.
-   **Large blobs missed** — increase `maxArea` and `maxThreshold`.
