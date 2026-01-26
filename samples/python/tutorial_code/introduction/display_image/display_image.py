## [imports]
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
## [imports]


def imread_unicode(path: str, flags: int = cv.IMREAD_COLOR):
    """Read an image from paths that may include non-ASCII characters (Windows-safe)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size:
            img = cv.imdecode(data, flags)
            if img is not None:
                return img
    except Exception:
        pass
    return cv.imread(path, flags)


## [imread]
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # When running from OpenCV source checkout, the sample image exists here:
    samples_dir = Path(__file__).resolve().parents[4]  # .../samples
    candidate = samples_dir / "data" / "starry_night.jpg"

    if not candidate.exists():
        sys.exit(
            "Could not find sample image 'starry_night.jpg'.\n"
            "If you installed OpenCV via pip, sample data is usually not included.\n"
            "Run with your own image path, e.g.:\n"
            "  python display_image.py C:\\path\\to\\image.jpg"
        )

    image_path = str(candidate)

img = imread_unicode(image_path)
## [imread]


## [empty]
if img is None:
    sys.exit("Could not read the image.")
## [empty]


## [imshow]
cv.imshow("Display window", img)
k = cv.waitKey(0)
## [imshow]


## [imsave]
if k == ord("s"):
    cv.imwrite("starry_night.png", img)
## [imsave]
