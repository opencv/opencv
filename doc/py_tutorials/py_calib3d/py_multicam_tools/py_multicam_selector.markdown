Multi-camera Calibration Image Selection {#tutorial_py_multicam_image_selector}
===============================================================================

Goal
----
This tool helps you pick good calibration images automatically.
It works with multiple cameras or a single video.

- Scores frames for **sharpness**, **exposure**, and **pattern coverage**
- Keeps a small, diverse set that makes calibration more stable

@note The script only **selects images**. It does not calibrate.
It writes YAML lists that you can feed into calibration later.

> **Intended use:** This tool is designed to pre-filter datasets for later calibration with
> `samples/cpp/calibration.cpp` or `samples/python/multiview_calibration.py`.

**Source:** `samples/python/multicam_image_selector.py`

Basics
------
Calibration works best when the board is sharp, well lit, and appears at varied positions and scales.
Picking those by hand is slow. This script scans all frames, scores them, and keeps the best ones.

See also:
- [Camera Calibration](#tutorial_py_calibration)
- [Camera Calibration and 3D Reconstruction](#tutorial_table_of_content_calib3d)

What the script does
--------------------
1. **Read inputs**
   - A root folder with subfolders per camera (`cam1/`, `cam2/`, …)
   - Or a video with `--video` (frames sampled by `--video-step`)

2. **Detect the pattern**
   - Supported: `chessboard`, `circles`, `acircles`, `charuco`, `aruco_grid`
   - Uses standard OpenCV calls: `cv.findChessboardCorners()`, `cv.cornerSubPix()`, `cv.findCirclesGrid()`, ArUco

3. **Score each frame**
   - **Sharpness** – Laplacian variance
   - **Exposure** – checks for under/overexposure
   - **Coverage** – number and spread of detected points
   - Final score = weighted sum of these

4. **Pick the best**
   - Up to `--per-camera` images per camera (`80` by default)
   - Methods:
     - `kmeans` (default): cluster for variety
     - `greedy`: pick diverse, high-score frames
     - `random`: draw from the top set
   - Sync options:
     - `--require-all-cams`: keep only frames present in all cameras
     - `--strict-all-cams`: error if no common frames
     - `--pairwise`: maximize coverage across camera pairs

5. **Write results**
   - YAML per camera with image paths
   - Master YAML listing all cameras
   - Optional CSV with metrics
   - Optional plots (scatter + histogram)

Frame IDs
---------
For multi-camera data, frames are aligned by **numeric IDs** in filenames or folders.
Example:
`cam1/frame_00123.jpg` and `cam2/frame_00123.jpg` are treated as the same moment.

Command-line options
--------------------

### Input
@code{.text}
--root <path>        Folder with per-camera subfolders
--video <path>       Read frames from one video
--video-step <int>   Take every Nth frame from video (default: 10)
@endcode

### Pattern
@code{.text}
--pattern <type>     chessboard | circles | acircles | charuco | aruco_grid
--rows <int>         Pattern rows (inner corners or dots)
--cols <int>         Pattern cols (inner corners or dots)
--aruco-dict <name>  ArUco dictionary (default: DICT_5X5_1000)
--square <float>     Square size [m] (Charuco/GridBoard)
--marker <float>     Marker size [m] (Charuco/GridBoard)
--separation <float> Marker separation [m] (GridBoard)
--expected-aruco-markers <int>  Expected markers (0 = auto)
@endcode

### Selection
@code{.text}
--per-camera <int>   Images to keep per camera (default: 80)
--selector <method>  kmeans | greedy | random
--require-all-cams   Keep only frames shared by all cameras
--strict-all-cams    Error if no common frames
--pairwise           Maximize camera-pair coverage
--seed <int>         Random seed (default: 123)
@endcode

### Processing
@code{.text}
--max-size <int>     Resize before scoring (0 = full res)
--min-sharpness <f>  Minimum sharpness (default: 0.0)
--min-corners <int>  Minimum corners detected (default: 0)
--jobs <int>         Worker threads (0 = auto)
@endcode

### Output
@code{.text}
--dump-metrics <path>  Save per-image metrics CSV
--viz-out <path>       Save scatter and histogram plots
--cache-file <path>    Persistent cache (default: .selector_cache.pkl)
--resume               Reuse cache if options match
--out <path>           Output directory for YAMLs (required)
@endcode

Examples
--------
### Chessboard (multi-camera, strict sync + plots/CSV)
@code{.bash}
python3 multicam_image_selector.py \
  --root calibration_images \
  --out calibration_output \
  --pattern chessboard \
  --rows 7 --cols 10 \
  --per-camera 15   # default is 80
  --selector kmeans \
  --max-size 1600 \
  --require-all-cams \
  --strict-all-cams \
  --dump-metrics calibration_output/metrics.csv \
  --viz-out calibration_output/viz
@endcode

### ChArUco from a video
@code{.bash}
python3 multicam_image_selector.py \
  --video session.mp4 \
  --out charuco_output \
  --pattern charuco \
  --rows 7 --cols 5 \
  --per-camera 60 \
  --video-step 10
@endcode

### Pairwise coverage
@code{.bash}
python3 multicam_image_selector.py \
  --root rig_data \
  --out paired_output \
  --pattern circles \
  --rows 4 --cols 11 \
  --per-camera 40 \
  --selector greedy \
  --pairwise
@endcode

### Example output (selection summary)
@code{.bash}
Selection summary:
  cam1: 15 images selected
  cam2: 15 images selected
  cam3: 15 images selected
  cam4: 15 images selected
  cam5: 15 images selected
  cam6: 15 images selected
  cam7: 15 images selected

Per-camera YAMLs: /Users/.../chessboard_yaml
Master YAML:     /Users/.../chessboard_yaml/master.yaml
Total time: 1.58s  |  Images: 210  |  Jobs: 6
@endcode

Outputs
-------
### Per-camera YAML
Each camera gets its own YAML.

Example: [cam1.yaml](assets/multicam_selector/cam1.yaml)

@code{.yaml}
%YAML:1.0
image_list:
- /abs/path/calibration_images/cam1/frame_0005.jpg
- /abs/path/calibration_images/cam1/frame_0007.jpg
- /abs/path/calibration_images/cam1/frame_0008.jpg
- /abs/path/calibration_images/cam1/frame_0001.jpg
- /abs/path/calibration_images/cam1/frame_0006.jpg
@endcode

### Master YAML
Collects all cameras into one file.

[master.yaml](assets/multicam_selector/master.yaml)

@code{.yaml}
%YAML:1.0
cameras:
- name: cam1
  yaml: /abs/path/calibration_output/cam1.yaml
- name: cam2
  yaml: /abs/path/calibration_output/cam2.yaml
@endcode

### Metrics CSV
Stores per-frame metrics.

[metrics.csv](assets/multicam_selector/metrics.csv)

@code{.text}
image_path,sharpness,exposure_ok,n_corners,center_x,center_y,log_scale,score
/path/cam1/frame_0005.jpg,0.0021,0.98,70,0.48,0.52,-0.21,0.89
@endcode

Columns:
- **sharpness** – higher means sharper image
- **exposure_ok** – closer to 1 means well exposed
- **n_corners** – number of detected points
- **center_x, center_y** – normalized board center (0–1)
- **log_scale** – relative board size in the frame
- **score** – final combined score

Plots
-----
Saved when `--viz-out` is set.

- **Centers scatterplot**
  All frames are shown as faint points, selected frames as filled points.
  This shows where the board was detected; a wide spread means better coverage.

  ![](assets/multicam_selector/cam1_centers.png)

- **Score histogram**
  Shows the distribution of selection scores.
  All frames appear as background bars, selected frames are highlighted.

  ![](assets/multicam_selector/cam1_score_hist.png)

---

Summary
-------
`multicam_image_selector.py` helps filter calibration data:

- Scores frames for sharpness, exposure, and coverage
- Selects a diverse subset per camera
- Produces per-camera YAMLs, a master YAML, a metrics CSV, and plots
- Prints a **selection summary** with number of images per camera

These outputs plug directly into calibration and improve accuracy.

---

Troubleshooting
---------------
- **No images selected** → lower `--min-sharpness` or `--min-corners`
- **Error with `--strict-all-cams`** → make sure filenames share numeric IDs
- **Plots not generated** → check that Matplotlib is installed

---

Tips
----
- Capture varied board positions and scales
- If many frames score low, check lighting or motion blur
- Use `--require-all-cams` only if filenames share consistent IDs
- Use `--max-size` for faster scoring on large images
- With `--resume` and a cache file, reruns are reproducible

See also
--------
- [Camera Calibration](#tutorial_py_calibration)
- [3-D Calibration Visualisation](#tutorial_py_multicam_vis) – inspect selected results in 3-D
