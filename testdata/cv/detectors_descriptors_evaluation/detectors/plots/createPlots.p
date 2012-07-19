set terminal png

set key box outside bottom

set size 1, 0.7
set title "Detectors evaluation under scale changes (bark dataset)"
set xlabel "dataset image index (increasing zoom+rotation)"
set ylabel "repeatability, %"
set output 'bark_repeatability.png'
set xr[2:6]
set yr[0:100]
plot "FAST_bark_repeatability.csv" title 'FAST' with linespoints, "GFTT_bark_repeatability.csv" title 'GFTT' with linespoints, "HARRIS_bark_repeatability.csv" title 'HARRIS' with linespoints,  "MSER_bark_repeatability.csv" title 'MSER' with linespoints, "STAR_bark_repeatability.csv" title 'STAR' with linespoints, "SIFT_bark_repeatability.csv" title 'SIFT' with linespoints, "SURF_bark_repeatability.csv" title 'SURF' with linespoints

set size 1, 1
set ylabel "correspondences count"
set output 'bark_correspondenceCount.png'
set yr[0:2000]
plot "FAST_bark_correspondenceCount.csv" title 'FAST' with linespoints,  "GFTT_bark_correspondenceCount.csv" title 'GFTT' with linespoints, "HARRIS_bark_correspondenceCount.csv" title 'HARRIS' with linespoints,  "MSER_bark_correspondenceCount.csv" title 'MSER' with linespoints, "STAR_bark_correspondenceCount.csv" title 'STAR' with linespoints, "SIFT_bark_correspondenceCount.csv" title 'SIFT' with linespoints, "SURF_bark_correspondenceCount.csv" title 'SURF' with linespoints

set size 1, 0.7
set title "Detectors evaluation under blur changes (bike dataset)"
set xlabel "dataset image index (increasing blur)"
set ylabel "repeatability, %"
set output 'bikes_repeatability.png'
set xr[2:6]
set yr[0:100]
plot "FAST_bikes_repeatability.csv" title 'FAST' with linespoints, "GFTT_bikes_repeatability.csv" title 'GFTT' with linespoints, "HARRIS_bikes_repeatability.csv" title 'HARRIS' with linespoints,  "MSER_bikes_repeatability.csv" title 'MSER' with linespoints, "STAR_bikes_repeatability.csv" title 'STAR' with linespoints, "SIFT_bikes_repeatability.csv" title 'SIFT' with linespoints, "SURF_bikes_repeatability.csv" title 'SURF' with linespoints

set size 1, 1
set ylabel "correspondences count"
set output 'bikes_correspondenceCount.png'
set yr[0:1200]
plot "FAST_bikes_correspondenceCount.csv" title 'FAST' with linespoints,  "GFTT_bikes_correspondenceCount.csv" title 'GFTT' with linespoints, "HARRIS_bikes_correspondenceCount.csv" title 'HARRIS' with linespoints,  "MSER_bikes_correspondenceCount.csv" title 'MSER' with linespoints, "STAR_bikes_correspondenceCount.csv" title 'STAR' with linespoints, "SIFT_bikes_correspondenceCount.csv" title 'SIFT' with linespoints, "SURF_bikes_correspondenceCount.csv" title 'SURF' with linespoints

set size 1, 0.7
set title "Detectors evaluation under size changes (boat dataset)"
set xlabel "dataset image index (increasing zoom+rotation)"
set ylabel "repeatability, %"
set output 'boat_repeatability.png'
set xr[2:6]
set yr[0:100]
plot "FAST_boat_repeatability.csv" title 'FAST' with linespoints, "GFTT_boat_repeatability.csv" title 'GFTT' with linespoints, "HARRIS_boat_repeatability.csv" title 'HARRIS' with linespoints,  "MSER_boat_repeatability.csv" title 'MSER' with linespoints, "STAR_boat_repeatability.csv" title 'STAR' with linespoints, "SIFT_boat_repeatability.csv" title 'SIFT' with linespoints, "SURF_boat_repeatability.csv" title 'SURF' with linespoints

set size 1, 1
set ylabel "correspondences count"
set output 'boat_correspondenceCount.png'
set yr[0:3500]
plot "FAST_boat_correspondenceCount.csv" title 'FAST' with linespoints,  "GFTT_boat_correspondenceCount.csv" title 'GFTT' with linespoints, "HARRIS_boat_correspondenceCount.csv" title 'HARRIS' with linespoints,  "MSER_boat_correspondenceCount.csv" title 'MSER' with linespoints, "STAR_boat_correspondenceCount.csv" title 'STAR' with linespoints, "SIFT_boat_correspondenceCount.csv" title 'SIFT' with linespoints, "SURF_boat_correspondenceCount.csv" title 'SURF' with linespoints

set size 1, 0.7
set title "Detectors evaluation under viewpoint changes (graf dataset)"
set xlabel "viewpoint angle"
set ylabel "repeatability, %"
set output 'graf_repeatability.png'
set xr[20:60]
set yr[0:100]
plot "FAST_graf_repeatability.csv" title 'FAST' with linespoints, "GFTT_graf_repeatability.csv" title 'GFTT' with linespoints, "HARRIS_graf_repeatability.csv" title 'HARRIS' with linespoints,  "MSER_graf_repeatability.csv" title 'MSER' with linespoints, "STAR_graf_repeatability.csv" title 'STAR' with linespoints, "SIFT_graf_repeatability.csv" title 'SIFT' with linespoints, "SURF_graf_repeatability.csv" title 'SURF' with linespoints

set size 1, 1
set ylabel "correspondences count"
set output 'graf_correspondenceCount.png'
set yr[0:2000]
plot "FAST_graf_correspondenceCount.csv" title 'FAST' with linespoints,  "GFTT_graf_correspondenceCount.csv" title 'GFTT' with linespoints, "HARRIS_graf_correspondenceCount.csv" title 'HARRIS' with linespoints,  "MSER_graf_correspondenceCount.csv" title 'MSER' with linespoints, "STAR_graf_correspondenceCount.csv" title 'STAR' with linespoints, "SIFT_graf_correspondenceCount.csv" title 'SIFT' with linespoints, "SURF_graf_correspondenceCount.csv" title 'SURF' with linespoints

set size 1, 0.7
set title "Detectors evaluation under light changes (leuven dataset)"
set xlabel "dataset image index (decreasing light)"
set ylabel "repeatability, %"
set output 'leuven_repeatability.png'
set xr[2:6]
set yr[0:100]
plot "FAST_leuven_repeatability.csv" title 'FAST' with linespoints, "GFTT_leuven_repeatability.csv" title 'GFTT' with linespoints, "HARRIS_leuven_repeatability.csv" title 'HARRIS' with linespoints,  "MSER_leuven_repeatability.csv" title 'MSER' with linespoints, "STAR_leuven_repeatability.csv" title 'STAR' with linespoints, "SIFT_leuven_repeatability.csv" title 'SIFT' with linespoints, "SURF_leuven_repeatability.csv" title 'SURF' with linespoints

set size 1, 1
set ylabel "correspondences count"
set output 'leuven_correspondenceCount.png'
set yr[0:1500]
plot "FAST_leuven_correspondenceCount.csv" title 'FAST' with linespoints,  "GFTT_leuven_correspondenceCount.csv" title 'GFTT' with linespoints, "HARRIS_leuven_correspondenceCount.csv" title 'HARRIS' with linespoints,  "MSER_leuven_correspondenceCount.csv" title 'MSER' with linespoints, "STAR_leuven_correspondenceCount.csv" title 'STAR' with linespoints, "SIFT_leuven_correspondenceCount.csv" title 'SIFT' with linespoints, "SURF_leuven_correspondenceCount.csv" title 'SURF' with linespoints

set size 1, 0.7
set title "Detectors evaluation under blur changes (trees)"
set xlabel "dataset image index (increasing blur)"
set ylabel "repeatability, %"
set output 'trees_repeatability.png'
set xr[2:6]
set yr[0:100]
plot "FAST_trees_repeatability.csv" title 'FAST' with linespoints, "GFTT_trees_repeatability.csv" title 'GFTT' with linespoints, "HARRIS_trees_repeatability.csv" title 'HARRIS' with linespoints,  "MSER_trees_repeatability.csv" title 'MSER' with linespoints, "STAR_trees_repeatability.csv" title 'STAR' with linespoints, "SIFT_trees_repeatability.csv" title 'SIFT' with linespoints, "SURF_trees_repeatability.csv" title 'SURF' with linespoints

set size 1, 1
set ylabel "correspondences count"
set output 'trees_correspondenceCount.png'
set yr[0:6000]
plot "FAST_trees_correspondenceCount.csv" title 'FAST' with linespoints,  "GFTT_trees_correspondenceCount.csv" title 'GFTT' with linespoints, "HARRIS_trees_correspondenceCount.csv" title 'HARRIS' with linespoints,  "MSER_trees_correspondenceCount.csv" title 'MSER' with linespoints, "STAR_trees_correspondenceCount.csv" title 'STAR' with linespoints, "SIFT_trees_correspondenceCount.csv" title 'SIFT' with linespoints, "SURF_trees_correspondenceCount.csv" title 'SURF' with linespoints

set size 1, 0.7
set title "Detectors evaluation under JPEG compression (ubc dataset)"
set xlabel "JPEG compression, %"
set ylabel "repeatability, %"
set output 'ubc_repeatability.png'
set xr[60:98]
set yr[0:100]
plot "FAST_ubc_repeatability.csv" title 'FAST' with linespoints, "GFTT_ubc_repeatability.csv" title 'GFTT' with linespoints, "HARRIS_ubc_repeatability.csv" title 'HARRIS' with linespoints,  "MSER_ubc_repeatability.csv" title 'MSER' with linespoints, "STAR_ubc_repeatability.csv" title 'STAR' with linespoints, "SIFT_ubc_repeatability.csv" title 'SIFT' with linespoints, "SURF_ubc_repeatability.csv" title 'SURF' with linespoints

set size 1, 1
set ylabel "correspondences count"
set output 'ubc_correspondenceCount.png'
set yr[0:3000]
plot "FAST_ubc_correspondenceCount.csv" title 'FAST' with linespoints,  "GFTT_ubc_correspondenceCount.csv" title 'GFTT' with linespoints, "HARRIS_ubc_correspondenceCount.csv" title 'HARRIS' with linespoints,  "MSER_ubc_correspondenceCount.csv" title 'MSER' with linespoints, "STAR_ubc_correspondenceCount.csv" title 'STAR' with linespoints, "SIFT_ubc_correspondenceCount.csv" title 'SIFT' with linespoints, "SURF_ubc_correspondenceCount.csv" title 'SURF' with linespoints

set size 1, 0.7
set title "Detectors evaluation under viewpoint changes (wall dataset)"
set xlabel "viewpoint angle"
set ylabel "repeatability, %"
set output 'wall_repeatability.png'
set xr[20:60]
set yr[0:100]
plot "FAST_wall_repeatability.csv" title 'FAST' with linespoints, "GFTT_wall_repeatability.csv" title 'GFTT' with linespoints, "HARRIS_wall_repeatability.csv" title 'HARRIS' with linespoints,  "MSER_wall_repeatability.csv" title 'MSER' with linespoints, "STAR_wall_repeatability.csv" title 'STAR' with linespoints, "SIFT_wall_repeatability.csv" title 'SIFT' with linespoints, "SURF_wall_repeatability.csv" title 'SURF' with linespoints

set size 1, 1
set ylabel "correspondences count"
set output 'wall_correspondenceCount.png'
set yr[0:5000]
plot "FAST_wall_correspondenceCount.csv" title 'FAST' with linespoints,  "GFTT_wall_correspondenceCount.csv" title 'GFTT' with linespoints, "HARRIS_wall_correspondenceCount.csv" title 'HARRIS' with linespoints,  "MSER_wall_correspondenceCount.csv" title 'MSER' with linespoints, "STAR_wall_correspondenceCount.csv" title 'STAR' with linespoints, "SIFT_wall_correspondenceCount.csv" title 'SIFT' with linespoints, "SURF_wall_correspondenceCount.csv" title 'SURF' with linespoints

