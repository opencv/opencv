Reading Geospatial Raster files with GDAL {#tutorial_raster_io_gdal}
=========================================

@tableofcontents

@prev_tutorial{tutorial_trackbar}
@next_tutorial{tutorial_video_input_psnr_ssim}

|    |    |
| -: | :- |
| Original author | Marvin Smith |
| Compatibility | OpenCV >= 3.0 |

Geospatial raster data is a heavily used product in Geographic Information Systems and
Photogrammetry. Raster data typically can represent imagery and Digital Elevation Models (DEM). The
standard library for loading GIS imagery is the Geographic Data Abstraction Library [(GDAL)](http://www.gdal.org). In this
example, we will show techniques for loading GIS raster formats using native OpenCV functions. In
addition, we will show some an example of how OpenCV can use this data for novel and interesting
purposes.

Goals
-----

The primary objectives for this tutorial:

-   How to use OpenCV [imread](@ref imread) to load satellite imagery.
-   How to use OpenCV [imread](@ref imread) to load SRTM Digital Elevation Models
-   Given the corner coordinates of both the image and DEM, correlate the elevation data to the
    image to find elevations for each pixel.
-   Show a basic, easy-to-implement example of a terrain heat map.
-   Show a basic use of DEM data coupled with ortho-rectified imagery.

To implement these goals, the following code takes a Digital Elevation Model as well as a GeoTiff
image of San Francisco as input. The image and DEM data is processed and generates a terrain heat
map of the image as well as labels areas of the city which would be affected should the water level
of the bay rise 10, 50, and 100 meters.

Code
----

@include cpp/tutorial_code/imgcodecs/GDAL_IO/gdal-image.cpp

How to Read Raster Data using GDAL
----------------------------------

This demonstration uses the default OpenCV imread function. The primary difference is that in order
to force GDAL to load the image, you must use the appropriate flag.
@snippet cpp/tutorial_code/imgcodecs/GDAL_IO/gdal-image.cpp load1
When loading digital elevation models, the actual numeric value of each pixel is essential and
cannot be scaled or truncated. For example, with image data a pixel represented as a double with a
value of 1 has an equal appearance to a pixel which is represented as an unsigned character with a
value of 255. With terrain data, the pixel value represents the elevation in meters. In order to
ensure that OpenCV preserves the native value, use the GDAL flag in imread with the ANYDEPTH flag.
@snippet cpp/tutorial_code/imgcodecs/GDAL_IO/gdal-image.cpp load2
If you know beforehand the type of DEM model you are loading, then it may be a safe bet to test the
Mat::type() or Mat::depth() using an assert or other mechanism. NASA or DOD specification documents
can provide the input types for various elevation models. The major types, SRTM and DTED, are both
signed shorts.

Notes
-----

### Lat/Lon (Geographic) Coordinates should normally be avoided

The Geographic Coordinate System is a spherical coordinate system, meaning that using them with
Cartesian mathematics is technically incorrect. This demo uses them to increase the readability and
is accurate enough to make the point. A better coordinate system would be Universal Transverse
Mercator.

### Finding the corner coordinates

One easy method to find the corner coordinates of an image is to use the command-line tool gdalinfo.
For imagery which is ortho-rectified and contains the projection information, you can use the [USGS
EarthExplorer](http://http://earthexplorer.usgs.gov).
@code{.bash}
\f$> gdalinfo N37W123.hgt

   Driver: SRTMHGT/SRTMHGT File Format
   Files: N37W123.hgt
   Size is 3601, 3601
   Coordinate System is:
   GEOGCS["WGS 84",
   DATUM["WGS_1984",

   ... more output ...

   Corner Coordinates:
   Upper Left  (-123.0001389,  38.0001389) (123d 0' 0.50"W, 38d 0' 0.50"N)
   Lower Left  (-123.0001389,  36.9998611) (123d 0' 0.50"W, 36d59'59.50"N)
   Upper Right (-121.9998611,  38.0001389) (121d59'59.50"W, 38d 0' 0.50"N)
   Lower Right (-121.9998611,  36.9998611) (121d59'59.50"W, 36d59'59.50"N)
   Center      (-122.5000000,  37.5000000) (122d30' 0.00"W, 37d30' 0.00"N)

    ... more output ...
@endcode
Results
-------

Below is the output of the program. Use the first image as the input. For the DEM model, download
the SRTM file located at the USGS here.
[<http://dds.cr.usgs.gov/srtm/version2_1/SRTM1/Region_04/N37W123.hgt.zip>](http://dds.cr.usgs.gov/srtm/version2_1/SRTM1/Region_04/N37W123.hgt.zip)

![Input Image](images/gdal_output.jpg)

![Heat Map](images/gdal_heat-map.jpg)

![Heat Map Overlay](images/gdal_flood-zone.jpg)
