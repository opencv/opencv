Color conversions {#imgproc_color_conversions}
=================
See cv::cvtColor and cv::ColorConversionCodes
@todo document other conversion modes

@anchor color_convert_rgb_gray
RGB \f$\leftrightarrow\f$ GRAY
------------------------------
Transformations within RGB space like adding/removing the alpha channel, reversing the channel
order, conversion to/from 16-bit RGB color (R5:G6:B5 or R5:G5:B5), as well as conversion
to/from grayscale using:
\f[\text{RGB[A] to Gray:} \quad Y  \leftarrow 0.299  \cdot R + 0.587  \cdot G + 0.114  \cdot B\f]
and
\f[\text{Gray to RGB[A]:} \quad R  \leftarrow Y, G  \leftarrow Y, B  \leftarrow Y, A  \leftarrow \max (ChannelRange)\f]
The conversion from a RGB image to gray is done with:
@code
    cvtColor(src, bwsrc, cv::COLOR_RGB2GRAY);
@endcode
More advanced channel reordering can also be done with cv::mixChannels.
@see cv::COLOR_BGR2GRAY, cv::COLOR_RGB2GRAY, cv::COLOR_GRAY2BGR, cv::COLOR_GRAY2RGB

@anchor color_convert_rgb_xyz
RGB \f$\leftrightarrow\f$ CIE XYZ.Rec 709 with D65 white point
--------------------------------------------------------------
\f[\begin{bmatrix} X  \\ Y  \\ Z
  \end{bmatrix} \leftarrow \begin{bmatrix} 0.412453 & 0.357580 & 0.180423 \\ 0.212671 & 0.715160 & 0.072169 \\ 0.019334 & 0.119193 & 0.950227
  \end{bmatrix} \cdot \begin{bmatrix} R  \\ G  \\ B
  \end{bmatrix}\f]
\f[\begin{bmatrix} R  \\ G  \\ B
  \end{bmatrix} \leftarrow \begin{bmatrix} 3.240479 & -1.53715 & -0.498535 \\ -0.969256 &  1.875991 & 0.041556 \\ 0.055648 & -0.204043 & 1.057311
  \end{bmatrix} \cdot \begin{bmatrix} X  \\ Y  \\ Z
  \end{bmatrix}\f]
\f$X\f$, \f$Y\f$ and \f$Z\f$ cover the whole value range (in case of floating-point images, \f$Z\f$ may exceed 1).

@see cv::COLOR_BGR2XYZ, cv::COLOR_RGB2XYZ, cv::COLOR_XYZ2BGR, cv::COLOR_XYZ2RGB

@anchor color_convert_rgb_ycrcb
RGB \f$\leftrightarrow\f$ YCrCb JPEG (or YCC)
---------------------------------------------
\f[Y  \leftarrow 0.299  \cdot R + 0.587  \cdot G + 0.114  \cdot B\f]
\f[Cr  \leftarrow (R-Y)  \cdot 0.713 + delta\f]
\f[Cb  \leftarrow (B-Y)  \cdot 0.564 + delta\f]
\f[R  \leftarrow Y + 1.403  \cdot (Cr - delta)\f]
\f[G  \leftarrow Y - 0.714  \cdot (Cr - delta) - 0.344  \cdot (Cb - delta)\f]
\f[B  \leftarrow Y + 1.773  \cdot (Cb - delta)\f]
where
\f[delta =  \left \{ \begin{array}{l l} 128 &  \mbox{for 8-bit images} \\ 32768 &  \mbox{for 16-bit images} \\ 0.5 &  \mbox{for floating-point images} \end{array} \right .\f]
Y, Cr, and Cb cover the whole value range.
@see cv::COLOR_BGR2YCrCb, cv::COLOR_RGB2YCrCb, cv::COLOR_YCrCb2BGR, cv::COLOR_YCrCb2RGB

@anchor color_convert_rgb_hsv
RGB \f$\leftrightarrow\f$ HSV
-----------------------------
In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and
scaled to fit the 0 to 1 range.

\f[V  \leftarrow max(R,G,B)\f]
\f[S  \leftarrow \fork{\frac{V-min(R,G,B)}{V}}{if \(V \neq 0\)}{0}{otherwise}\f]
\f[H  \leftarrow \forkthree{{60(G - B)}/{(V-min(R,G,B))}}{if \(V=R\)}{{120+60(B - R)}/{(V-min(R,G,B))}}{if \(V=G\)}{{240+60(R - G)}/{(V-min(R,G,B))}}{if \(V=B\)}\f]
If \f$H<0\f$ then \f$H \leftarrow H+360\f$ . On output \f$0 \leq V \leq 1\f$, \f$0 \leq S \leq 1\f$,
\f$0 \leq H \leq 360\f$ .

The values are then converted to the destination data type:
- 8-bit images: \f$V  \leftarrow 255 V, S  \leftarrow 255 S, H  \leftarrow H/2  \text{(to fit to 0 to 255)}\f$
- 16-bit images: (currently not supported) \f$V <- 65535 V, S <- 65535 S, H <- H\f$
- 32-bit images: H, S, and V are left as is

@see cv::COLOR_BGR2HSV, cv::COLOR_RGB2HSV, cv::COLOR_HSV2BGR, cv::COLOR_HSV2RGB

@anchor color_convert_rgb_hls
RGB \f$\leftrightarrow\f$ HLS
-----------------------------
In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and
scaled to fit the 0 to 1 range.

\f[V_{max}  \leftarrow {max}(R,G,B)\f]
\f[V_{min}  \leftarrow {min}(R,G,B)\f]
\f[L  \leftarrow \frac{V_{max} + V_{min}}{2}\f]
\f[S  \leftarrow \fork { \frac{V_{max} - V_{min}}{V_{max} + V_{min}} }{if  \(L < 0.5\) }
    { \frac{V_{max} - V_{min}}{2 - (V_{max} + V_{min})} }{if  \(L \ge 0.5\) }\f]
\f[H  \leftarrow \forkthree {{60(G - B)}/{(V_{max}-V_{min})}}{if  \(V_{max}=R\) }
  {{120+60(B - R)}/{(V_{max}-V_{min})}}{if  \(V_{max}=G\) }
  {{240+60(R - G)}/{(V_{max}-V_{min})}}{if  \(V_{max}=B\) }\f]
If \f$H<0\f$ then \f$H \leftarrow H+360\f$ . On output \f$0 \leq L \leq 1\f$, \f$0 \leq S \leq
1\f$, \f$0 \leq H \leq 360\f$ .

The values are then converted to the destination data type:
- 8-bit images:  \f$V  \leftarrow 255 \cdot V, S  \leftarrow 255 \cdot S, H  \leftarrow H/2 \; \text{(to fit to 0 to 255)}\f$
- 16-bit images: (currently not supported)  \f$V <- 65535 \cdot V, S <- 65535 \cdot S, H <- H\f$
- 32-bit images: H, S, V are left as is

@see cv::COLOR_BGR2HLS, cv::COLOR_RGB2HLS, cv::COLOR_HLS2BGR, cv::COLOR_HLS2RGB

@anchor color_convert_rgb_lab
RGB \f$\leftrightarrow\f$ CIE L\*a\*b\*
---------------------------------------
In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and
scaled to fit the 0 to 1 range.

\f[\vecthree{X}{Y}{Z} \leftarrow \vecthreethree{0.412453}{0.357580}{0.180423}{0.212671}{0.715160}{0.072169}{0.019334}{0.119193}{0.950227} \cdot \vecthree{R}{G}{B}\f]
\f[X  \leftarrow X/X_n,  \text{where} X_n = 0.950456\f]
\f[Z  \leftarrow Z/Z_n,  \text{where} Z_n = 1.088754\f]
\f[L  \leftarrow \fork{116*Y^{1/3}-16}{for \(Y>0.008856\)}{903.3*Y}{for \(Y \le 0.008856\)}\f]
\f[a  \leftarrow 500 (f(X)-f(Y)) + delta\f]
\f[b  \leftarrow 200 (f(Y)-f(Z)) + delta\f]
where
\f[f(t)= \fork{t^{1/3}}{for \(t>0.008856\)}{7.787 t+16/116}{for \(t\leq 0.008856\)}\f]
and
\f[delta =  \fork{128}{for 8-bit images}{0}{for floating-point images}\f]

This outputs \f$0 \leq L \leq 100\f$, \f$-127 \leq a \leq 127\f$, \f$-127 \leq b \leq 127\f$ . The values
are then converted to the destination data type:
- 8-bit images:  \f$L  \leftarrow L*255/100, \; a  \leftarrow a + 128, \; b  \leftarrow b + 128\f$
- 16-bit images:  (currently not supported)
- 32-bit images:  L, a, and b are left as is

@see cv::COLOR_BGR2Lab, cv::COLOR_RGB2Lab, cv::COLOR_Lab2BGR, cv::COLOR_Lab2RGB

@anchor color_convert_rgb_luv
RGB \f$\leftrightarrow\f$ CIE L\*u\*v\*
---------------------------------------
In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and
scaled to fit 0 to 1 range.

\f[\vecthree{X}{Y}{Z} \leftarrow \vecthreethree{0.412453}{0.357580}{0.180423}{0.212671}{0.715160}{0.072169}{0.019334}{0.119193}{0.950227} \cdot \vecthree{R}{G}{B}\f]
\f[L  \leftarrow \fork{116*Y^{1/3} - 16}{for \(Y>0.008856\)}{903.3 Y}{for \(Y\leq 0.008856\)}\f]
\f[u'  \leftarrow 4*X/(X + 15*Y + 3 Z)\f]
\f[v'  \leftarrow 9*Y/(X + 15*Y + 3 Z)\f]
\f[u  \leftarrow 13*L*(u' - u_n)  \quad \text{where} \quad u_n=0.19793943\f]
\f[v  \leftarrow 13*L*(v' - v_n)  \quad \text{where} \quad v_n=0.46831096\f]

This outputs \f$0 \leq L \leq 100\f$, \f$-134 \leq u \leq 220\f$, \f$-140 \leq v \leq 122\f$ .

The values are then converted to the destination data type:
-   8-bit images:  \f$L  \leftarrow 255/100 L, \; u  \leftarrow 255/354 (u + 134), \; v  \leftarrow 255/262 (v + 140)\f$
-   16-bit images:   (currently not supported)
-   32-bit images:   L, u, and v are left as is

Note that when converting integer Luv images to RGB the intermediate X, Y and Z values are truncated to \f$ [0, 2] \f$ range to fit white point limitations. It may lead to incorrect representation of colors with odd XYZ values.

The above formulae for converting RGB to/from various color spaces have been taken from multiple
sources on the web, primarily from the Charles Poynton site <http://www.poynton.com/ColorFAQ.html>

@see cv::COLOR_BGR2Luv, cv::COLOR_RGB2Luv, cv::COLOR_Luv2BGR, cv::COLOR_Luv2RGB

@anchor color_convert_bayer
Bayer \f$\rightarrow\f$ RGB
---------------------------
The Bayer pattern is widely used in CCD and CMOS cameras. It enables you to get color pictures
from a single plane where R,G, and B pixels (sensors of a particular component) are interleaved
as follows:

![Bayer pattern](pics/bayer.png)

The output RGB components of a pixel are interpolated from 1, 2, or 4 neighbors of the pixel
having the same color. There are several modifications of the above pattern that can be achieved
by shifting the pattern one pixel left and/or one pixel up. The two letters \f$C_1\f$ and \f$C_2\f$ in
the conversion constants CV_Bayer \f$C_1 C_2\f$ 2BGR and CV_Bayer \f$C_1 C_2\f$ 2RGB indicate the
particular pattern type. These are components from the second row, second and third columns,
respectively. For example, the above pattern has a very popular "BG" type.

@see cv::COLOR_BayerBG2BGR, cv::COLOR_BayerGB2BGR, cv::COLOR_BayerRG2BGR, cv::COLOR_BayerGR2BGR, cv::COLOR_BayerBG2RGB, cv::COLOR_BayerGB2RGB, cv::COLOR_BayerRG2RGB, cv::COLOR_BayerGR2RGB
