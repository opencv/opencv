HDR imaging
==========

.. highlight:: cpp

makeHDR
-----------
Creates HDR image from a set of bracketed exposures using algorithm by Debevec and Malik.

"Recovering High Dynamic Range Radiance Maps from Photographs", Debevec, Malik, 1997

.. ocv:function:: void makeHDR(InputArrayOfArrays srcImgs, const std::vector<float>& expTimes, OutputArray dst, bool align = false)
   
   :param src_imgs: vector of 8-bit 3-channel images
    
   :param exp_times: exposure times for each of source images
   
   :param dst: output image
   
   :param align: if true, images are first aligned using median threshold bitmap algorithm. See :ocv:func:`getExpShift`.
   
tonemap
-----------
Tonemaps image.

.. ocv:function:: tonemap(InputArray src, OutputArray dst, int algorithm, const std::vector<float>& params = std::vector<float>())
   
   :param src: input HDR image
    
   :param dst: floating-point image in [0; 1] range
   
   :param algorithm:   
                     * TONEMAP_LINEAR - simple linear mapping
                     
                     * TONEMAP_DRAGO - "Adaptive Logarithmic Mapping For Displaying HighContrast Scenes", Drago et al., 2003
                     
                     * TONEMAP_REINHARD - "Dynamic Range Reduction Inspired by Photoreceptor Physiology", Reinhard, Devlin, 2005
                     
                     * TONEMAP_DURAND - "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images", Durand, Dorsey, 2002
   
   :param params: vector of parameters for specified algorithm.
                  If some parameters are missing default values are used.
                  The first element is gamma value for gamma correction.
                  
                  * TONEMAP_LINEAR: 
				  
                      No parameters.
                  
                  * TONEMAP_DRAGO:
				  
                      params[1] - value for bias function. Range [0.7, 0.9], default 0.85.
                  
                  * TONEMAP_REINHARD: 
                  
                      params[1] - result intensity. Range [-8, 8], default 0.
                  
                      params[2] - chromatic adaptation. Range [0, 1], default 0.
                                     
                      params[3] - light adaptation. Range [0, 1], default 0;
                                     
                  * TONEMAP_DURAND: 

                      params[1] - result contrast on logarithmic scale.
                                     
                      params[2] - bilateral filter sigma in the color space.
                                   
                      params[3] - bilateral filter sigma in the coordinate space.
   
exposureFusion
-----------
Fuses a bracketed exposure sequence into a single image without converting to HDR first.

"Exposure Fusion", Mertens et al., 2007

.. ocv:function:: exposureFusion(InputArrayOfArrays src_imgs, OutputArray dst, bool align = false, float wc = 1, float ws = 1, float we = 0)
   
   :param src_imgs: vector of 8-bit 3-channel images
   
   :param dst: output image. Although it's a floating-point image tonemapping is not necessary.
   
   :param align: if true, images are first aligned using median threshold bitmap algorithm. See :ocv:func:`getExpShift`.
   
   :param wc: contrast factor weight
   
   :param ws: saturation factor weight
   
   :param we: well-exposedness factor weight
   
getExpShift
-----------
Calculates translation vector that can be used to align img1 with img0.
Uses median threshold bitmap algorithm by Ward.

"Fast, Robust Image Registration for Compositing High Dynamic Range Photographs from Handheld Exposures", Ward, 2003

.. ocv:function:: getExpShift(InputArray img0, InputArray img1, int max_bits = 6, int exclude_range = 4)
   
   :param img0: 8-bit 1-channel image
    
   :param img1: 8-bit 1-channel image
   
   :param max_bits: logarithm to the base 2 of maximal shift in each dimension
   
   :param exclude_range: range value for exclusion bitmap. Refer to the article.
  
shiftMat
-----------
Shifts image filling the new regions with zeros.

.. ocv:function:: shiftMat(InputArray src, Point shift, OutputArray dst)
   
   :param src: input image
    
   :param shift: shift vector
   
   :param dst: output image