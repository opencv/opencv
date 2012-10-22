.. _Retina_Model:

Discovering the human retina and its use for image processing
*************************************************************

Goal
=====

In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   + discover the main two channels outing from your retina

   + see the basics to use the retina model

   + discover some parameters tweaks


.. image:: images/retina_TreeHdr_small.jpg
   :alt: Illustration of the retina luminance compression effect
   :align: center

For example in the above figure, a High Dynamic Range image (left) is processed by the retina model (right). The left image is coded on more than 8bit/color channel so that displaying this on 8bit format hides many details. However, as your retina does, using complementary processing. Here, local luminance adaptation, spatio-temporal noise removal and spectral whitening play an important role thus transmitting accurate information on low range data channels.
   
Theory
=======

The proposed model originates from Jeanny Herault at `Gipsa <http://www.gipsa-lab.inpg.fr>`_ preliminary work. It is involved in image processing applications with `Listic <http://www.listic.univ-savoie.fr>`_ (code maintainer) lab. The model allows the following human retina properties to be used :

* spectral whithening (mid-frequency details enhancement)

* high frequency spatio-temporal noise reduction (temporal noise and high frequency spatial noise are minimized)

* low frequency luminance energy reduction (also allows luminance range compression)

* local logarithmic luminance compression allows details to be enhanced even in low light conditions

The retina model presents two outputs that benefit from these behaviors.

* The first one is called the Parvocellular channel. It is mainly active in the foveal retina area (high resolution central vision with color sensitive photoreceptors), its aim is to provide accurate color vision for visual details remaining static on the retina. On the other hand objects moving on the retina projection are blurried.

* The second well known channel is the magnocellular channel. It is mainly active in the retina peripheral vision and send signals related to change events (motion, transient events, etc.). It helps visual system to focus/center retina on 'transient'/moving areas for more detailled analysis thus improving visual scene context and object classification.

NOTE : regarding the proposed model, we apply these two channels on the entire input images. This allows enhanced visual details and motion information to be extracted on all the considered images.  

As an illustration, we apply in the following the retina model on a webcam video stream of a dark visual scene. In this visual scene, captured in an amphitheater of the university, some students are moving while talking to the teacher. 


In this video sequence, because of the dark ambiance, signal to noise ratio is low and color artifacts are present on visual features edges because of the low quality image capture toolchain.

.. image:: images/studentsSample_input.jpg
   :alt: an input video stream extract sample
   :align: center

Below is shown the retina foveal vision applied on the entire image. In the used retina configuration, global luminance is preserved and local contrasts are enhanced. Also, signal to noise ratio is improved : since high frequency spatio-temporal noise is reduced, enhanced details are not corrupted by any enhanced noise.

.. image:: images/studentsSample_parvo.jpg
   :alt: the retina Parvocellular output. Enhanced details, luminance adaptation and noise removal. A processing tool for image analysis.
   :align: center

Below is the output of the magnocellular output of the retina model. Its signals are strong where transient events occur. Here, a student is moving at the bottom of the image thus generating high energy. The remaining of the image is static however, it is corrupted by a strong noise. Here, the retina filters out most of the noise thus generating low false motion area 'alarms'. This channel can be used as a transient/moving areas detector : it would provide relevant information for a low cost segmentation tool that would highlight areas in which an event is occuring.

.. image:: images/studentsSample_magno.jpg
   :alt: the retina Magnocellular output. Enhanced transient signals (motion, etc.). A preprocessing tool for event detection.
   :align: center


Use : this model can be used basically for spatio-temporal video effects but also in the aim of :
  
* performing texture analysis with enhanced signal to noise ratio and enhanced details robust against input images luminance ranges (check out the parvocellular retina channel output)

* performing motion analysis also taking benefit of the previously cited properties  (check out the magnocellular retina channel output)
For more information, refer to the following papers :

* Benoit A., Caplier A., Durette B., Herault, J., "Using Human Visual System Modeling For Bio-Inspired Low Level Image Processing", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773. DOI <http://dx.doi.org/10.1016/j.cviu.2010.01.011>

* Please have a look at the reference work of Jeanny Herault that you can read in his book :

Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.

This retina filter code includes the research contributions of phd/research collegues from which code has been redrawn by the author :

* take a look at the *retinacolor.hpp* module to discover Brice Chaix de Lavarene phD color mosaicing/demosaicing and his reference paper: B. Chaix de Lavarene, D. Alleysson, B. Durette, J. Herault (2007). "Efficient demosaicing through recursive filtering", IEEE International Conference on Image Processing ICIP 2007

* take a look at *imagelogpolprojection.hpp* to discover retina spatial log sampling which originates from Barthelemy Durette phd with Jeanny Herault. A Retina / V1 cortex projection is also proposed and originates from Jeanny's discussions. ====> more informations in the above cited Jeanny Heraults's book.


