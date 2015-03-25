Retina : a Bio mimetic human retina model
*****************************************

.. highlight:: cpp

Retina
======

.. ocv:class:: Retina

Class which provides the main controls to the Gipsa/Listic labs human  retina model. Spatio-temporal filtering modelling the two main retina information channels :

* foveal vision for detailled color vision : the parvocellular pathway).

* periphearal vision for sensitive transient signals detection (motion and events) : the magnocellular pathway.

**NOTE : See the Retina tutorial in the tutorial/contrib section for complementary explanations.**

The retina can be settled up with various parameters, by default, the retina cancels mean luminance and enforces all details of the visual scene. In order to use your own parameters, you can use at least one time the *write(std::string fs)* method which will write a proper XML file with all default parameters. Then, tweak it on your own and reload them at any time using method *setup(std::string fs)*. These methods update a *Retina::RetinaParameters* member structure that is described hereafter. ::

  class Retina
  {
  public:
    // parameters setup instance
    struct RetinaParameters; // this class is detailled later

    // constructors
    Retina (Size inputSize);
    Retina (Size inputSize, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod=RETINA_COLOR_BAYER, const bool useRetinaLogSampling=false, const double reductionFactor=1.0, const double samplingStrenght=10.0);

    // main method for input frame processing
    void run (const Mat &inputImage);

    // output buffers retreival methods
    // -> foveal color vision details channel with luminance and noise correction
    void getParvo (Mat &retinaOutput_parvo);
    void getParvo (std::valarray< float > &retinaOutput_parvo);
    const std::valarray< float > & getParvo () const;
    // -> peripheral monochrome motion and events (transient information) channel
    void getMagno (Mat &retinaOutput_magno);
    void getMagno (std::valarray< float > &retinaOutput_magno);
    const std::valarray< float > & getMagno () const;

    // reset retina buffers... equivalent to closing your eyes for some seconds
    void clearBuffers ();

    // retreive input and output buffers sizes
    Size inputSize ();
    Size outputSize ();

    // setup methods with specific parameters specification of global xml config file loading/write
    void setup (std::string retinaParameterFile="", const bool applyDefaultSetupOnFailure=true);
    void setup (FileStorage &fs, const bool applyDefaultSetupOnFailure=true);
    void setup (RetinaParameters newParameters);
    struct Retina::RetinaParameters getParameters ();
    const std::string printSetup ();
    virtual void write (std::string fs) const;
    virtual void write (FileStorage &fs) const;
    void setupOPLandIPLParvoChannel (const bool colorMode=true, const bool normaliseOutput=true, const float photoreceptorsLocalAdaptationSensitivity=0.7, const float photoreceptorsTemporalConstant=0.5, const float photoreceptorsSpatialConstant=0.53, const float horizontalCellsGain=0, const float HcellsTemporalConstant=1, const float HcellsSpatialConstant=7, const float ganglionCellsSensitivity=0.7);
    void setupIPLMagnoChannel (const bool normaliseOutput=true, const float parasolCells_beta=0, const float parasolCells_tau=0, const float parasolCells_k=7, const float amacrinCellsTemporalCutFrequency=1.2, const float V0CompressionParameter=0.95, const float localAdaptintegration_tau=0, const float localAdaptintegration_k=7);
    void setColorSaturation (const bool saturateColors=true, const float colorSaturationValue=4.0);
    void activateMovingContoursProcessing (const bool activate);
    void activateContoursProcessing (const bool activate);
  };

.. Sample code::

   * An example on retina tone mapping can be found at opencv_source_code/samples/cpp/OpenEXRimages_HDR_Retina_toneMapping.cpp
   * An example on retina tone mapping on video input can be found at opencv_source_code/samples/cpp/OpenEXRimages_HDR_Retina_toneMapping.cpp
   * A complete example illustrating the retina interface can be found at opencv_source_code/samples/cpp/retinaDemo.cpp

Description
+++++++++++

Class which allows the `Gipsa <http://www.gipsa-lab.inpg.fr>`_ (preliminary work) / `Listic <http://www.listic.univ-savoie.fr>`_ (code maintainer) labs retina model to be used. This class allows human retina spatio-temporal image processing to be applied on still images, images sequences and video sequences. Briefly, here are the main human retina model properties:

* spectral whithening (mid-frequency details enhancement)

* high frequency spatio-temporal noise reduction (temporal noise and high frequency spatial noise are minimized)

* low frequency luminance reduction (luminance range compression) : high luminance regions do not hide details in darker regions anymore

* local logarithmic luminance compression allows details to be enhanced even in low light conditions

Use : this model can be used basically for spatio-temporal video effects but also in the aim of :

* performing texture analysis with enhanced signal to noise ratio and enhanced details robust against input images luminance ranges (check out the parvocellular retina channel output, by using the provided **getParvo** methods)

* performing motion analysis also taking benefit of the previously cited properties  (check out the magnocellular retina channel output, by using the provided **getMagno** methods)

For more information, refer to the following papers :

* Benoit A., Caplier A., Durette B., Herault, J., "Using Human Visual System Modeling For Bio-Inspired Low Level Image Processing", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773. DOI <http://dx.doi.org/10.1016/j.cviu.2010.01.011>

* Please have a look at the reference work of Jeanny Herault that you can read in his book :

Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.

This retina filter code includes the research contributions of phd/research collegues from which code has been redrawn by the author :

* take a look at the *retinacolor.hpp* module to discover Brice Chaix de Lavarene phD color mosaicing/demosaicing and his reference paper: B. Chaix de Lavarene, D. Alleysson, B. Durette, J. Herault (2007). "Efficient demosaicing through recursive filtering", IEEE International Conference on Image Processing ICIP 2007

* take a look at *imagelogpolprojection.hpp* to discover retina spatial log sampling which originates from Barthelemy Durette phd with Jeanny Herault. A Retina / V1 cortex projection is also proposed and originates from Jeanny's discussions. ====> more informations in the above cited Jeanny Heraults's book.

Demos and experiments !
=======================

**NOTE : Complementary to the following examples, have a look at the Retina tutorial in the tutorial/contrib section for complementary explanations.**

Take a look at the provided C++ examples provided with OpenCV :

* **samples/cpp/retinademo.cpp** shows how to use the retina module for details enhancement (Parvo channel output) and transient maps observation (Magno channel output). You can play with images, video sequences and webcam video.
    Typical uses are (provided your OpenCV installation is situated in folder *OpenCVReleaseFolder*)

    * image processing : **OpenCVReleaseFolder/bin/retinademo -image myPicture.jpg**

    * video processing : **OpenCVReleaseFolder/bin/retinademo -video myMovie.avi**

    * webcam processing: **OpenCVReleaseFolder/bin/retinademo -video**

   **Note :** This demo generates the file *RetinaDefaultParameters.xml* which contains the default parameters of the retina. Then, rename this as *RetinaSpecificParameters.xml*, adjust the parameters the way you want and reload the program to check the effect.


* **samples/cpp/OpenEXRimages_HDR_Retina_toneMapping.cpp** shows how to use the retina to perform High Dynamic Range (HDR) luminance compression

   Then, take a HDR image using bracketing with your camera and generate an OpenEXR image and then process it using the demo.

   Typical use, supposing that you have the OpenEXR image *memorial.exr* (present in the samples/cpp/ folder)

   **OpenCVReleaseFolder/bin/OpenEXRimages_HDR_Retina_toneMapping memorial.exr**

      Note that some sliders are made available to allow you to play with luminance compression.


Methods description
===================

Here are detailled the main methods to control the retina model

Retina::Retina
++++++++++++++

.. ocv:function:: Retina::Retina(Size inputSize)
.. ocv:function:: Retina::Retina(Size inputSize, const bool colorMode, RETINA_COLORSAMPLINGMETHOD colorSamplingMethod = RETINA_COLOR_BAYER, const bool useRetinaLogSampling = false, const double reductionFactor = 1.0, const double samplingStrenght = 10.0 )

    Constructors

    :param inputSize: the input frame size
    :param colorMode: the chosen processing mode : with or without color processing
    :param colorSamplingMethod: specifies which kind of color sampling will be used
        * RETINA_COLOR_RANDOM: each pixel position is either R, G or B in a random choice
        * RETINA_COLOR_DIAGONAL: color sampling is RGBRGBRGB..., line 2 BRGBRGBRG..., line 3, GBRGBRGBR...
        * RETINA_COLOR_BAYER: standard bayer sampling
    :param useRetinaLogSampling: activate retina log sampling, if true, the 2 following parameters can be used
    :param reductionFactor: only usefull if param useRetinaLogSampling=true, specifies the reduction factor of the output frame (as the center (fovea) is high resolution and corners can be underscaled, then a reduction of the output is allowed without precision leak
    :param samplingStrenght: only usefull if param useRetinaLogSampling=true, specifies the strenght of the log scale that is applied

Retina::activateContoursProcessing
++++++++++++++++++++++++++++++++++

.. ocv:function:: void Retina::activateContoursProcessing(const bool activate)

    Activate/desactivate the Parvocellular pathway processing (contours information extraction), by default, it is activated

    :param activate: true if Parvocellular (contours information extraction) output should be activated, false if not... if activated, the Parvocellular output can be retrieved using the **getParvo** methods

Retina::activateMovingContoursProcessing
++++++++++++++++++++++++++++++++++++++++

.. ocv:function:: void Retina::activateMovingContoursProcessing(const bool activate)

    Activate/desactivate the Magnocellular pathway processing (motion information extraction), by default, it is activated

    :param activate: true if Magnocellular output should be activated, false if not... if activated, the Magnocellular output can be retrieved using the **getMagno** methods

Retina::clearBuffers
++++++++++++++++++++

.. ocv:function:: void Retina::clearBuffers()

    Clears all retina buffers (equivalent to opening the eyes after a long period of eye close ;o) whatchout the temporal transition occuring just after this method call.

Retina::getParvo
++++++++++++++++

.. ocv:function:: void Retina::getParvo( Mat & retinaOutput_parvo )
.. ocv:function:: void Retina::getParvo( std::valarray<float> & retinaOutput_parvo )
.. ocv:function:: const std::valarray<float> & Retina::getParvo() const

    Accessor of the details channel of the retina (models foveal vision)

    :param retinaOutput_parvo: the output buffer (reallocated if necessary), format can be :

        * a Mat, this output is rescaled for standard 8bits image processing use in OpenCV

        * a 1D std::valarray Buffer (encoding is R1, R2, ... Rn), this output is the original retina filter model output, without any quantification or rescaling

Retina::getMagno
++++++++++++++++

.. ocv:function:: void Retina::getMagno( Mat & retinaOutput_magno )
.. ocv:function:: void Retina::getMagno( std::valarray<float> & retinaOutput_magno )
.. ocv:function:: const std::valarray<float> & Retina::getMagno() const

    Accessor of the motion channel of the retina (models peripheral vision)

    :param retinaOutput_magno: the output buffer (reallocated if necessary), format can be :

        * a Mat, this output is rescaled for standard 8bits image processing use in OpenCV

        * a 1D std::valarray Buffer (encoding is R1, R2, ... Rn), this output is the original retina filter model output, without any quantification or rescaling

Retina::getParameters
+++++++++++++++++++++

.. ocv:function:: Retina::RetinaParameters Retina::getParameters()

    Retrieve the current parameters values in a *Retina::RetinaParameters* structure

    :return: the current parameters setup

Retina::inputSize
+++++++++++++++++

.. ocv:function:: Size Retina::inputSize()

    Retreive retina input buffer size

    :return: the retina input buffer size

Retina::outputSize
++++++++++++++++++

.. ocv:function:: Size Retina::outputSize()

    Retreive retina output buffer size that can be different from the input if a spatial log transformation is applied

    :return: the retina output buffer size

Retina::printSetup
++++++++++++++++++

.. ocv:function:: const std::string Retina::printSetup()

    Outputs a string showing the used parameters setup

    :return: a string which contains formatted parameters information

Retina::run
+++++++++++

.. ocv:function:: void Retina::run(const Mat & inputImage)

    Method which allows retina to be applied on an input image, after run, encapsulated retina module is ready to deliver its outputs using dedicated acccessors, see getParvo and getMagno methods

    :param inputImage: the input Mat image to be processed, can be gray level or BGR coded in any format (from 8bit to 16bits)

Retina::setColorSaturation
++++++++++++++++++++++++++

.. ocv:function:: void Retina::setColorSaturation(const bool saturateColors = true, const float colorSaturationValue = 4.0 )

    Activate color saturation as the final step of the color demultiplexing process -> this saturation is a sigmoide function applied to each channel of the demultiplexed image.

    :param saturateColors: boolean that activates color saturation (if true) or desactivate (if false)
    :param colorSaturationValue: the saturation factor : a simple factor applied on the chrominance buffers


Retina::setup
+++++++++++++

.. ocv:function:: void Retina::setup(std::string retinaParameterFile = "", const bool applyDefaultSetupOnFailure = true )
.. ocv:function:: void Retina::setup(FileStorage & fs, const bool applyDefaultSetupOnFailure = true )
.. ocv:function:: void Retina::setup(RetinaParameters newParameters)

    Try to open an XML retina parameters file to adjust current retina instance setup => if the xml file does not exist, then default setup is applied => warning, Exceptions are thrown if read XML file is not valid

    :param retinaParameterFile: the parameters filename
    :param applyDefaultSetupOnFailure: set to true if an error must be thrown on error
    :param fs: the open Filestorage which contains retina parameters
    :param newParameters: a parameters structures updated with the new target configuration

Retina::write
+++++++++++++

.. ocv:function:: void Retina::write( std::string fs ) const
.. ocv:function:: void Retina::write( FileStorage& fs ) const

    Write xml/yml formated parameters information

    :param fs: the filename of the xml file that will be open and writen with formatted parameters information

Retina::setupIPLMagnoChannel
++++++++++++++++++++++++++++

.. ocv:function:: void Retina::setupIPLMagnoChannel(const bool normaliseOutput = true, const float parasolCells_beta = 0, const float parasolCells_tau = 0, const float parasolCells_k = 7, const float amacrinCellsTemporalCutFrequency = 1.2f, const float V0CompressionParameter = 0.95f, const float localAdaptintegration_tau = 0, const float localAdaptintegration_k = 7 )

    Set parameters values for the Inner Plexiform Layer (IPL) magnocellular channel this channel processes signals output from OPL processing stage in peripheral vision, it allows motion information enhancement. It is decorrelated from the details channel. See reference papers for more details.

    :param normaliseOutput: specifies if (true) output is rescaled between 0 and 255 of not (false)
    :param parasolCells_beta: the low pass filter gain used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), typical value is 0
    :param parasolCells_tau: the low pass filter time constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is frame, typical value is 0 (immediate response)
    :param parasolCells_k: the low pass filter spatial constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is pixels, typical value is 5
    :param amacrinCellsTemporalCutFrequency: the time constant of the first order high pass fiter of the magnocellular way (motion information channel), unit is frames, typical value is 1.2
    :param V0CompressionParameter: the compression strengh of the ganglion cells local adaptation output, set a value between 0.6 and 1 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 0.95
    :param localAdaptintegration_tau: specifies the temporal constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
    :param localAdaptintegration_k: specifies the spatial constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation

Retina::setupOPLandIPLParvoChannel
++++++++++++++++++++++++++++++++++

.. ocv:function:: void Retina::setupOPLandIPLParvoChannel(const bool colorMode = true, const bool normaliseOutput = true, const float photoreceptorsLocalAdaptationSensitivity = 0.7f, const float photoreceptorsTemporalConstant = 0.5f, const float photoreceptorsSpatialConstant = 0.53f, const float horizontalCellsGain = 0, const float HcellsTemporalConstant = 1, const float HcellsSpatialConstant = 7, const float ganglionCellsSensitivity = 0.7f )

    Setup the OPL and IPL parvo channels (see biologocal model) OPL is referred as Outer Plexiform Layer of the retina, it allows the spatio-temporal filtering which withens the spectrum and reduces spatio-temporal noise while attenuating global luminance (low frequency energy) IPL parvo is the OPL next processing stage, it refers to a part of the Inner Plexiform layer of the retina, it allows high contours sensitivity in foveal vision. See reference papers for more informations.

    :param colorMode: specifies if (true) color is processed of not (false) to then processing gray level image
    :param normaliseOutput: specifies if (true) output is rescaled between 0 and 255 of not (false)
    :param photoreceptorsLocalAdaptationSensitivity: the photoreceptors sensitivity renage is 0-1 (more log compression effect when value increases)
    :param photoreceptorsTemporalConstant: the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
    :param photoreceptorsSpatialConstant: the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
    :param horizontalCellsGain: gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
    :param HcellsTemporalConstant: the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
    :param HcellsSpatialConstant: the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel, this value is also used for local contrast computing when computing the local contrast adaptation at the ganglion cells level (Inner Plexiform Layer parvocellular channel model)
    :param ganglionCellsSensitivity: the compression strengh of the ganglion cells local adaptation output, set a value between 0.6 and 1 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 0.7


Retina::RetinaParameters
========================

.. ocv:struct:: Retina::RetinaParameters

  This structure merges all the parameters that can be adjusted threw the **Retina::setup()**, **Retina::setupOPLandIPLParvoChannel** and **Retina::setupIPLMagnoChannel** setup methods
  Parameters structure for better clarity, check explenations on the comments of methods : setupOPLandIPLParvoChannel and setupIPLMagnoChannel. ::

    class RetinaParameters{
        struct OPLandIplParvoParameters{ // Outer Plexiform Layer (OPL) and Inner Plexiform Layer Parvocellular (IplParvo) parameters
               OPLandIplParvoParameters():colorMode(true),
                  normaliseOutput(true), // specifies if (true) output is rescaled between 0 and 255 of not (false)
                  photoreceptorsLocalAdaptationSensitivity(0.7f), // the photoreceptors sensitivity renage is 0-1 (more log compression effect when value increases)
                  photoreceptorsTemporalConstant(0.5f),// the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
                  photoreceptorsSpatialConstant(0.53f),// the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
                  horizontalCellsGain(0.0f),//gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
                  hcellsTemporalConstant(1.f),// the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
                  hcellsSpatialConstant(7.f),//the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel, this value is also used for local contrast computing when computing the local contrast adaptation at the ganglion cells level (Inner Plexiform Layer parvocellular channel model)
                  ganglionCellsSensitivity(0.7f)//the compression strengh of the ganglion cells local adaptation output, set a value between 0.6 and 1 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 0.7
                  {};// default setup
               bool colorMode, normaliseOutput;
               float photoreceptorsLocalAdaptationSensitivity, photoreceptorsTemporalConstant, photoreceptorsSpatialConstant, horizontalCellsGain, hcellsTemporalConstant, hcellsSpatialConstant, ganglionCellsSensitivity;
           };
           struct IplMagnoParameters{ // Inner Plexiform Layer Magnocellular channel (IplMagno)
               IplMagnoParameters():
                  normaliseOutput(true), //specifies if (true) output is rescaled between 0 and 255 of not (false)
                  parasolCells_beta(0.f), // the low pass filter gain used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), typical value is 0
                  parasolCells_tau(0.f), //the low pass filter time constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is frame, typical value is 0 (immediate response)
                  parasolCells_k(7.f), //the low pass filter spatial constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is pixels, typical value is 5
                  amacrinCellsTemporalCutFrequency(1.2f), //the time constant of the first order high pass fiter of the magnocellular way (motion information channel), unit is frames, typical value is 1.2
                  V0CompressionParameter(0.95f), the compression strengh of the ganglion cells local adaptation output, set a value between 0.6 and 1 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 0.95
                  localAdaptintegration_tau(0.f), // specifies the temporal constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
                  localAdaptintegration_k(7.f) // specifies the spatial constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
                  {};// default setup
               bool normaliseOutput;
               float parasolCells_beta, parasolCells_tau, parasolCells_k, amacrinCellsTemporalCutFrequency, V0CompressionParameter, localAdaptintegration_tau, localAdaptintegration_k;
           };
            struct OPLandIplParvoParameters OPLandIplParvo;
            struct IplMagnoParameters IplMagno;
    };
