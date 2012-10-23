.. _Retina_Model:

Discovering the human retina and its use for image processing
*************************************************************

Goal
=====

I present here a model of human retina that shows some interesting properties for image preprocessing and enhancement.
In this tutorial you will learn how to:

.. container:: enumeratevisibleitemswithsquare

   + discover the main two channels outing from your retina

   + see the basics to use the retina model

   + discover some parameters tweaks


General overview
================

The proposed model originates from Jeanny Herault at `Gipsa <http://www.gipsa-lab.inpg.fr>`_. It is involved in image processing applications with `Listic <http://www.listic.univ-savoie.fr>`_ (code maintainer) lab. The model allows the following human retina properties to be used :

* spectral whithening (mid-frequency details enhancement and low frequency luminance energy reduction)

* high frequency spatio-temporal noise cancelling (temporal noise and high frequency spatial noise are minimized)

* local logarithmic luminance compression allows details to be enhanced even in low light conditions


.. image:: images/retina_TreeHdr_small.jpg
   :alt: Illustration of the retina luminance compression effect
   :align: center

For example in the above figure, a High Dynamic Range image (left) is processed by the retina model (right). The left image is coded on more than 8bit/color channel so that displaying this on 8bit format hides many details. However, as your retina does, using complementary processing. Here, local luminance adaptation, spatio-temporal noise removal and spectral whitening play an important role thus transmitting accurate information on low range data channels.

The retina model output channels
================================

The retina model presents two outputs that benefit from the above cited behaviors.

* The first one is called the Parvocellular channel. It is mainly active in the foveal retina area (high resolution central vision with color sensitive photoreceptors), its aim is to provide accurate color vision for visual details remaining static on the retina. On the other hand objects moving on the retina projection are blurried.

* The second well known channel is the magnocellular channel. It is mainly active in the retina peripheral vision and send signals related to change events (motion, transient events, etc.). These outing signals also help visual system to focus/center retina on 'transient'/moving areas for more detailled analysis thus improving visual scene context and object classification.

**NOTE :** regarding the proposed model, contrary to the real retina, we apply these two channels on the entire input images using the same resoltion. This allows enhanced visual details and motion information to be extracted on all the considered images... but remember, that these two channels are complementary, if Magnocellular channel gives strong energy in an area, then, the Parvo channel is certainly blurried there since there is a transient event.

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

Retina use case
===============

This model can be used basically for spatio-temporal video effects but also in the aim of :
  
* performing texture analysis with enhanced signal to noise ratio and enhanced details robust against input images luminance ranges (check out the parvocellular retina channel output)

* performing motion analysis also taking benefit of the previously cited properties.

For more information, refer to the following papers :

* Benoit A., Caplier A., Durette B., Herault, J., "Using Human Visual System Modeling For Bio-Inspired Low Level Image Processing", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773. DOI <http://dx.doi.org/10.1016/j.cviu.2010.01.011>

* Please have a look at the reference work of Jeanny Herault that you can read in his book :

Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891.

This retina filter code includes the research contributions of phd/research collegues from which code has been redrawn by the author :

* take a look at the *retinacolor.hpp* module to discover Brice Chaix de Lavarene phD color mosaicing/demosaicing and his reference paper: B. Chaix de Lavarene, D. Alleysson, B. Durette, J. Herault (2007). "Efficient demosaicing through recursive filtering", IEEE International Conference on Image Processing ICIP 2007

* take a look at *imagelogpolprojection.hpp* to discover retina spatial log sampling which originates from Barthelemy Durette phd with Jeanny Herault. A Retina / V1 cortex projection is also proposed and originates from Jeanny's discussions. ====> more informations in the above cited Jeanny Heraults's book.

Code tutorial
=============

Please refer to the original tutorial source code in file *opencv_folder/samples/cpp/tutorial_code/contrib/retina_tutorial.cpp*.

To compile it, assuming OpenCV is correctly installed, use the following command. It requires the opencv_core *(cv::Mat and friends objects management)*, opencv_highgui *(display and image/video read)* and opencv_contrib *(Retina description)* libraries to compile. 

.. code-block:: cpp

   // compile
   gcc retina_tutorial.cpp -o Retina_tuto -lopencv_core -lopencv_highgui -lopencv_contrib
   
   // Run commands : add 'log' as a last parameter to apply a spatial log sampling (simulates retina sampling)
   // run on webcam
   ./Retina_tuto -video
   // run on video file
   ./Retina_tuto -video myVideo.avi
   // run on an image
   ./Retina_tuto -image myPicture.jpg
   // run on an image with log sampling
   ./Retina_tuto -image myPicture.jpg log

Here is a code explanation :

Retina definition is present in the contrib package and a simple include allows to use it

.. code-block:: cpp

   #include "opencv2/opencv.hpp"

Provide user some hints to run the program with a help function

.. code-block:: cpp

   // the help procedure
   static void help(std::string errorMessage)
   {
    std::cout<<"Program init error : "<<errorMessage<<std::endl;
    std::cout<<"\nProgram call procedure : retinaDemo [processing mode] [Optional : media target] [Optional LAST parameter: \"log\" to activate retina log sampling]"<<std::endl;
    std::cout<<"\t[processing mode] :"<<std::endl;
    std::cout<<"\t -image : for still image processing"<<std::endl;
    std::cout<<"\t -video : for video stream processing"<<std::endl;
    std::cout<<"\t[Optional : media target] :"<<std::endl;
    std::cout<<"\t if processing an image or video file, then, specify the path and filename of the target to process"<<std::endl;
    std::cout<<"\t leave empty if processing video stream coming from a connected video device"<<std::endl;
    std::cout<<"\t[Optional : activate retina log sampling] : an optional last parameter can be specified for retina spatial log sampling"<<std::endl;
    std::cout<<"\t set \"log\" without quotes to activate this sampling, output frame size will be divided by 4"<<std::endl;
    std::cout<<"\nExamples:"<<std::endl;
    std::cout<<"\t-Image processing : ./retinaDemo -image lena.jpg"<<std::endl;
    std::cout<<"\t-Image processing with log sampling : ./retinaDemo -image lena.jpg log"<<std::endl;
    std::cout<<"\t-Video processing : ./retinaDemo -video myMovie.mp4"<<std::endl;
    std::cout<<"\t-Live video processing : ./retinaDemo -video"<<std::endl;
    std::cout<<"\nPlease start again with new parameters"<<std::endl;
    std::cout<<"****************************************************"<<std::endl;
    std::cout<<" NOTE : this program generates the default retina parameters file 'RetinaDefaultParameters.xml'"<<std::endl;
    std::cout<<" => you can use this to fine tune parameters and load them if you save to file 'RetinaSpecificParameters.xml'"<<std::endl;
   }

Then, start the main program and first declare a *cv::Mat* matrix in which input images will be loaded. Also allocate a *cv::VideoCapture* object ready to load video streams (if necessary)

.. code-block:: cpp

  int main(int argc, char* argv[]) {
    // declare the retina input buffer... that will be fed differently in regard of the input media
    cv::Mat inputFrame;
    cv::VideoCapture videoCapture; // in case a video media is used, its manager is declared here


In the main program, before processing, first check input command parameters

.. code-block:: cpp

  // welcome message
    std::cout<<"****************************************************"<<std::endl;
    std::cout<<"* Retina demonstration : demonstrates the use of is a wrapper class of the Gipsa/Listic Labs retina model."<<std::endl;
    std::cout<<"* This demo will try to load the file 'RetinaSpecificParameters.xml' (if exists).\nTo create it, copy the autogenerated template 'RetinaDefaultParameters.xml'.\nThen twaek it with your own retina parameters."<<std::endl;
    // basic input arguments checking
    if (argc<2)
    {
        help("bad number of parameter");
        return -1;
    }

    bool useLogSampling = !strcmp(argv[argc-1], "log"); // check if user wants retina log sampling processing

    std::string inputMediaType=argv[1];

    //////////////////////////////////////////////////////////////////////////////
    // checking input media type (still image, video file, live video acquisition)
    if (!strcmp(inputMediaType.c_str(), "-image") && argc >= 3)
    {
        std::cout<<"RetinaDemo: processing image "<<argv[2]<<std::endl;
        // image processing case
        inputFrame = cv::imread(std::string(argv[2]), 1); // load image in RGB mode
    }else
        if (!strcmp(inputMediaType.c_str(), "-video"))
        {
            if (argc == 2 || (argc == 3 && useLogSampling)) // attempt to grab images from a video capture device
            {
                videoCapture.open(0);
            }else// attempt to grab images from a video filestream
            {
                std::cout<<"RetinaDemo: processing video stream "<<argv[2]<<std::endl;
                videoCapture.open(argv[2]);
            }

            // grab a first frame to check if everything is ok
            videoCapture>>inputFrame;
        }else
        {
            // bad command parameter
            help("bad command parameter");
            return -1;
        }

Once all input parameters are processed, a first image should have been loaded, if not, display error and stop program :

.. code-block:: cpp

    if (inputFrame.empty())
    {
        help("Input media could not be loaded, aborting");
        return -1;
    }

Now, everything is ready to run the retina model. I propose here to allocate a retina instance and to manage the eventual log sampling option. The Retina constructor expects at least a cv::Size object that shows the input data size that will have to be managed. One can activate other options such as color and its related color multiplexing strategy (here Bayer multiplexing is chosen using enum cv::RETINA_COLOR_BAYER). If using log sampling, the image reduction factor (smaller output images) and log sampling strengh can be adjusted.

.. code-block:: cpp
	
	// pointer to a retina object
        cv::Ptr<cv::Retina> myRetina;

        // if the last parameter is 'log', then activate log sampling (favour foveal vision and subsamples peripheral vision)
        if (useLogSampling)
        {
            myRetina = new cv::Retina(inputFrame.size(), true, cv::RETINA_COLOR_BAYER, true, 2.0, 10.0);
        }
        else// -> else allocate "classical" retina :
            myRetina = new cv::Retina(inputFrame.size());

        
Once done, the proposed code writes a default xml file that contains the default parameters of the retina. This is usefull to make your own config using this template. Here generated template xml file is called *RetinaDefaultParameters.xml*.

.. code-block:: cpp

        // save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
        myRetina->write("RetinaDefaultParameters.xml");

In the following line, the retina attempts to load another xml file called *RetinaSpecificParameters.xml*. If you created it and introduced your own setup, it will be loaded, in the other case, default retina parameters are used.

.. code-block:: cpp

        // load parameters if file exists
        myRetina->setup("RetinaSpecificParameters.xml");

It is not required here but just to show it is possible, you can reset the retina buffers to zero to force it to forget past events.

.. code-block:: cpp

	// reset all retina buffers (imagine you close your eyes for a long time)
        myRetina->clearBuffers();

Now, it is time to run the retina ! First create some output buffers ready to receive the two retina channels outputs

.. code-block:: cpp

        // declare retina output buffers
        cv::Mat retinaOutput_parvo;
        cv::Mat retinaOutput_magno;

Then, run retina in a loop, load new frames from video sequence if necessary and retina outputs.

.. code-block:: cpp

        // processing loop with no stop condition
        while(true)
        {
            // if using video stream, then, grabbing a new frame, else, input remains the same
            if (videoCapture.isOpened())
                videoCapture>>inputFrame;

            // run retina filter on the loaded input frame
            myRetina->run(inputFrame);
            // Retrieve and display retina output
            myRetina->getParvo(retinaOutput_parvo);
            myRetina->getMagno(retinaOutput_magno);
            cv::imshow("retina input", inputFrame);
            cv::imshow("Retina Parvo", retinaOutput_parvo);
            cv::imshow("Retina Magno", retinaOutput_magno);
            cv::waitKey(10);
        }

That's done ! But if you want to secure the system, take care and manage Exceptions. The retina can throw some when it sees unrelevant data (no input frame, wrong setup, etc.)
Then, i recommend to surround all the retina code by a try/catch system like this :

.. code-block:: cpp

    try{
         // pointer to a retina object
         cv::Ptr<cv::Retina> myRetina;
         [---]
         // processing loop with no stop condition
         while(true)
         {
             [---]
         }

    }catch(cv::Exception e)
    {
        std::cerr<<"Error using Retina : "<<e.what()<<std::endl;
    }
