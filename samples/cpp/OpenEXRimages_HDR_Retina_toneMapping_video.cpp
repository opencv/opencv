
//============================================================================
// Name        : OpenEXRimages_HDR_Retina_toneMapping_video.cpp
// Author      : Alexandre Benoit (benoit.alexandre.vision@gmail.com)
// Version     : 0.2
// Copyright   : Alexandre Benoit, LISTIC Lab, december 2011
// Description : HDR compression (tone mapping) for image sequences with the help of the Gipsa/Listic's retina in C++, Ansi-style
// Known issues: the input OpenEXR sequences can have bad computed pixels that should be removed
//               => a simple method consists of cutting histogram edges (a slider for this on the UI is provided)
//               => however, in image sequences, this histogramm cut must be done in an elegant way from frame to frame... still not done...
//============================================================================

#include <iostream>
#include <stdio.h>
#include <cstring>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

static void help(string errorMessage)
{
    cout<<"Program init error : "<<errorMessage<<endl;
    cout<<"\nProgram call procedure : ./OpenEXRimages_HDR_Retina_toneMapping [OpenEXR image sequence to process] [OPTIONNAL start frame] [OPTIONNAL end frame]"<<endl;
    cout<<"\t[OpenEXR image sequence to process] : sprintf style ready prototype filename of the input HDR images to process, must be an OpenEXR format, see http://www.openexr.com/ to get some samples or create your own using camera bracketing and Photoshop or equivalent software for OpenEXR image synthesis"<<endl;
    cout<<"\t\t => WARNING : image index number of digits cannot exceed 10"<<endl;
    cout<<"\t[start frame] : the starting frame tat should be considered"<<endl;
    cout<<"\t[end frame] : the ending frame tat should be considered"<<endl;
    cout<<"\nExamples:"<<endl;
    cout<<"\t-Image processing : ./OpenEXRimages_HDR_Retina_toneMapping_video memorial%3d.exr 20 45"<<endl;
    cout<<"\t-Image processing : ./OpenEXRimages_HDR_Retina_toneMapping_video memorial%3d.exr 20 45 log"<<endl;
    cout<<"\t ==> to process images from memorial020d.exr to memorial045d.exr"<<endl;

}

// simple procedure for 1D curve tracing
static void drawPlot(const Mat curve, const string figureTitle, const int lowerLimit, const int upperLimit)
{
    //cout<<"curve size(h,w) = "<<curve.size().height<<", "<<curve.size().width<<endl;
    Mat displayedCurveImage = Mat::ones(200, curve.size().height, CV_8U);

    Mat windowNormalizedCurve;
    normalize(curve, windowNormalizedCurve, 0, 200, CV_MINMAX, CV_32F);

    displayedCurveImage = Scalar::all(255); // set a white background
    int binW = cvRound((double)displayedCurveImage.cols/curve.size().height);

    for( int i = 0; i < curve.size().height; i++ )
        rectangle( displayedCurveImage, Point(i*binW, displayedCurveImage.rows),
                Point((i+1)*binW, displayedCurveImage.rows - cvRound(windowNormalizedCurve.at<float>(i))),
                Scalar::all(0), -1, 8, 0 );
    rectangle( displayedCurveImage, Point(0, 0),
            Point((lowerLimit)*binW, 200),
            Scalar::all(128), -1, 8, 0 );
    rectangle( displayedCurveImage, Point(displayedCurveImage.cols, 0),
            Point((upperLimit)*binW, 200),
            Scalar::all(128), -1, 8, 0 );

    imshow(figureTitle, displayedCurveImage);
}

/*
 * objective : get the gray level map of the input image and rescale it to the range [0-255] if rescale0_255=TRUE, simply trunks else
 */
static void rescaleGrayLevelMat(const Mat &inputMat, Mat &outputMat, const float histogramClippingLimit, const bool rescale0_255)
 {
     // adjust output matrix wrt the input size but single channel
     cout<<"Input image rescaling with histogram edges cutting (in order to eliminate bad pixels created during the HDR image creation) :"<<endl;
     //cout<<"=> image size (h,w,channels) = "<<inputMat.size().height<<", "<<inputMat.size().width<<", "<<inputMat.channels()<<endl;
     //cout<<"=> pixel coding (nbchannel, bytes per channel) = "<<inputMat.elemSize()/inputMat.elemSize1()<<", "<<inputMat.elemSize1()<<endl;

     // get min and max values to use afterwards if no 0-255 rescaling is used
     double maxInput, minInput, histNormRescalefactor=1.f;
     double histNormOffset=0.f;
     minMaxLoc(inputMat, &minInput, &maxInput);
     histNormRescalefactor=255.f/(maxInput-minInput);
     histNormOffset=minInput;
     cout<<"Hist max,min = "<<maxInput<<", "<<minInput<<" => scale, offset = "<<histNormRescalefactor<<", "<<histNormOffset<<endl;
     // rescale between 0-255, keeping floating point values
     Mat normalisedImage;
     normalize(inputMat, normalisedImage, 0.f, 255.f, NORM_MINMAX);
     if (rescale0_255)
        normalisedImage.copyTo(outputMat);
     // extract a 8bit image that will be used for histogram edge cut
     Mat intGrayImage;
     if (inputMat.channels()==1)
     {
         normalisedImage.convertTo(intGrayImage, CV_8U);
     }else
     {
         Mat rgbIntImg;
         normalisedImage.convertTo(rgbIntImg, CV_8UC3);
         cvtColor(rgbIntImg, intGrayImage, CV_BGR2GRAY);
     }

     // get histogram density probability in order to cut values under above edges limits (here 5-95%)... usefull for HDR pixel errors cancellation
     Mat dst, hist;
     int histSize = 256;
     calcHist(&intGrayImage, 1, 0, Mat(), hist, 1, &histSize, 0);
     Mat normalizedHist;

     normalize(hist, normalizedHist, 1.f, 0.f, NORM_L1, CV_32F); // normalize histogram so that its sum equals 1

     // compute density probability
     Mat denseProb=Mat::zeros(normalizedHist.size(), CV_32F);
     denseProb.at<float>(0)=normalizedHist.at<float>(0);
     int histLowerLimit=0, histUpperLimit=0;
     for (int i=1;i<normalizedHist.size().height;++i)
     {
         denseProb.at<float>(i)=denseProb.at<float>(i-1)+normalizedHist.at<float>(i);
         //cout<<normalizedHist.at<float>(i)<<", "<<denseProb.at<float>(i)<<endl;
         if ( denseProb.at<float>(i)<histogramClippingLimit)
             histLowerLimit=i;
         if ( denseProb.at<float>(i)<1.f-histogramClippingLimit)
             histUpperLimit=i;
     }
     // deduce min and max admitted gray levels
     float minInputValue = (float)histLowerLimit/histSize*255.f;
     float maxInputValue = (float)histUpperLimit/histSize*255.f;

     cout<<"=> Histogram limits "
             <<"\n\t"<<histogramClippingLimit*100.f<<"% index = "<<histLowerLimit<<" => normalizedHist value = "<<denseProb.at<float>(histLowerLimit)<<" => input gray level = "<<minInputValue
             <<"\n\t"<<(1.f-histogramClippingLimit)*100.f<<"% index = "<<histUpperLimit<<" => normalizedHist value = "<<denseProb.at<float>(histUpperLimit)<<" => input gray level = "<<maxInputValue
             <<endl;
     //drawPlot(denseProb, "input histogram density probability", histLowerLimit, histUpperLimit);
     drawPlot(normalizedHist, "input histogram", histLowerLimit, histUpperLimit);

    if(rescale0_255) // rescale between 0-255 if asked to
    {
        threshold( outputMat, outputMat, maxInputValue, maxInputValue, 2 ); //THRESH_TRUNC, clips values above maxInputValue
        threshold( outputMat, outputMat, minInputValue, minInputValue, 3 ); //THRESH_TOZERO, clips values under minInputValue
        // rescale image range [minInputValue-maxInputValue] to [0-255]
        outputMat-=minInputValue;
        outputMat*=255.f/(maxInputValue-minInputValue);
    }else
    {
        inputMat.copyTo(outputMat);
        // update threshold in the initial input image range
        maxInputValue=(float)((maxInputValue-255.f)/histNormRescalefactor+maxInput);
        minInputValue=(float)(minInputValue/histNormRescalefactor+minInput);
        cout<<"===> Input Hist clipping values (max,min) = "<<maxInputValue<<", "<<minInputValue<<endl;
        threshold( outputMat, outputMat, maxInputValue, maxInputValue, 2 ); //THRESH_TRUNC, clips values above maxInputValue
        threshold( outputMat, outputMat, minInputValue, minInputValue, 3 ); //
    }
 }

 // basic callback method for interface management
 Mat inputImage;
 Mat imageInputRescaled;
 float globalRescalefactor=1;
 Scalar globalOffset=0;
 int histogramClippingValue;
 static void callBack_rescaleGrayLevelMat(int, void*)
 {
     cout<<"Histogram clipping value changed, current value = "<<histogramClippingValue<<endl;
    // rescale and process
    inputImage+=globalOffset;
    inputImage*=globalRescalefactor;
    inputImage+=Scalar(50, 50, 50, 50); // WARNING value linked to the hardcoded value (200.0) used in the globalRescalefactor in order to center on the 128 mean value... experimental but... basic compromise
    rescaleGrayLevelMat(inputImage, imageInputRescaled, (float)histogramClippingValue/100.f, true);

 }

 Ptr<Retina> retina;
 int retinaHcellsGain;
 int localAdaptation_photoreceptors, localAdaptation_Gcells;
 static void callBack_updateRetinaParams(int, void*)
 {
     retina->setupOPLandIPLParvoChannel(true, true, (float)(localAdaptation_photoreceptors/200.0), 0.5f, 0.43f, (float)retinaHcellsGain, 1.f, 7.f, (float)(localAdaptation_Gcells/200.0));
 }

 int colorSaturationFactor;
 static void callback_saturateColors(int, void*)
 {
     retina->setColorSaturation(true, (float)colorSaturationFactor);
 }

// loadNewFrame : loads a n image wrt filename parameters. it also manages image rescaling/histogram edges cutting (acts differently at first image i.e. if firstTimeread=true)
static void loadNewFrame(const string filenamePrototype, const int currentFileIndex, const bool firstTimeread)
{
     char *currentImageName=NULL;
    currentImageName = (char*)malloc(sizeof(char)*filenamePrototype.size()+10);

    // grab the first frame
    sprintf(currentImageName, filenamePrototype.c_str(), currentFileIndex);

     //////////////////////////////////////////////////////////////////////////////
     // checking input media type (still image, video file, live video acquisition)
     cout<<"RetinaDemo: reading image : "<<currentImageName<<endl;
     // image processing case
     // declare the retina input buffer... that will be fed differently in regard of the input media
     inputImage = imread(currentImageName, -1); // load image in RGB mode
     cout<<"=> image size (h,w) = "<<inputImage.size().height<<", "<<inputImage.size().width<<endl;
     if (inputImage.empty())
     {
        help("could not load image, program end");
            return;;
         }

    // rescaling/histogram clipping stage
    // rescale between 0 and 1
    // TODO : take care of this step !!! maybe disable of do this in a nicer way ... each successive image should get the same transformation... but it depends on the initial image format
    double maxInput, minInput;
    minMaxLoc(inputImage, &minInput, &maxInput);
    cout<<"ORIGINAL IMAGE pixels values range (max,min) : "<<maxInput<<", "<<minInput<<endl;

    if (firstTimeread)
    {
        /* the first time, get the pixel values range and rougthly update scaling value
        in order to center values around 128 and getting a range close to [0-255],
        => actually using a little less in order to let some more flexibility in range evolves...
        */
        double maxInput1, minInput1;
        minMaxLoc(inputImage, &minInput1, &maxInput1);
        cout<<"FIRST IMAGE pixels values range (max,min) : "<<maxInput1<<", "<<minInput1<<endl;
        globalRescalefactor=(float)(50.0/(maxInput1-minInput1)); // less than 255 for flexibility... experimental value to be carefull about
        double channelOffset = -1.5*minInput;
        globalOffset= Scalar(channelOffset, channelOffset, channelOffset, channelOffset);
    }
    // call the generic input image rescaling callback
    callBack_rescaleGrayLevelMat(1,NULL);
}

 int main(int argc, char* argv[]) {
     // welcome message
     cout<<"*********************************************************************************"<<endl;
     cout<<"* Retina demonstration for High Dynamic Range compression (tone-mapping) : demonstrates the use of a wrapper class of the Gipsa/Listic Labs retina model."<<endl;
     cout<<"* This retina model allows spatio-temporal image processing (applied on still images, video sequences)."<<endl;
     cout<<"* This demo focuses demonstration of the dynamic compression capabilities of the model"<<endl;
     cout<<"* => the main application is tone mapping of HDR images (i.e. see on a 8bit display a more than 8bits coded (up to 16bits) image with details in high and low luminance ranges"<<endl;
     cout<<"* The retina model still have the following properties:"<<endl;
     cout<<"* => It applies a spectral whithening (mid-frequency details enhancement)"<<endl;
     cout<<"* => high frequency spatio-temporal noise reduction"<<endl;
     cout<<"* => low frequency luminance to be reduced (luminance range compression)"<<endl;
     cout<<"* => local logarithmic luminance compression allows details to be enhanced in low light conditions\n"<<endl;
     cout<<"* for more information, reer to the following papers :"<<endl;
     cout<<"* Benoit A., Caplier A., Durette B., Herault, J., \"USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING\", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011"<<endl;
     cout<<"* Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891."<<endl;
     cout<<"* => reports comments/remarks at benoit.alexandre.vision@gmail.com"<<endl;
     cout<<"* => more informations and papers at : http://sites.google.com/site/benoitalexandrevision/"<<endl;
     cout<<"*********************************************************************************"<<endl;
     cout<<"** WARNING : this sample requires OpenCV to be configured with OpenEXR support **"<<endl;
     cout<<"*********************************************************************************"<<endl;
     cout<<"*** You can use free tools to generate OpenEXR images from images sets   :    ***"<<endl;
     cout<<"*** =>  1. take a set of photos from the same viewpoint using bracketing      ***"<<endl;
     cout<<"*** =>  2. generate an OpenEXR image with tools like qtpfsgui.sourceforge.net ***"<<endl;
     cout<<"*** =>  3. apply tone mapping with this program                               ***"<<endl;
     cout<<"*********************************************************************************"<<endl;

     // basic input arguments checking
     if (argc<4)
     {
         help("bad number of parameter");
         return -1;
     }

     bool useLogSampling = !strcmp(argv[argc-1], "log"); // check if user wants retina log sampling processing

     int startFrameIndex=0, endFrameIndex=0, currentFrameIndex=0;
     sscanf(argv[2], "%d", &startFrameIndex);
     sscanf(argv[3], "%d", &endFrameIndex);
     string inputImageNamePrototype(argv[1]);

     //////////////////////////////////////////////////////////////////////////////
     // checking input media type (still image, video file, live video acquisition)
     cout<<"RetinaDemo: setting up system with first image..."<<endl;
     loadNewFrame(inputImageNamePrototype, startFrameIndex, true);

     if (inputImage.empty())
     {
        help("could not load image, program end");
            return -1;
         }

     //////////////////////////////////////////////////////////////////////////////
     // Program start in a try/catch safety context (Retina may throw errors)
     try
     {
         /* create a retina instance with default parameters setup, uncomment the initialisation you wanna test
          * -> if the last parameter is 'log', then activate log sampling (favour foveal vision and subsamples peripheral vision)
          */
         if (useLogSampling)
                {
                     retina = new Retina(inputImage.size(),true, RETINA_COLOR_BAYER, true, 2.0, 10.0);
                 }
         else// -> else allocate "classical" retina :
             retina = new Retina(inputImage.size());

        // save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
        retina->write("RetinaDefaultParameters.xml");

                 // desactivate Magnocellular pathway processing (motion information extraction) since it is not usefull here
                 retina->activateMovingContoursProcessing(false);

         // declare retina output buffers
         Mat retinaOutput_parvo;

         /////////////////////////////////////////////
         // prepare displays and interactions
         histogramClippingValue=0; // default value... updated with interface slider

         string retinaInputCorrected("Retina input image (with cut edges histogram for basic pixels error avoidance)");
         namedWindow(retinaInputCorrected,1);
         createTrackbar("histogram edges clipping limit", "Retina input image (with cut edges histogram for basic pixels error avoidance)",&histogramClippingValue,50,callBack_rescaleGrayLevelMat);

         string RetinaParvoWindow("Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping");
         namedWindow(RetinaParvoWindow, 1);
         colorSaturationFactor=3;
         createTrackbar("Color saturation", "Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", &colorSaturationFactor,5,callback_saturateColors);

         retinaHcellsGain=40;
         createTrackbar("Hcells gain", "Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping",&retinaHcellsGain,100,callBack_updateRetinaParams);

         localAdaptation_photoreceptors=197;
         localAdaptation_Gcells=190;
         createTrackbar("Ph sensitivity", "Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", &localAdaptation_photoreceptors,199,callBack_updateRetinaParams);
         createTrackbar("Gcells sensitivity", "Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", &localAdaptation_Gcells,199,callBack_updateRetinaParams);

        string powerTransformedInput("EXR image with basic processing : 16bits=>8bits with gamma correction");

         /////////////////////////////////////////////
         // apply default parameters of user interaction variables
         callBack_updateRetinaParams(1,NULL); // first call for default parameters setup
         callback_saturateColors(1, NULL);

         // processing loop with stop condition
         currentFrameIndex=startFrameIndex;
         while(currentFrameIndex <= endFrameIndex)
         {
             loadNewFrame(inputImageNamePrototype, currentFrameIndex, false);

             if (inputImage.empty())
             {
                cout<<"Could not load new image (index = "<<currentFrameIndex<<"), program end"<<endl;
                return -1;
             }
            // display input & process standard power transformation
            imshow("EXR image original image, 16bits=>8bits linear rescaling ", imageInputRescaled);
            Mat gammaTransformedImage;
            pow(imageInputRescaled, 1./5, gammaTransformedImage); // apply gamma curve: img = img ** (1./5)
            imshow(powerTransformedInput, gammaTransformedImage);
             // run retina filter
             retina->run(imageInputRescaled);
             // Retrieve and display retina output
             retina->getParvo(retinaOutput_parvo);
             imshow(retinaInputCorrected, imageInputRescaled/255.f);
             imshow(RetinaParvoWindow, retinaOutput_parvo);
             waitKey(4);
            // jump to next frame
            ++currentFrameIndex;
         }
     }catch(Exception e)
     {
         cerr<<"Error using Retina : "<<e.what()<<endl;
     }

     // Program end message
     cout<<"Retina demo end"<<endl;

     return 0;
 }
