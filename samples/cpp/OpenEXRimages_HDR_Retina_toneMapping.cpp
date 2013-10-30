
//============================================================================
// Name        : OpenEXRimages_HDR_Retina_toneMapping.cpp
// Author      : Alexandre Benoit (benoit.alexandre.vision@gmail.com)
// Version     : 0.1
// Copyright   : Alexandre Benoit, LISTIC Lab, july 2011
// Description : HDR compression (tone mapping) with the help of the Gipsa/Listic's retina in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cstring>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

static void help(string errorMessage)
{
    cout<<"Program init error : "<<errorMessage<<endl;
    cout<<"\nProgram call procedure : ./OpenEXRimages_HDR_Retina_toneMapping [OpenEXR image to process]"<<endl;
    cout<<"\t[OpenEXR image to process] : the input HDR image to process, must be an OpenEXR format, see http://www.openexr.com/ to get some samples or create your own using camera bracketing and Photoshop or equivalent software for OpenEXR image synthesis"<<endl;
    cout<<"\nExamples:"<<endl;
    cout<<"\t-Image processing : ./OpenEXRimages_HDR_Retina_toneMapping memorial.exr"<<endl;
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
 * objective : get the gray level map of the input image and rescale it to the range [0-255]
 */
 static void rescaleGrayLevelMat(const Mat &inputMat, Mat &outputMat, const float histogramClippingLimit)
 {

     // adjust output matrix wrt the input size but single channel
     cout<<"Input image rescaling with histogram edges cutting (in order to eliminate bad pixels created during the HDR image creation) :"<<endl;
     //cout<<"=> image size (h,w,channels) = "<<inputMat.size().height<<", "<<inputMat.size().width<<", "<<inputMat.channels()<<endl;
     //cout<<"=> pixel coding (nbchannel, bytes per channel) = "<<inputMat.elemSize()/inputMat.elemSize1()<<", "<<inputMat.elemSize1()<<endl;

     // rescale between 0-255, keeping floating point values
     normalize(inputMat, outputMat, 0.0, 255.0, NORM_MINMAX);

     // extract a 8bit image that will be used for histogram edge cut
     Mat intGrayImage;
     if (inputMat.channels()==1)
     {
         outputMat.convertTo(intGrayImage, CV_8U);
     }else
     {
         Mat rgbIntImg;
         outputMat.convertTo(rgbIntImg, CV_8UC3);
         cvtColor(rgbIntImg, intGrayImage, CV_BGR2GRAY);
     }

     // get histogram density probability in order to cut values under above edges limits (here 5-95%)... usefull for HDR pixel errors cancellation
     Mat dst, hist;
     int histSize = 256;
     calcHist(&intGrayImage, 1, 0, Mat(), hist, 1, &histSize, 0);
     Mat normalizedHist;
     normalize(hist, normalizedHist, 1, 0, NORM_L1, CV_32F); // normalize histogram so that its sum equals 1

     double min_val, max_val;
     Mat histArr(normalizedHist);
     cvMinMaxLoc(&histArr, &min_val, &max_val);
     //cout<<"Hist max,min = "<<max_val<<", "<<min_val<<endl;

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
         if ( denseProb.at<float>(i)<1-histogramClippingLimit)
             histUpperLimit=i;
     }
     // deduce min and max admitted gray levels
     float minInputValue = (float)histLowerLimit/histSize*255;
     float maxInputValue = (float)histUpperLimit/histSize*255;

     cout<<"=> Histogram limits "
             <<"\n\t"<<histogramClippingLimit*100<<"% index = "<<histLowerLimit<<" => normalizedHist value = "<<denseProb.at<float>(histLowerLimit)<<" => input gray level = "<<minInputValue
             <<"\n\t"<<(1-histogramClippingLimit)*100<<"% index = "<<histUpperLimit<<" => normalizedHist value = "<<denseProb.at<float>(histUpperLimit)<<" => input gray level = "<<maxInputValue
             <<endl;
     //drawPlot(denseProb, "input histogram density probability", histLowerLimit, histUpperLimit);
     drawPlot(normalizedHist, "input histogram", histLowerLimit, histUpperLimit);

     // rescale image range [minInputValue-maxInputValue] to [0-255]
     outputMat-=minInputValue;
     outputMat*=255.0/(maxInputValue-minInputValue);
     // cut original histogram and back project to original image
     threshold( outputMat, outputMat, 255.0, 255.0, 2 ); //THRESH_TRUNC, clips values above 255
     threshold( outputMat, outputMat, 0.0, 0.0, 3 ); //THRESH_TOZERO, clips values under 0

 }
 // basic callback method for interface management
 Mat inputImage;
 Mat imageInputRescaled;
 int histogramClippingValue;
 static void callBack_rescaleGrayLevelMat(int, void*)
 {
     cout<<"Histogram clipping value changed, current value = "<<histogramClippingValue<<endl;
     rescaleGrayLevelMat(inputImage, imageInputRescaled, (float)(histogramClippingValue/100.0));
     normalize(imageInputRescaled, imageInputRescaled, 0.0, 255.0, NORM_MINMAX);
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
     if (argc<2)
     {
         help("bad number of parameter");
         return -1;
     }

     bool useLogSampling = !strcmp(argv[argc-1], "log"); // check if user wants retina log sampling processing

     string inputImageName=argv[1];

     //////////////////////////////////////////////////////////////////////////////
     // checking input media type (still image, video file, live video acquisition)
     cout<<"RetinaDemo: processing image "<<inputImageName<<endl;
     // image processing case
     // declare the retina input buffer... that will be fed differently in regard of the input media
     inputImage = imread(inputImageName, -1); // load image in RGB mode
     cout<<"=> image size (h,w) = "<<inputImage.size().height<<", "<<inputImage.size().width<<endl;
     if (!inputImage.total())
     {
        help("could not load image, program end");
            return -1;
         }
     // rescale between 0 and 1
     normalize(inputImage, inputImage, 0.0, 1.0, NORM_MINMAX);
     Mat gammaTransformedImage;
     pow(inputImage, 1./5, gammaTransformedImage); // apply gamma curve: img = img ** (1./5)
     imshow("EXR image original image, 16bits=>8bits linear rescaling ", inputImage);
     imshow("EXR image with basic processing : 16bits=>8bits with gamma correction", gammaTransformedImage);
     if (inputImage.empty())
     {
         help("Input image could not be loaded, aborting");
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
         //inputRescaleMat = inputImage;
         //outputRescaleMat = imageInputRescaled;
         namedWindow("Retina input image (with cut edges histogram for basic pixels error avoidance)",1);
         createTrackbar("histogram edges clipping limit", "Retina input image (with cut edges histogram for basic pixels error avoidance)",&histogramClippingValue,50,callBack_rescaleGrayLevelMat);

         namedWindow("Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", 1);
         colorSaturationFactor=3;
         createTrackbar("Color saturation", "Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", &colorSaturationFactor,5,callback_saturateColors);

         retinaHcellsGain=40;
         createTrackbar("Hcells gain", "Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping",&retinaHcellsGain,100,callBack_updateRetinaParams);

         localAdaptation_photoreceptors=197;
         localAdaptation_Gcells=190;
         createTrackbar("Ph sensitivity", "Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", &localAdaptation_photoreceptors,199,callBack_updateRetinaParams);
         createTrackbar("Gcells sensitivity", "Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", &localAdaptation_Gcells,199,callBack_updateRetinaParams);


         /////////////////////////////////////////////
         // apply default parameters of user interaction variables
         rescaleGrayLevelMat(inputImage, imageInputRescaled, (float)histogramClippingValue/100);
         retina->setColorSaturation(true,(float)colorSaturationFactor);
         callBack_updateRetinaParams(1,NULL); // first call for default parameters setup

         // processing loop with stop condition
         bool continueProcessing=true;
         while(continueProcessing)
         {
             // run retina filter
             retina->run(imageInputRescaled);
             // Retrieve and display retina output
             retina->getParvo(retinaOutput_parvo);
             imshow("Retina input image (with cut edges histogram for basic pixels error avoidance)", imageInputRescaled/255.0);
             imshow("Retina Parvocellular pathway output : 16bit=>8bit image retina tonemapping", retinaOutput_parvo);
             waitKey(10);
         }
     }catch(Exception e)
     {
         cerr<<"Error using Retina : "<<e.what()<<endl;
     }

     // Program end message
     cout<<"Retina demo end"<<endl;

     return 0;
 }
