/**
 * @file buttonsqt.cpp
 * @brief Demo Application to test some new interface functions of HighGUI-Module        
 * @author Harald Schmidt
 */

#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

#ifdef _WIN32
    #include <io.h>             // _findfirst, _findnext
#else
    #include <unistd.h>
    #include <dirent.h>
#endif

using namespace cv;
using namespace std;


void UpdateContent( string strpos, string name );
void calcSaturation( Mat original, int iSatur );

/**
 * @function UpdateContent
 * @brief Interface-Function to set content in buttonbar
 */
void UpdateContent( string strpos, string name )
{
    try {
        setButtonBarContent("ColorImg", EMOD_Label, 1, (char *) strpos.c_str());
        setButtonBarContent("ColorImg", EMOD_Label, 2, (char *) name.c_str());
    }
    catch (const std::exception &e)
    {
        std::cout << "setButtonBarContent Error: " << e.what() << endl;
    }
    printf("\n");
}

void calcSaturation( Mat original, int iSatur )
{
    Mat hsv,melt,mix;
    vector<Mat> planes;

    cvtColor( original, hsv, CV_BGR2HSV );
    split(hsv,planes);	// planes[1] == Saturation 
    double scale = iSatur / 100.0;

    planes[1] = Mat_<uchar>(planes[1] * scale); 
    merge(planes, melt);
    cvtColor(melt,mix,CV_HSV2BGR);
    cv::imshow("Saturation", mix);
}


/**
 * @function main
 * @brief Main function
 */
int main( int , char** argv )
{
    Mat img;
    Mat imgArray[5];
    string nameVec[5];
    string missing = "";
    
    string exe_name = argv[0];  
    int ipos = exe_name.rfind("/");
    if ( ipos >= 0 ) exe_name = &exe_name[ipos+1];
    
    string winOrg = "ColorImg";
    
    // read window position + size from *.cfg
    cv::ConfigBase myCfg;
    cv::readConfig( exe_name.c_str(), winOrg.c_str(), &myCfg );

    int initPosX       = myCfg.initPosX;
    int initPosY       = myCfg.initPosY;
    int initWidth      = myCfg.initWidth;
    int initHeight     = myCfg.initHeight;
    int Mode           = myCfg.WindowMode;

    double ticks = (double) getTickCount();
    int sec = (int) (ticks / getTickFrequency());
    int randomcode = (sec %2);  

    int WinMode = CV_WINDOW_NORMAL;

    if ( Mode >= 0 )
    {
	// set WinMode not by random, but by entry in *.cfg
        WinMode = myCfg.WindowMode;
	cout << "\n" << exe_name << " pos=" << initPosX << "," 
         << initPosY << " (" << initWidth << "*" << initHeight 
         << ") Mode=" << Mode << " (buttonsqt.cpp)" << endl;
    
    } else {
	// set WinMode by random value..... 
	if ( randomcode == 0 ) 
	{ 
	  WinMode = CV_WINDOW_AUTOSIZE;
	  printf("\nrandom startUp: CV_WINDOW_AUTOSIZE\n" );
	} else {
	  printf("\nrandom startUp: CV_WINDOW_NORMAL\n" );
	}  
    }
	 
    cv::namedWindow("ColorImg", WinMode);
    cv::namedWindow("Saturation", WinMode);
	
    nameVec[0] = "baboon.jpg";
    nameVec[1] = "board.jpg";
    nameVec[2] = "building.jpg";
    nameVec[3] = "fruits.jpg";
    nameVec[4] = "lena.jpg";

    int cnt =0;
    for ( int i=0; i < 5 ; i++ )
    {
	img = cv::imread(nameVec[i]);
        if ( img.empty() ) {
            missing += nameVec[i] + "\n";
        } else {
            imgArray[cnt] = img;
            cnt++;
        }
    }

    // Log count of loaded images
    std::string msg = cv::format("%d/5 images loaded",cnt);
    
    // only QT:
    // cv::displayOverlay("ColorImg",msg, 4000);
    
    if ( cnt < 5 )
    {
        string info = "Sorry, missing images !\nPlease copy following images to executable dir:\n";
        info += missing;
		printf("------\n%s------", info.c_str() );
        cv::dispInfoBox( "ColorImg", "New Function Info", info );
        randomcode = 2;
        if ( cnt <= 0 ) return 0;
    }

    bool USE_hms   = false;
    int iKey = 999;
    int iSaveQ = 0;

    if ( randomcode == 2 )
    {
       setButtonBarContent("ColorImg", EMOD_Label, 3, "<font color=red>wrong FILENAMES</font>");
    } else {

        if ( WinMode == CV_WINDOW_AUTOSIZE ) 
	    setButtonBarContent("ColorImg", EMOD_Label, 3, "<font color=blue>AUTOSIZE</font>");

        if ( WinMode == CV_WINDOW_NORMAL ) 
	    setButtonBarContent("ColorImg", EMOD_Label, 3, "<font color=green>NORMAL</font>");
    }

    //--------------------------------------------
    std::vector<std::string> stringVec;

    // call with two parameters => fetch field content only
    //             cv::getCommandVec("ColorImg", stringVec )
    // call with three parameters => fetch field content + command 
    //             cv::getCommandVec("ColorImg", stringVec, csBuffer )
     
    if ( cv::getCommandVec("ColorImg", stringVec ) )
    {
        printf("\n\ninitial values:" );
        for ( unsigned int j=0; j < stringVec.size() ; j++ )
        {
            printf("\n  <%s>", stringVec[j].c_str() );
            // may be you have activated checkbox in *.cfg ....
            if ( stringVec[j] == "h:m:s|1"   ) USE_hms  = true;
        }
    }

    int iSatur = 50;
    if ( cv::getCommandVec("Saturation", stringVec ) )
    {
        for ( unsigned int j=0; j < stringVec.size() ; j++ )
        {
            int iposSat = stringVec[j].find("Sat");
            if ( iposSat >= 0 )
            {
                    iposSat = stringVec[j].find("|");
                    iSatur = atoi(&stringVec[j][iposSat+1]);
            }
        }
    }

    if ( initPosX < 0 )
    {
      // There is no definition inside *.cfg where to place "ColorImg"
      // so do an adjustment depending on screen resolution.....
      
      // move window to x1=25% of screen width on startup
      // window width=50% of screen is ignored in case of CV_WINDOW_AUTOSIZE
    
      cv::adjustWindowPos( "ColorImg",25,50, 0, 90 );
    }
    
    int idx = 0;

    
    //---------------------------  Start of processing loop
    while (iKey != 27)
    {
        char csBuffer[255];
        csBuffer[0] = 0;

	if ( iKey > 0 )
        {
	  cv::imshow("ColorImg", imgArray[idx]);
	  calcSaturation( imgArray[idx], iSatur );
	
	  printf("\n iKey = %d calcSaturation(iSatur=%d) done ....", iKey, iSatur );
        
	  string strPos = cv::format("%d/%d", idx+1, cnt );
	  UpdateContent( strPos, nameVec[idx] ); // set content of a label field in "ColorImg"

	  // let statusline contain filename for window "Saturation"
	  // $StatusLine is a part of *.cfg and should contain filename of course...
	  cv::setMapContent("Saturation", "filename", (char * ) nameVec[idx].c_str() );
        }

        iKey = cv::waitKey(5);

        //------------------------ some events from window "Saturation"  ?
        cv::getCommandVec("Saturation", stringVec, csBuffer );
        if ( strlen(csBuffer) > 0  )
        {  
           string strCmd = string(csBuffer);
           if (strCmd == "[Help]_EMOD_PushText")  iKey = 'h';
           if (strCmd == "-")  iKey = '-';
           if (strCmd == "+")  iKey = '+';
          
           // There is no direct message for a changed value of saturation by user interaction.
           // But the waitKey time loop does this job some milliseconds later   
           
	   for ( unsigned int j=0; j < stringVec.size() ; j++ )
           {
                int iposS = stringVec[j].find("Sat");
                if ( iposS >= 0 )
                {
                   iposS = stringVec[j].find("|");
                   iSatur = atoi(&stringVec[j][iposS+1]);						
                   calcSaturation( imgArray[idx], iSatur );
		   // Remember the word emit in the following line of *.cfg:
		   // "$SliderSpin Sat 0,100,35 emit"
		   // By this way we define "Sat_EMOD_Spin" as the keyword
		   // for triggering calcSaturation. Additional we define emit
		   // to set "Sat_EMOD_Spin" as a command.
		   
		   printf("\nj=%d [csBuffer=%s][%s] calcSaturation done ....",j,csBuffer, stringVec[j].c_str() );
                }
           }
        }

        //------------------------ some events from window "ColorImg"  ?
        string strCmd = "";
        cv::getCommandVec("ColorImg", stringVec, csBuffer );
        if ( strlen(csBuffer) > 0  )
        {
            printf("\ngetCommandVec(ColorImg)->[%s]\n  ", csBuffer );
            strCmd = string(csBuffer);
        }
        for ( unsigned int j=0; j < stringVec.size() ; j++ )
        {
           // printf("<%s>", stringVec[j].c_str() );
          if ( stringVec[j] == "h:m:s|1"  ) USE_hms  = true;
          if ( stringVec[j] == "h:m:s|0"  ) USE_hms  = false;

          // ------------- content of SpinField :
          int iposQ = stringVec[j].find("JPGQ");
          if ( iposQ >= 0 )
          {
	    iposQ = stringVec[j].find("|");
	    iSaveQ = atoi(&stringVec[j][iposQ+1]);
          }   
        }
    
        if (strCmd == "LoadImg") iKey = 'l';
        if (strCmd == "SaveImg") iKey = 's';
        if (strCmd == "PrevImg") iKey = 'p';
        if (strCmd == "NextImg") iKey = 'n';
        if (strCmd == "Help_EMOD_PushText")  iKey = 'h';
        if (strCmd == "Info")  iKey = 'i';
        if (strCmd == "h:m:s_EMOD_CheckBox")
        {
            printf("\n use_hms=%d ", (int) USE_hms );
        }
    
        if ( iKey == ' ') iKey = 'n';
        if ( iKey == 'p')
        {
            idx--;
            if (idx < 0) idx = 0;
            // only QT:
            // displayStatusBar("ColorImg","Loading previous image .....", 500 );
        }
        if ( iKey == 'n')
        {
            idx++;
            if (idx >= cnt) idx = 0;
            // only QT:
            // displayStatusBar("ColorImg","Loading next image .....", 500 );
        }

        if ( iKey == 's')
        {
            char szParam[64];
            sprintf(szParam,"_Q%02d", iSaveQ  );
            vector<int> ParamV;
            ParamV.resize(2);
            ParamV[0] = CV_IMWRITE_JPEG_QUALITY;
            ParamV[1] = iSaveQ;

            string filename = "SaveImg";
            if ( USE_hms ) {
                int tick =  (int) (0.001 * getTickCount()) ;
                sprintf(csBuffer,"%010d", tick);
                filename += string(csBuffer);
                filename += string(szParam) + ".jpg";
            } else {
                string savename = nameVec[idx];
                int iposDot = savename.find_last_of(".");
                filename = savename.substr(0,iposDot);
                filename += string(szParam) + ".jpg";
            }
            cv::imwrite(filename,imgArray[idx],ParamV);
            printf("\nfile [%s] written ", filename.c_str() );
        }

        
        if ( iKey == '-')
        {
	  printf("\ndo Zoom decrease ");
	}
        
        if ( iKey == '+')
        {
	  printf("\ndo Zoom increase ");
	}
        
        if ( iKey == 'h')
        {
            displayStatusBar("ColorImg","displayStatusBar without delay....", 0 );
            displayStatusBar("Saturation","displayStatusBar with 5000 msec ....", 5000 );

            printf("\n >>>> demo for use of buttons from *.cfg <<<<");
            string info = "Please open the *.cfg file fitting to the executable you are working on.\nThe HighGUI module has interpreted this *.cfg file. ";
            info += "To get more information what is internal done, just increment the value between    <verboseLevel></verboseLevel> \n";
            info += "Thus you will get more \nconsole output of the module's activities.";
            cv::dispInfoBox( "ColorImg", "New Function Info", info );

            info  = "All lines between <Wnd1></Wnd1> are for configuring the buttonbar and statusline of one window.\n";
	    info += "<Wnd2></Wnd2> is designed for the next window (and so on ...) \n";
	    info += "The first line inside <Wnd1></Wnd1> has to be the name of the window, and has to fit to the name in your source.\n";
	    info += "Between <Wnd3></Wnd3> you find a definition for an unused window, named \"Dummy\". The names $Zoom, $Panning, $SaveImg, $PropWnd, describe standard controls of  QtGui. ";
	    info += "All other controls are available from OpenCV2.4.4 on. \n";	    
	    info += "Each line generates one control (or a special group). All standard controls have icons, but do not deeper interact with the application\n";
	    
	    cv::dispInfoBox( "ColorImg", "New Qt Function Info", info );

	    info += "Each example application says to the HighGUI Module: \n   Give me the name of the last clicked button.\nIn your application sorce code you just call:\n";
            info += "     cvGetCommandVec(\"ColorImg\", stringVec, csBuffer );\n";
            info += "And csBuffer will be will filled with the last command.";
            cv::dispInfoBox( "ColorImg", "New Function Info", info );

            info = "If <LanguageTransTab> is defined, you can activate\n   $applyLanguage \nto get some button names translated.";
            info += "This makes sense because getCommandVec gives you the original name back.";
            cv::dispInfoBox( "ColorImg", "New Function Info", info );

            info =  "All buttons without frame (QToolButton) appear in the context menu too.\nFor this type of button only one word (without leading $) is necessary in *.cfg\n";
            cv::dispInfoBox( "ColorImg", "New Function Info", info );

            info = "There is no keyword 'emit' in the definition:\n  $Spin JPGQ 0,100,75 \nWe dont want to have any action until 'SaveImg' is pressed.";
            cv::dispInfoBox( "ColorImg", "New Function Info", info );

            char msg4[] = "If you see 'SaveImg' and '$SaveImg' in different lines of *.cfg \nkeep in mind '$SaveImg' to be the floppy disk icon-button  ";
            cv::dispInfoBox( "ColorImg", "New Function Info", msg4 );
            char msg5[] = "This box is internal displayed by\nQMessageBox::information( this, pCaption,pInfo )\nas part of the opencv_highgui24*.dll";
            cv::dispInfoBox( "ColorImg", "New Function Info", msg5 );
        }
        
        if ( iKey == 'i')
        {
	  displayStatusBar("ColorImg","displayStatusBar without delay....", 0 );
	  displayStatusBar("Saturation","displayStatusBar with 5000 msec ....", 5000 );
	}
        
    } // waitKey loop
    
    destroyAllWindows();	

    return 0;
}