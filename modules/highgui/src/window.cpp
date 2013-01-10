/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <map>
#include "opencv2/core/opengl_interop.hpp"


// in later times, use this file as a dispatcher to implementations like cvcap.cpp

CV_IMPL void cvSetWindowProperty(const char* name, int prop_id, double prop_value)
{
    switch(prop_id)
    {
    //change between fullscreen or not.
    case CV_WND_PROP_FULLSCREEN:

        if (!name || (prop_value!=CV_WINDOW_NORMAL && prop_value!=CV_WINDOW_FULLSCREEN))//bad argument
            break;

        #if defined (HAVE_QT)
            cvSetModeWindow_QT(name,prop_value);
        #elif defined WIN32 || defined _WIN32
            cvSetModeWindow_W32(name,prop_value);
        #elif defined (HAVE_GTK)
            cvSetModeWindow_GTK(name,prop_value);
        #elif defined (HAVE_CARBON)
            cvSetModeWindow_CARBON(name,prop_value);
        #elif defined (HAVE_COCOA)
            cvSetModeWindow_COCOA(name,prop_value);
        #endif
    break;

    case CV_WND_PROP_AUTOSIZE:
        #if defined (HAVE_QT)
            cvSetPropWindow_QT(name,prop_value);
        #endif
    break;

    case CV_WND_PROP_ASPECTRATIO:
        #if defined (HAVE_QT)
            cvSetRatioWindow_QT(name,prop_value);
        #endif
    break;

    default:;
    }
}

/* return -1 if error */
CV_IMPL double cvGetWindowProperty(const char* name, int prop_id)
{
    if (!name)
        return -1;

    switch(prop_id)
    {
    case CV_WND_PROP_FULLSCREEN:

        #if defined (HAVE_QT)
            return cvGetModeWindow_QT(name);
        #elif defined WIN32 || defined _WIN32
            return cvGetModeWindow_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetModeWindow_GTK(name);
        #elif defined (HAVE_CARBON)
            return cvGetModeWindow_CARBON(name);
        #elif defined (HAVE_COCOA)
            return cvGetModeWindow_COCOA(name);
        #else
            return -1;
        #endif
    break;

    case CV_WND_PROP_AUTOSIZE:

        #if defined (HAVE_QT)
            return cvGetPropWindow_QT(name);
        #elif defined WIN32 || defined _WIN32
            return cvGetPropWindowAutoSize_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetPropWindowAutoSize_GTK(name);
        #else
            return -1;
        #endif
    break;

    case CV_WND_PROP_ASPECTRATIO:

        #if defined (HAVE_QT)
            return cvGetRatioWindow_QT(name);
        #elif defined WIN32 || defined _WIN32
            return cvGetRatioWindow_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetRatioWindow_GTK(name);
        #else
            return -1;
        #endif
    break;

    case CV_WND_PROP_OPENGL:

        #if defined (HAVE_QT)
            return cvGetOpenGlProp_QT(name);
        #elif defined WIN32 || defined _WIN32
            return cvGetOpenGlProp_W32(name);
        #elif defined (HAVE_GTK)
            return cvGetOpenGlProp_GTK(name);
        #else
            return -1;
        #endif
    break;

    default:
        return -1;
    }
}

void cv::namedWindow( const string& winname, int flags )
{
    cvNamedWindow( winname.c_str(), flags );
}

void cv::destroyWindow( const string& winname )
{
    cvDestroyWindow( winname.c_str() );
}

void cv::destroyAllWindows()
{
    cvDestroyAllWindows();
}

void cv::resizeWindow( const string& winname, int width, int height )
{
    cvResizeWindow( winname.c_str(), width, height );
}

void cv::moveWindow( const string& winname, int x, int y )
{
    cvMoveWindow( winname.c_str(), x, y );
}

// -----------------------------------

void cv::adjustWindowPos( const string& winname, int xp, int xwp, int yp, int yhp )
{
   // set window pos+size with params in percentage of screen width/height
   #ifdef _WIN32
      int cx,cy;
      cx = GetSystemMetrics(SM_CXSCREEN);
      cy = GetSystemMetrics(SM_CYSCREEN);
      int    x = (int) (0.01 * ( xp * cx ));
      int    y = (int) (0.01 * ( yp * cy ));
      int neww = (int) (0.01 * (xwp * cx ));
      int newh = (int) (0.01 * (yhp * cy ));
      cvMoveWindow( winname.c_str(), x, y );
      cvResizeWindow( winname.c_str(),neww, newh );
   #else
       #if defined(HAVE_QT)
          cvAdjustWindowPos_QT( winname.c_str(),  xp, xwp, yp,  yhp );
       #endif
		(void) winname;
		(void) xp;
		(void) xwp;
		(void) yp;
		(void) yhp;
       // Dummy calls if Qt is not available:
      #if defined(HAVE_GTK)
		// TODO: 
		// cvAdjustWindowPos_GTK( winname.c_str(),  xp, xwp, yp,  yhp );
      #endif
   #endif
}


#if defined(HAVE_QT)

void cv::dispInfoBox( const string winname, const char* caption, const string& text ) 
{
    cvDispInfoBox_QT( winname.c_str(), caption, text.c_str() );
}

int cv::getButtonBarContent(const string winname, int idx, char * txt )
{
       return cvGetButtonBarContent( winname.c_str(), idx, txt );
}

int cv::setButtonBarContent(const string winname, int etype, int idx, const char * txt )
{
       return cvSetButtonBarContent( winname.c_str(), etype, idx,  txt );
}

int cv::setMapContent(const string winname, const string& varname, const char * text )
{
      return cvSetMapContent( winname.c_str(), varname.c_str(), text );
}

#else

// some dummy definitions in case of missing QT
void cv::dispInfoBox( const string , const char* , const string&  ) 
{
}

int cv::getButtonBarContent(const string , int , char * )
{
   return 0;
}

int cv::setButtonBarContent(const string , int , int , const char *  )
{
   return 0;
}

int cv::setMapContent(const string , const string& , const char *  )
{
   return 0;
}

#endif

// -----------------------------------

void cv::setWindowProperty(const string& winname, int prop_id, double prop_value)
{
    cvSetWindowProperty( winname.c_str(), prop_id, prop_value);
}

double cv::getWindowProperty(const string& winname, int prop_id)
{
    return cvGetWindowProperty(winname.c_str(), prop_id);
}

int cv::waitKey(int delay)
{
    return cvWaitKey(delay);
}

int cv::createTrackbar(const string& trackbarName, const string& winName,
                   int* value, int count, TrackbarCallback callback,
                   void* userdata)
{
    return cvCreateTrackbar2(trackbarName.c_str(), winName.c_str(),
                             value, count, callback, userdata);
}

void cv::setTrackbarPos( const string& trackbarName, const string& winName, int value )
{
    cvSetTrackbarPos(trackbarName.c_str(), winName.c_str(), value );
}

int cv::getTrackbarPos( const string& trackbarName, const string& winName )
{
    return cvGetTrackbarPos(trackbarName.c_str(), winName.c_str());
}

void cv::setMouseCallback( const string& windowName, MouseCallback onMouse, void* param)
{
    cvSetMouseCallback(windowName.c_str(), onMouse, param);
}

int cv::startWindowThread()
{
    return cvStartWindowThread();
}




int cv::readConfig( const char* file, const char * name, CvConfigBase * cfg  )
{
  // read some basic data from *.cfg but no controls here
  
  cfg->initWidth  = -1;
  cfg->initHeight = -1;
  cfg->initPosX   = -1;
  cfg->initPosY   = -1;
  cfg->WindowMode = -1;
  cfg->wndname = std::string(name);
  
  char csCfgFile[512];
  strcpy( csCfgFile, file); 
  char * p = strrchr( csCfgFile,'.');

  if ( p != NULL )
  {
	  *p = 0;
	  strcat( csCfgFile, ".cfg") ;
  } else {
	  // linux
	  strcat( csCfgFile, ".cfg") ;
  }
 
  // TODO: 
  // - use GetModuleFileName(NULL, szFilename, MAX_PATH) to get executable name
  //   in window_w32.cpp 
  // - use  QString exe_name = QFileInfo(QApplication::applicationFilePath()).fileName();
  //   in window_QT.cpp for the same purpose
  // - how to do it with GTK ? 
  //   g_get_prgname() or g_get_application_name() deliver the name of the window
  //   and not the name of the executable. Whats wrong ??
	
 
  cfg->fs.open(csCfgFile, cv::FileStorage::READ);
  if (!cfg->fs.isOpened())
  {
      return -1;
  }
    
  cv::FileNode n = cfg->fs["verboseLevel"];
  cv::FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
  for (; it != it_end; ++it) {
    cv::FileNode node = *it;
    if ( node.type() == CV_NODE_INTEGER )
    {
      cfg->m_verboseLevel = (int) *it;    
    }
  }        
  
  std::string CfgWndName = "";
  
  //---------------------------------------- cv::readConfig
  for ( int cnt=0; cnt <= 20 ; cnt++ )
  {
      std::string CfgWndN = cv::format("Wnd%d",cnt);
         
      n = cfg->fs[CfgWndN];
      if (  n.size() <= 0 ) continue;
	
      int linecnt = -1;
      it = n.begin(), it_end = n.end(); // Go through the node
      for (; it != it_end; ++it) {
	
	cv::FileNode node = *it;
	std::string content = "?";
	if ( node.type() == CV_NODE_INTEGER )
	{
	    content = cv::format("%d (integer)", (int) *it);
	}	
	if ( node.type() == cv::FileNode::STRING )
	{
	    content = (std::string) *it;
	    char csBuffer[512];
	    strcpy(csBuffer, content.c_str() );
	    if (csBuffer[0] == '#') content = "";
	    if ( content.length() > 0 )
	    {   
		linecnt++;
		if ( linecnt == 0 )
		{	
			CfgWndName = std::string(csBuffer);	
			if ( CfgWndName != cfg->wndname ) {
			  linecnt = -5000;
			} 
		}

		if ( linecnt > 0 )
		{
		
		  if ( cfg->m_verboseLevel > 1 )
		  {
		      printf("\n   [%s]", content.c_str() );
		  }  
	
		  // aus cmdparser.cpp kommt:
		  // vector<string> split_string(const string& str, const string& delimiters)
		  // vector<string> baseVec = split_string( strBase, " ");

	
		  if ( strstr(csBuffer,"CV_WINDOW_") != NULL )
		  { 
		    // may be there is a window position or size behind the window mode in *.cfg
		    std::string sizestr(csBuffer);		  
		    std::string strBase(csBuffer);		  
		    
		    SplitList baselist(sizestr," ");
  
		    if ( baselist.size() >= 2 )
		    {
		        std::string Mode = baselist[0];
			if ( Mode == "CV_WINDOW_NORMAL"   ) cfg->WindowMode = CV_WINDOW_NORMAL;
			if ( Mode == "CV_WINDOW_AUTOSIZE" ) cfg->WindowMode = CV_WINDOW_AUTOSIZE;
		
			SplitList poslist(baselist[1],",");
			if ( poslist.size() == 2 )
			{
			  cfg->initPosX = atoi( poslist[0].c_str() );
			  cfg->initPosY = atoi( poslist[1].c_str() );
			}
			if ( baselist.size() == 3 )
			{
			  SplitList sizelist(baselist[2],"*");
			  if ( sizelist.size() == 2 )
			  {
			    cfg->initWidth = atoi( sizelist[0].c_str() );
			    cfg->initHeight = atoi( sizelist[1].c_str() );
			  }
			}
		    }

		    if ( cfg->m_verboseLevel > 0 )
		    {
			  printf("\n%s:[%s]  %d,%d (%d*%d Pixel) WindowMode=%d  (cv::readConfig)", 
				CfgWndN.c_str(), cfg->wndname.c_str(), cfg->initPosX, cfg->initPosY , cfg->initWidth, cfg->initHeight, cfg->WindowMode );
		    }	    
		  }
		  	  
		}		
	    }
	}
		
      }
  }
     
  return 0;   
}




// OpenGL support

void cv::setOpenGlDrawCallback(const string& name, OpenGlDrawCallback callback, void* userdata)
{
    cvSetOpenGlDrawCallback(name.c_str(), callback, userdata);
}

void cv::setOpenGlContext(const string& windowName)
{
    cvSetOpenGlContext(windowName.c_str());
}

void cv::updateWindow(const string& windowName)
{
    cvUpdateWindow(windowName.c_str());
}

#ifdef HAVE_OPENGL
namespace
{
    std::map<std::string, cv::GlTexture2D> wndTexs;
    std::map<std::string, cv::GlTexture2D> ownWndTexs;
    std::map<std::string, cv::GlBuffer> ownWndBufs;



    void CV_CDECL glDrawTextureCallback(void* userdata)
    {
	cv::GlTexture2D* texObj = static_cast<cv::GlTexture2D*>(userdata);

        cv::render(*texObj);
    }

}
#endif // HAVE_OPENGL



void cv::imshow( const string& winname, InputArray _img )
{
#ifndef HAVE_OPENGL
    Mat img = _img.getMat();
    CvMat c_img = img;
    cvShowImage(winname.c_str(), &c_img);
#else
    const double useGl = getWindowProperty(winname, WND_PROP_OPENGL);

    if (useGl <= 0)
    {
        Mat img = _img.getMat();
        CvMat c_img = img;
        cvShowImage(winname.c_str(), &c_img);
    }
    else
    {
        const double autoSize = getWindowProperty(winname, WND_PROP_AUTOSIZE);

        if (autoSize > 0)
        {
            Size size = _img.size();
            resizeWindow(winname, size.width, size.height);
        }

        setOpenGlContext(winname);

        if (_img.kind() == _InputArray::OPENGL_TEXTURE2D)
        {
            cv::GlTexture2D& tex = wndTexs[winname];

            tex = _img.getGlTexture2D();

            tex.setAutoRelease(false);

            setOpenGlDrawCallback(winname, glDrawTextureCallback, &tex);
        }
        else
        {
            cv::GlTexture2D& tex = ownWndTexs[winname];

            if (_img.kind() == _InputArray::GPU_MAT)
            {
                cv::GlBuffer& buf = ownWndBufs[winname];
                buf.copyFrom(_img);
                buf.setAutoRelease(false);

                tex.copyFrom(buf);
                tex.setAutoRelease(false);
            }
            else
            {
                tex.copyFrom(_img);
            }

            tex.setAutoRelease(false);

            setOpenGlDrawCallback(winname, glDrawTextureCallback, &tex);
        }

        updateWindow(winname);
    }
#endif
}

// Without OpenGL

#ifndef HAVE_OPENGL

CV_IMPL void cvSetOpenGlDrawCallback(const char*, CvOpenGlDrawCallback, void*)
{
    CV_Error(CV_OpenGlNotSupported, "The library is compiled without OpenGL support");
}

CV_IMPL void cvSetOpenGlContext(const char*)
{
    CV_Error(CV_OpenGlNotSupported, "The library is compiled without OpenGL support");
}

CV_IMPL void cvUpdateWindow(const char*)
{
    CV_Error(CV_OpenGlNotSupported, "The library is compiled without OpenGL support");
}


#endif // !HAVE_OPENGL

#if defined (HAVE_QT)

CvFont cv::fontQt(const string& nameFont, int pointSize, Scalar color, int weight,  int style, int /*spacing*/)
{
return cvFontQt(nameFont.c_str(), pointSize,color,weight, style);
}

void cv::addText( const Mat& img, const string& text, Point org, CvFont font)
{
    CvMat _img = img;
    cvAddText( &_img, text.c_str(), org,&font);
}

void cv::displayStatusBar(const string& name,  const string& text, int delayms)
{
    cvDisplayStatusBar(name.c_str(),text.c_str(), delayms);
}

void cv::displayOverlay(const string& name,  const string& text, int delayms)
{
    cvDisplayOverlay(name.c_str(),text.c_str(), delayms);
}

int cv::startLoop(int (*pt2Func)(int argc, char *argv[]), int argc, char* argv[])
{
    return cvStartLoop(pt2Func, argc, argv);
}

void cv::stopLoop()
{
    cvStopLoop();
}

void cv::saveWindowParameters(const string& windowName)
{
    cvSaveWindowParameters(windowName.c_str());
}

void cv::loadWindowParameters(const string& windowName)
{
    cvLoadWindowParameters(windowName.c_str());
}

int cv::createButton(const string& button_name, ButtonCallback on_change, void* userdata, int button_type , bool initial_button_state  )
{
    return cvCreateButton(button_name.c_str(), on_change, userdata, button_type , initial_button_state );
}

//--------- Get events+content from buttonbar in case of HAVE_QT :
bool cv::getCommandVec( const string& winname, vector<string> & stringVec,  char * cmd )
{
	// All controls inside a buttonbar are configured by a *.cfg file
    stringVec.clear();  // clear content vector
  
	// read last command (e.g. pressed button ) and content of buttonbar    
    if ( cmd != NULL ) cvGetCommand( winname.c_str(), cmd ); 
	
    char buffer[512];
    int idx = 0;
    int iRet = 1;
    while ( iRet == 1 ) 
    {
      iRet = cvGetButtonBarContent( winname.c_str(), idx, buffer );
      if ( iRet == 1 ) stringVec.push_back( string(buffer) );
      idx++;
    }
    return true;
}
//---------



#else

CvFont cv::fontQt(const string&, int, Scalar, int,  int, int)
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
    return CvFont();
}

void cv::addText( const Mat&, const string&, Point, CvFont)
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
}

void cv::displayStatusBar(const string&,  const string&, int)
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
}

void cv::displayOverlay(const string&,  const string&, int )
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
}

int cv::startLoop(int (*)(int argc, char *argv[]), int , char**)
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
    return 0;
}

void cv::stopLoop()
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
}

void cv::saveWindowParameters(const string&)
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
}

void cv::loadWindowParameters(const string&)
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
}

int cv::createButton(const string&, ButtonCallback, void*, int , bool )
{
    CV_Error(CV_StsNotImplemented, "The library is compiled without QT support");
    return 0;
}

// Dummy function in case of Qt switched off,
// but application with Qt scpecific calls....
bool cv::getCommandVec( const string& , vector<string> & ,  char* )
{
    return false;
}



#endif

#if   defined WIN32 || defined _WIN32         // see window_w32.cpp
#elif defined (HAVE_GTK)      // see window_gtk.cpp
#elif defined (HAVE_COCOA)   // see window_carbon.cpp
#elif defined (HAVE_CARBON)
#elif defined (HAVE_QT) //YV see window_QT.cpp

#else

// No windowing system present at compile time ;-(
//
// We will build place holders that don't break the API but give an error
// at runtime. This way people can choose to replace an installed HighGUI
// version with a more capable one without a need to recompile dependent
// applications or libraries.


#define CV_NO_GUI_ERROR(funcname) \
    cvError( CV_StsError, funcname, \
    "The function is not implemented. " \
    "Rebuild the library with Windows, GTK+ 2.x or Carbon support. "\
    "If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script", \
    __FILE__, __LINE__ )


CV_IMPL int cvNamedWindow( const char*, int )
{
    CV_NO_GUI_ERROR("cvNamedWindow");
    return -1;
}

CV_IMPL void cvDestroyWindow( const char* )
{
    CV_NO_GUI_ERROR( "cvDestroyWindow" );
}

CV_IMPL void
cvDestroyAllWindows( void )
{
    CV_NO_GUI_ERROR( "cvDestroyAllWindows" );
}

CV_IMPL void
cvShowImage( const char*, const CvArr* )
{
    CV_NO_GUI_ERROR( "cvShowImage" );
}

CV_IMPL void cvResizeWindow( const char*, int, int )
{
    CV_NO_GUI_ERROR( "cvResizeWindow" );
}

CV_IMPL void cvMoveWindow( const char*, int, int )
{
    CV_NO_GUI_ERROR( "cvMoveWindow" );
}

CV_IMPL int
cvCreateTrackbar( const char*, const char*,
                  int*, int, CvTrackbarCallback )
{
    CV_NO_GUI_ERROR( "cvCreateTrackbar" );
    return -1;
}

CV_IMPL int
cvCreateTrackbar2( const char* /*trackbar_name*/, const char* /*window_name*/,
                   int* /*val*/, int /*count*/, CvTrackbarCallback2 /*on_notify2*/,
                   void* /*userdata*/ )
{
    CV_NO_GUI_ERROR( "cvCreateTrackbar2" );
    return -1;
}

CV_IMPL void
cvSetMouseCallback( const char*, CvMouseCallback, void* )
{
    CV_NO_GUI_ERROR( "cvSetMouseCallback" );
}

CV_IMPL int cvGetTrackbarPos( const char*, const char* )
{
    CV_NO_GUI_ERROR( "cvGetTrackbarPos" );
    return -1;
}

CV_IMPL void cvSetTrackbarPos( const char*, const char*, int )
{
    CV_NO_GUI_ERROR( "cvSetTrackbarPos" );
}

CV_IMPL void* cvGetWindowHandle( const char* )
{
    CV_NO_GUI_ERROR( "cvGetWindowHandle" );
    return 0;
}

CV_IMPL const char* cvGetWindowName( void* )
{
    CV_NO_GUI_ERROR( "cvGetWindowName" );
    return 0;
}

CV_IMPL int cvWaitKey( int )
{
    CV_NO_GUI_ERROR( "cvWaitKey" );
    return -1;
}

CV_IMPL int cvInitSystem( int , char** )
{

    CV_NO_GUI_ERROR( "cvInitSystem" );
    return -1;
}

CV_IMPL int cvStartWindowThread()
{

    CV_NO_GUI_ERROR( "cvStartWindowThread" );
    return -1;
}

//-------- Qt ---------
CV_IMPL void cvAddText( const CvArr*, const char*, CvPoint , CvFont* )
{
    CV_NO_GUI_ERROR("cvAddText");
}

CV_IMPL void cvDisplayStatusBar(const char* , const char* , int )
{
    CV_NO_GUI_ERROR("cvDisplayStatusBar");
}

CV_IMPL void cvDisplayOverlay(const char* , const char* , int )
{
    CV_NO_GUI_ERROR("cvNamedWindow");
}

CV_IMPL int cvStartLoop(int (*)(int argc, char *argv[]), int , char* argv[])
{
    (void)argv;
    CV_NO_GUI_ERROR("cvStartLoop");
    return -1;
}

CV_IMPL void cvStopLoop()
{
    CV_NO_GUI_ERROR("cvStopLoop");
}

CV_IMPL void cvSaveWindowParameters(const char* )
{
    CV_NO_GUI_ERROR("cvSaveWindowParameters");
}

// CV_IMPL void cvLoadWindowParameterss(const char* name)
// {
//     CV_NO_GUI_ERROR("cvLoadWindowParameters");
// }

CV_IMPL int cvCreateButton(const char*, void (*)(int, void*), void*, int, int)
{
    CV_NO_GUI_ERROR("cvCreateButton");
    return -1;
}

// Some dummy functions in case of Qt switched off,
// but application with Qt specific calls....

CV_IMPL int cvGetButtonBarContent(const char *, int, char * )
{
    // CV_NO_GUI_ERROR("cvGetButtonBarContent");
    return -1;
}

CV_IMPL int cvSetButtonBarContent(const char *, int, int, const char * )
{
    // CV_NO_GUI_ERROR("cvSetButtonBarContent");
    return -1;
}

CV_IMPL void cvDispInfoBox_QT(const char*, const char* , const char * )
{
    // CV_NO_GUI_ERROR("cvDispInfoBox_QT");
    return;
}



#endif

/* End of file. */
