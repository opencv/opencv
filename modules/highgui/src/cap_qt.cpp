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
#include "cv.h"

// Original implementation by   Mark Asbach
//                              Institute of Communications Engineering
//                              RWTH Aachen University
//
// For implementation details and background see:
// http://developer.apple.com/samplecode/qtframestepper.win/listing1.html
//
// Please note that timing will only be correct for videos that contain a visual track
// that has full length (compared to other tracks)


// standard includes
#include <cstdio>
#include <cassert>

// Mac OS includes
#include <Carbon/Carbon.h>
#include <CoreFoundation/CoreFoundation.h>
#include <QuickTime/QuickTime.h>


// Global state (did we call EnterMovies?)
static int did_enter_movies = 0;

// ----------------------------------------------------------------------------------------
#pragma mark Reading Video Files

/// Movie state structure for QuickTime movies
typedef struct CvCapture_QT_Movie
{
	Movie      myMovie;            // movie handle
	GWorldPtr  myGWorld;           // we render into an offscreen GWorld

	CvSize     size;               // dimensions of the movie
	TimeValue  movie_start_time;   // movies can start at arbitrary times
	long       number_of_frames;   // duration in frames
	long       next_frame_time;
	long       next_frame_number;

	IplImage * image_rgb;          // will point to the PixMap of myGWorld
	IplImage * image_bgr;          // will be returned by icvRetrieveFrame_QT()

} CvCapture_QT_Movie;


static       int         icvOpenFile_QT_Movie      (CvCapture_QT_Movie * capture, const char  * filename);
static       int         icvClose_QT_Movie         (CvCapture_QT_Movie * capture);
static       double      icvGetProperty_QT_Movie   (CvCapture_QT_Movie * capture, int property_id);
static       int         icvSetProperty_QT_Movie   (CvCapture_QT_Movie * capture, int property_id, double value);
static       int         icvGrabFrame_QT_Movie     (CvCapture_QT_Movie * capture);
static const void      * icvRetrieveFrame_QT_Movie (CvCapture_QT_Movie * capture, int);


static CvCapture_QT_Movie * icvCaptureFromFile_QT (const char * filename)
{
    static int did_enter_movies = 0;
	if (! did_enter_movies)
	{
		EnterMovies();
		did_enter_movies = 1;
	}

    CvCapture_QT_Movie * capture = 0;

    if (filename)
    {
        capture = (CvCapture_QT_Movie *) cvAlloc (sizeof (*capture));
        memset (capture, 0, sizeof(*capture));

        if (!icvOpenFile_QT_Movie (capture, filename))
            cvFree( &capture );
    }

    return capture;
}



/**
 * convert full path to CFStringRef and open corresponding Movie. Then
 * step over 'interesting frame times' to count total number of frames
 * for video material with varying frame durations and create offscreen
 * GWorld for rendering the movie frames.
 *
 * @author Mark Asbach <asbach@ient.rwth-aachen.de>
 * @date   2005-11-04
 */
static int icvOpenFile_QT_Movie (CvCapture_QT_Movie * capture, const char * filename)
{
	Rect          myRect;
	short         myResID        = 0;
	Handle        myDataRef      = nil;
	OSType        myDataRefType  = 0;
	OSErr         myErr          = noErr;


	// no old errors please
	ClearMoviesStickyError ();

	// initialize pointers to zero
	capture->myMovie  = 0;
	capture->myGWorld = nil;

	// initialize numbers with invalid values
	capture->next_frame_time   = -1;
	capture->next_frame_number = -1;
	capture->number_of_frames  = -1;
	capture->movie_start_time  = -1;
	capture->size              = cvSize (-1,-1);


	// we would use CFStringCreateWithFileSystemRepresentation (kCFAllocatorDefault, filename) on Mac OS X 10.4
	CFStringRef   inPath = CFStringCreateWithCString (kCFAllocatorDefault, filename, kCFStringEncodingISOLatin1);
	OPENCV_ASSERT ((inPath != nil), "icvOpenFile_QT_Movie", "couldnt create CFString from a string");

	// create the data reference
	myErr = QTNewDataReferenceFromFullPathCFString (inPath, kQTPOSIXPathStyle, 0, & myDataRef, & myDataRefType);
	if (myErr != noErr)
	{
		fprintf (stderr, "Couldn't create QTNewDataReferenceFromFullPathCFString().\n");
		return 0;
	}

	// get the Movie
	myErr = NewMovieFromDataRef(& capture->myMovie, newMovieActive | newMovieAsyncOK /* | newMovieIdleImportOK */,
								& myResID, myDataRef, myDataRefType);

	// dispose of the data reference handle - we no longer need it
	DisposeHandle (myDataRef);

	// if NewMovieFromDataRef failed, we already disposed the DataRef, so just return with an error
	if (myErr != noErr)
	{
		fprintf (stderr, "Couldn't create a NewMovieFromDataRef() - error is %d.\n",  myErr);
		return 0;
	}

	// count the number of video 'frames' in the movie by stepping through all of the
	// video 'interesting times', or in other words, the places where the movie displays
	// a new video sample. The time between these interesting times is not necessarily constant.
	{
		OSType      whichMediaType = VisualMediaCharacteristic;
		TimeValue   theTime        = -1;

		// find out movie start time
		GetMovieNextInterestingTime (capture->myMovie, short (nextTimeMediaSample + nextTimeEdgeOK),
		                             1, & whichMediaType, TimeValue (0), 0, & theTime, NULL);
		if (theTime == -1)
		{
			fprintf (stderr, "Couldn't inquire first frame time\n");
			return 0;
		}
		capture->movie_start_time  = theTime;
		capture->next_frame_time   = theTime;
		capture->next_frame_number = 0;

		// count all 'interesting times' of the movie
		capture->number_of_frames  = 0;
		while (theTime >= 0)
		{
			GetMovieNextInterestingTime (capture->myMovie, short (nextTimeMediaSample),
			                             1, & whichMediaType, theTime, 0, & theTime, NULL);
			capture->number_of_frames++;
		}
	}

	// get the bounding rectangle of the movie
	GetMoviesError ();
	GetMovieBox (capture->myMovie, & myRect);
	capture->size = cvSize (myRect.right - myRect.left, myRect.bottom - myRect.top);

	// create gworld for decompressed image
	myErr = QTNewGWorld (& capture->myGWorld, k32ARGBPixelFormat /* k24BGRPixelFormat geht leider nicht */,
	                     & myRect, nil, nil, 0);
	OPENCV_ASSERT (myErr == noErr, "icvOpenFile_QT_Movie", "couldnt create QTNewGWorld() for output image");
	SetMovieGWorld (capture->myMovie, capture->myGWorld, nil);

	// build IplImage header that will point to the PixMap of the Movie's GWorld later on
	capture->image_rgb = cvCreateImageHeader (capture->size, IPL_DEPTH_8U, 4);

	// create IplImage that hold correctly formatted result
	capture->image_bgr = cvCreateImage (capture->size, IPL_DEPTH_8U, 3);

	// okay, that's it - should we wait until the Movie is playable?
	return 1;
}

/**
 * dispose of QuickTime Movie and free memory buffers
 *
 * @author Mark Asbach <asbach@ient.rwth-aachen.de>
 * @date   2005-11-04
 */
static int icvClose_QT_Movie (CvCapture_QT_Movie * capture)
{
	OPENCV_ASSERT (capture,          "icvClose_QT_Movie", "'capture' is a NULL-pointer");

	// deallocate and free resources
	if (capture->myMovie)
	{
		cvReleaseImage       (& capture->image_bgr);
		cvReleaseImageHeader (& capture->image_rgb);
		DisposeGWorld        (capture->myGWorld);
		DisposeMovie         (capture->myMovie);
	}

	// okay, that's it
	return 1;
}

/**
 * get a capture property
 *
 * @author Mark Asbach <asbach@ient.rwth-aachen.de>
 * @date   2005-11-05
 */
static double icvGetProperty_QT_Movie (CvCapture_QT_Movie * capture, int property_id)
{
	OPENCV_ASSERT (capture,                        "icvGetProperty_QT_Movie", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (capture->myMovie,               "icvGetProperty_QT_Movie", "invalid Movie handle");
	OPENCV_ASSERT (capture->number_of_frames >  0, "icvGetProperty_QT_Movie", "movie has invalid number of frames");
	OPENCV_ASSERT (capture->movie_start_time >= 0, "icvGetProperty_QT_Movie", "movie has invalid start time");

    // inquire desired property
    switch (property_id)
    {
		case CV_CAP_PROP_POS_FRAMES:
			return (capture->next_frame_number);

		case CV_CAP_PROP_POS_MSEC:
		case CV_CAP_PROP_POS_AVI_RATIO:
			{
				TimeValue   position  = capture->next_frame_time - capture->movie_start_time;

				if (property_id == CV_CAP_PROP_POS_MSEC)
				{
					TimeScale   timescale = GetMovieTimeScale (capture->myMovie);
					return (static_cast<double> (position) * 1000.0 / timescale);
				}
				else
				{
					TimeValue   duration  = GetMovieDuration  (capture->myMovie);
					return (static_cast<double> (position) / duration);
				}
			}
			break; // never reached

		case CV_CAP_PROP_FRAME_WIDTH:
			return static_cast<double> (capture->size.width);

		case CV_CAP_PROP_FRAME_HEIGHT:
			return static_cast<double> (capture->size.height);

		case CV_CAP_PROP_FPS:
			{
				TimeValue   duration  = GetMovieDuration  (capture->myMovie);
				TimeScale   timescale = GetMovieTimeScale (capture->myMovie);

				return (capture->number_of_frames / (static_cast<double> (duration) / timescale));
			}

		case CV_CAP_PROP_FRAME_COUNT:
			return static_cast<double> (capture->number_of_frames);

		case CV_CAP_PROP_FOURCC:  // not implemented
		case CV_CAP_PROP_FORMAT:  // not implemented
		case CV_CAP_PROP_MODE:    // not implemented
		default:
			// unhandled or unknown capture property
			OPENCV_ERROR (CV_StsBadArg, "icvSetProperty_QT_Movie", "unknown or unhandled property_id");
			return CV_StsBadArg;
    }

    return 0;
}

/**
 * set a capture property. With movie files, it is only possible to set the
 * position (i.e. jump to a given time or frame number)
 *
 * @author Mark Asbach <asbach@ient.rwth-aachen.de>
 * @date   2005-11-05
 */
static int icvSetProperty_QT_Movie (CvCapture_QT_Movie * capture, int property_id, double value)
{
	OPENCV_ASSERT (capture,                        "icvSetProperty_QT_Movie", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (capture->myMovie,               "icvSetProperty_QT_Movie", "invalid Movie handle");
	OPENCV_ASSERT (capture->number_of_frames >  0, "icvSetProperty_QT_Movie", "movie has invalid number of frames");
	OPENCV_ASSERT (capture->movie_start_time >= 0, "icvSetProperty_QT_Movie", "movie has invalid start time");

    // inquire desired property
	//
	// rework these three points to really work through 'interesting times'.
	// with the current implementation, they result in wrong times or wrong frame numbers with content that
	// features varying frame durations
    switch (property_id)
    {
		case CV_CAP_PROP_POS_MSEC:
		case CV_CAP_PROP_POS_AVI_RATIO:
			{
				TimeValue    destination;
				OSType       myType     = VisualMediaCharacteristic;
				OSErr        myErr      = noErr;

				if (property_id == CV_CAP_PROP_POS_MSEC)
				{
					TimeScale  timescale   = GetMovieTimeScale      (capture->myMovie);
					           destination = static_cast<TimeValue> (value / 1000.0 * timescale + capture->movie_start_time);
				}
				else
				{
					TimeValue  duration    = GetMovieDuration       (capture->myMovie);
					           destination = static_cast<TimeValue> (value * duration + capture->movie_start_time);
				}

				// really seek?
				if (capture->next_frame_time == destination)
					break;

				// seek into which direction?
				if (capture->next_frame_time < destination)
				{
					while (capture->next_frame_time < destination)
					{
						capture->next_frame_number++;
						GetMovieNextInterestingTime (capture->myMovie, nextTimeStep, 1, & myType, capture->next_frame_time,
						                             1, & capture->next_frame_time, NULL);
						myErr = GetMoviesError();
						if (myErr != noErr)
						{
							fprintf (stderr, "Couldn't go on to GetMovieNextInterestingTime() in icvGrabFrame_QT.\n");
							return 0;
						}
					}
				}
				else
				{
					while (capture->next_frame_time > destination)
					{
						capture->next_frame_number--;
						GetMovieNextInterestingTime (capture->myMovie, nextTimeStep, 1, & myType, capture->next_frame_time,
						                             -1, & capture->next_frame_time, NULL);
						myErr = GetMoviesError();
						if (myErr != noErr)
						{
							fprintf (stderr, "Couldn't go back to GetMovieNextInterestingTime() in icvGrabFrame_QT.\n");
							return 0;
						}
					}
				}
			}
			break;

		case CV_CAP_PROP_POS_FRAMES:
			{
				TimeValue    destination = static_cast<TimeValue> (value);
				short        direction   = (destination > capture->next_frame_number) ? 1 : -1;
				OSType       myType      = VisualMediaCharacteristic;
				OSErr        myErr       = noErr;

				while (destination != capture->next_frame_number)
				{
					capture->next_frame_number += direction;
					GetMovieNextInterestingTime (capture->myMovie, nextTimeStep, 1, & myType, capture->next_frame_time,
												 direction, & capture->next_frame_time, NULL);
					myErr = GetMoviesError();
					if (myErr != noErr)
					{
						fprintf (stderr, "Couldn't step to desired frame number in icvGrabFrame_QT.\n");
						return 0;
					}
				}
			}
			break;

		default:
			// unhandled or unknown capture property
			OPENCV_ERROR (CV_StsBadArg, "icvSetProperty_QT_Movie", "unknown or unhandled property_id");
			return 0;
	}

	// positive result means success
	return 1;
}

/**
 * the original meaning of this method is to acquire raw frame data for the next video
 * frame but not decompress it. With the QuickTime video reader, this is reduced to
 * advance to the current frame time.
 *
 * @author Mark Asbach <asbach@ient.rwth-aachen.de>
 * @date   2005-11-06
 */
static int icvGrabFrame_QT_Movie (CvCapture_QT_Movie * capture)
{
	OPENCV_ASSERT (capture,          "icvGrabFrame_QT_Movie", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (capture->myMovie, "icvGrabFrame_QT_Movie", "invalid Movie handle");

	TimeValue    myCurrTime;
	OSType       myType     = VisualMediaCharacteristic;
	OSErr        myErr      = noErr;


	// jump to current video sample
	SetMovieTimeValue (capture->myMovie, capture->next_frame_time);
	myErr = GetMoviesError();
	if (myErr != noErr)
	{
		fprintf (stderr, "Couldn't SetMovieTimeValue() in icvGrabFrame_QT_Movie.\n");
		return  0;
	}

	// where are we now?
	myCurrTime = GetMovieTime (capture->myMovie, NULL);

	// increment counters
	capture->next_frame_number++;
	GetMovieNextInterestingTime (capture->myMovie, nextTimeStep, 1, & myType, myCurrTime, 1, & capture->next_frame_time, NULL);
	myErr = GetMoviesError();
	if (myErr != noErr)
	{
		fprintf (stderr, "Couldn't GetMovieNextInterestingTime() in icvGrabFrame_QT_Movie.\n");
		return 0;
	}

	// that's it
    return 1;
}

/**
 * render the current frame into an image buffer and convert to OpenCV IplImage
 * buffer layout (BGR sampling)
 *
 * @author Mark Asbach <asbach@ient.rwth-aachen.de>
 * @date   2005-11-06
 */
static const void * icvRetrieveFrame_QT_Movie (CvCapture_QT_Movie * capture, int)
{
	OPENCV_ASSERT (capture,            "icvRetrieveFrame_QT_Movie", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (capture->myMovie,   "icvRetrieveFrame_QT_Movie", "invalid Movie handle");
	OPENCV_ASSERT (capture->image_rgb, "icvRetrieveFrame_QT_Movie", "invalid source image");
	OPENCV_ASSERT (capture->image_bgr, "icvRetrieveFrame_QT_Movie", "invalid destination image");

	PixMapHandle  myPixMapHandle = nil;
	OSErr         myErr          = noErr;


	// invalidates the movie's display state so that the Movie Toolbox
	// redraws the movie the next time we call MoviesTask
	UpdateMovie (capture->myMovie);
	myErr = GetMoviesError ();
	if (myErr != noErr)
	{
		fprintf (stderr, "Couldn't UpdateMovie() in icvRetrieveFrame_QT_Movie().\n");
		return 0;
	}

	// service active movie (= redraw immediately)
	MoviesTask (capture->myMovie, 0L);
	myErr = GetMoviesError ();
	if (myErr != noErr)
	{
		fprintf (stderr, "MoviesTask() didn't succeed in icvRetrieveFrame_QT_Movie().\n");
		return 0;
	}

	// update IplImage header that points to PixMap of the Movie's GWorld.
	// unfortunately, cvCvtColor doesn't know ARGB, the QuickTime pixel format,
	// so we pass a modfied address.
	// ATTENTION: don't access the last pixel's alpha entry, it's inexistant
	myPixMapHandle = GetGWorldPixMap (capture->myGWorld);
	LockPixels (myPixMapHandle);
	cvSetData (capture->image_rgb, GetPixBaseAddr (myPixMapHandle) + 1, GetPixRowBytes (myPixMapHandle));

	// covert RGB of GWorld to BGR
	cvCvtColor (capture->image_rgb, capture->image_bgr, CV_RGBA2BGR);

	// allow QuickTime to access the buffer again
	UnlockPixels (myPixMapHandle);

    // always return the same image pointer
	return capture->image_bgr;
}


// ----------------------------------------------------------------------------------------
#pragma mark -
#pragma mark Capturing from Video Cameras

#ifdef USE_VDIG_VERSION

	/// SequenceGrabber state structure for QuickTime
	typedef struct CvCapture_QT_Cam_vdig
	{
		ComponentInstance  grabber;
		short              channel;
		GWorldPtr          myGWorld;
		PixMapHandle       pixmap;

		CvSize             size;
		long               number_of_frames;

		IplImage         * image_rgb; // will point to the PixMap of myGWorld
		IplImage         * image_bgr; // will be returned by icvRetrieveFrame_QT()

	} CvCapture_QT_Cam;

#else

	typedef struct CvCapture_QT_Cam_barg
	{
		SeqGrabComponent   grabber;
		SGChannel          channel;
		GWorldPtr          gworld;
		Rect               bounds;
		ImageSequence      sequence;

		volatile bool      got_frame;

		CvSize             size;
		IplImage         * image_rgb; // will point to the PixMap of myGWorld
		IplImage         * image_bgr; // will be returned by icvRetrieveFrame_QT()

	} CvCapture_QT_Cam;

#endif

static       int         icvOpenCamera_QT        (CvCapture_QT_Cam * capture, const int index);
static       int         icvClose_QT_Cam         (CvCapture_QT_Cam * capture);
static       double      icvGetProperty_QT_Cam   (CvCapture_QT_Cam * capture, int property_id);
static       int         icvSetProperty_QT_Cam   (CvCapture_QT_Cam * capture, int property_id, double value);
static       int         icvGrabFrame_QT_Cam     (CvCapture_QT_Cam * capture);
static const void      * icvRetrieveFrame_QT_Cam (CvCapture_QT_Cam * capture, int);


/**
 * Initialize memory structure and call method to open camera
 *
 * @author Mark Asbach <asbach@ient.rwth-aachen.de>
 * @date 2006-01-29
 */
static CvCapture_QT_Cam * icvCaptureFromCam_QT (const int index)
{
	if (! did_enter_movies)
	{
		EnterMovies();
		did_enter_movies = 1;
	}

    CvCapture_QT_Cam * capture = 0;

    if (index >= 0)
    {
        capture = (CvCapture_QT_Cam *) cvAlloc (sizeof (*capture));
        memset (capture, 0, sizeof(*capture));

        if (!icvOpenCamera_QT (capture, index))
            cvFree (&capture);
    }

    return capture;
}

/// capture properties currently unimplemented for QuickTime camera interface
static double icvGetProperty_QT_Cam (CvCapture_QT_Cam * capture, int property_id)
{
	assert (0);
	return 0;
}

/// capture properties currently unimplemented for QuickTime camera interface
static int icvSetProperty_QT_Cam (CvCapture_QT_Cam * capture, int property_id, double value)
{
	assert (0);
	return 0;
}

#ifdef USE_VDIG_VERSION
#pragma mark Capturing using VDIG

/**
 * Open a quicktime video grabber component. This could be an attached
 * IEEE1394 camera, a web cam, an iSight or digitizer card / video converter.
 *
 * @author Mark Asbach <asbach@ient.rwth-aachen.de>
 * @date 2006-01-29
 */
static int icvOpenCamera_QT (CvCapture_QT_Cam * capture, const int index)
{
	OPENCV_ASSERT (capture,            "icvOpenCamera_QT", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (index >=0, "icvOpenCamera_QT", "camera index is negative");

	ComponentDescription	component_description;
	Component				component = 0;
	int                     number_of_inputs = 0;
	Rect                    myRect;
	ComponentResult			result = noErr;


	// travers all components and count video digitizer channels
	component_description.componentType         = videoDigitizerComponentType;
	component_description.componentSubType      = 0L;
	component_description.componentManufacturer = 0L;
	component_description.componentFlags        = 0L;
	component_description.componentFlagsMask    = 0L;
	do
	{
		// traverse component list
		component = FindNextComponent (component, & component_description);

		// found a component?
		if (component)
		{
			// dump component name
			#ifndef NDEBUG
				ComponentDescription  desc;
				Handle                nameHandle = NewHandleClear (200);
				char                  nameBuffer [255];

				result = GetComponentInfo (component, & desc, nameHandle, nil, nil);
				OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt GetComponentInfo()");
				OPENCV_ASSERT (*nameHandle, "icvOpenCamera_QT", "No name returned by GetComponentInfo()");
				snprintf (nameBuffer, (**nameHandle) + 1, "%s", (char *) (* nameHandle + 1));
				printf ("- Videodevice: %s\n", nameBuffer);
				DisposeHandle (nameHandle);
			#endif

			// open component to count number of inputs
			capture->grabber = OpenComponent (component);
			if (capture->grabber)
			{
				result = VDGetNumberOfInputs (capture->grabber, & capture->channel);
				if (result != noErr)
					fprintf (stderr, "Couldnt GetNumberOfInputs: %d\n", (int) result);
				else
				{
					#ifndef NDEBUG
						printf ("  Number of inputs: %d\n", (int) capture->channel + 1);
					#endif

					// add to overall number of inputs
					number_of_inputs += capture->channel + 1;

					// did the user select an input that falls into this device's
					// range of inputs? Then leave the loop
					if (number_of_inputs > index)
					{
						// calculate relative channel index
						capture->channel = index - number_of_inputs + capture->channel + 1;
						OPENCV_ASSERT (capture->channel >= 0, "icvOpenCamera_QT", "negative channel number");

						// dump channel name
						#ifndef NDEBUG
							char  name[256];
							Str255  nameBuffer;

							result = VDGetInputName (capture->grabber, capture->channel, nameBuffer);
							OPENCV_ASSERT (result == noErr, "ictOpenCamera_QT", "couldnt GetInputName()");
							snprintf (name, *nameBuffer, "%s", (char *) (nameBuffer + 1));
							printf ("  Choosing input %d - %s\n", (int) capture->channel, name);
						#endif

						// leave the loop
						break;
					}
				}

				// obviously no inputs of this device/component were needed
				CloseComponent (capture->grabber);
			}
		}
	}
	while (component);

	// did we find the desired input?
	if (! component)
	{
		fprintf(stderr, "Not enough inputs available - can't choose input %d\n", index);
		return 0;
	}

	// -- Okay now, we selected the digitizer input, lets set up digitizer destination --

	ClearMoviesStickyError();

	// Select the desired input
	result = VDSetInput (capture->grabber, capture->channel);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt select video digitizer input");

	// get the bounding rectangle of the video digitizer
	result = VDGetActiveSrcRect (capture->grabber, capture->channel, & myRect);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt create VDGetActiveSrcRect from digitizer");
	myRect.right = 640; myRect.bottom = 480;
	capture->size = cvSize (myRect.right - myRect.left, myRect.bottom - myRect.top);
	printf ("Source rect is %d, %d -- %d, %d\n", (int) myRect.left, (int) myRect.top, (int) myRect.right, (int) myRect.bottom);

	// create offscreen GWorld
	result = QTNewGWorld (& capture->myGWorld, k32ARGBPixelFormat, & myRect, nil, nil, 0);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt create QTNewGWorld() for output image");

	// get pixmap
	capture->pixmap = GetGWorldPixMap (capture->myGWorld);
	result = GetMoviesError ();
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt get pixmap");

	// set digitizer rect
	result = VDSetDigitizerRect (capture->grabber, & myRect);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt create VDGetActiveSrcRect from digitizer");

	// set destination of digitized input
	result = VDSetPlayThruDestination (capture->grabber, capture->pixmap, & myRect, nil, nil);
	printf ("QuickTime error: %d\n", (int) result);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set video destination");

	// get destination of digitized images
	result = VDGetPlayThruDestination (capture->grabber, & capture->pixmap, nil, nil, nil);
	printf ("QuickTime error: %d\n", (int) result);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt get video destination");
	OPENCV_ASSERT (capture->pixmap != nil, "icvOpenCamera_QT", "empty set video destination");

	// get the bounding rectangle of the video digitizer
	GetPixBounds (capture->pixmap, & myRect);
	capture->size = cvSize (myRect.right - myRect.left, myRect.bottom - myRect.top);

	// build IplImage header that will point to the PixMap of the Movie's GWorld later on
	capture->image_rgb = cvCreateImageHeader (capture->size, IPL_DEPTH_8U, 4);
	OPENCV_ASSERT (capture->image_rgb, "icvOpenCamera_QT", "couldnt create image header");

	// create IplImage that hold correctly formatted result
	capture->image_bgr = cvCreateImage (capture->size, IPL_DEPTH_8U, 3);
	OPENCV_ASSERT (capture->image_bgr, "icvOpenCamera_QT", "couldnt create image");

	// notify digitizer component, that we well be starting grabbing soon
	result = VDCaptureStateChanging (capture->grabber, vdFlagCaptureIsForRecord | vdFlagCaptureStarting | vdFlagCaptureLowLatency);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set capture state");


	// yeah, we did it
	return 1;
}

static int icvClose_QT_Cam (CvCapture_QT_Cam * capture)
{
	OPENCV_ASSERT (capture, "icvClose_QT_Cam", "'capture' is a NULL-pointer");

	ComponentResult	result = noErr;

	// notify digitizer component, that we well be stopping grabbing soon
	result = VDCaptureStateChanging (capture->grabber, vdFlagCaptureStopping);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set capture state");

	// release memory
	cvReleaseImage       (& capture->image_bgr);
	cvReleaseImageHeader (& capture->image_rgb);
	DisposeGWorld        (capture->myGWorld);
	CloseComponent       (capture->grabber);

	// sucessful
	return 1;
}

static int icvGrabFrame_QT_Cam (CvCapture_QT_Cam * capture)
{
	OPENCV_ASSERT (capture,          "icvGrabFrame_QT_Cam", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (capture->grabber, "icvGrabFrame_QT_Cam", "'grabber' is a NULL-pointer");

	ComponentResult	result = noErr;

	// grab one frame
	result = VDGrabOneFrame (capture->grabber);
	if (result != noErr)
	{
		fprintf (stderr, "VDGrabOneFrame failed\n");
		return 0;
	}

	// successful
	return 1;
}

static const void * icvRetrieveFrame_QT_Cam (CvCapture_QT_Cam * capture, int)
{
	OPENCV_ASSERT (capture, "icvRetrieveFrame_QT_Cam", "'capture' is a NULL-pointer");

	PixMapHandle  myPixMapHandle = nil;

	// update IplImage header that points to PixMap of the Movie's GWorld.
	// unfortunately, cvCvtColor doesn't know ARGB, the QuickTime pixel format,
	// so we pass a modfied address.
	// ATTENTION: don't access the last pixel's alpha entry, it's inexistant
	//myPixMapHandle = GetGWorldPixMap (capture->myGWorld);
	myPixMapHandle = capture->pixmap;
	LockPixels (myPixMapHandle);
	cvSetData (capture->image_rgb, GetPixBaseAddr (myPixMapHandle) + 1, GetPixRowBytes (myPixMapHandle));

	// covert RGB of GWorld to BGR
	cvCvtColor (capture->image_rgb, capture->image_bgr, CV_RGBA2BGR);

	// allow QuickTime to access the buffer again
	UnlockPixels (myPixMapHandle);

    // always return the same image pointer
	return capture->image_bgr;
}

#else
#pragma mark Capturing using Sequence Grabber

static OSErr icvDataProc_QT_Cam (SGChannel channel, Ptr raw_data, long len, long *, long, TimeValue, short, long refCon)
{
	CvCapture_QT_Cam  * capture = (CvCapture_QT_Cam *) refCon;
	CodecFlags          ignore;
	ComponentResult     err     = noErr;


	// we need valid pointers
	OPENCV_ASSERT (capture,          "icvDataProc_QT_Cam", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (capture->gworld,  "icvDataProc_QT_Cam", "'gworld' is a NULL-pointer");
	OPENCV_ASSERT (raw_data,         "icvDataProc_QT_Cam", "'raw_data' is a NULL-pointer");

	// create a decompression sequence the first time
	if (capture->sequence == 0)
	{
		ImageDescriptionHandle   description = (ImageDescriptionHandle) NewHandle(0);
		
		// we need a decompression sequence that fits the raw data coming from the camera
		err = SGGetChannelSampleDescription (channel, (Handle) description);
		OPENCV_ASSERT (err == noErr, "icvDataProc_QT_Cam", "couldnt get channel sample description");
		
		//*************************************************************************************//
		//This fixed a bug when Quicktime is called twice to grab a frame (black band bug) - Yannick Verdie 2010
		Rect sourceRect;
		sourceRect.top = 0;
		sourceRect.left = 0;	
		sourceRect.right = (**description).width;
		sourceRect.bottom = (**description).height;
		
		MatrixRecord scaleMatrix;
		RectMatrix(&scaleMatrix,&sourceRect,&capture->bounds);
		
		err = DecompressSequenceBegin (&capture->sequence, description, capture->gworld, 0,&capture->bounds,&scaleMatrix, srcCopy, NULL, 0, codecNormalQuality, bestSpeedCodec);
		//**************************************************************************************//
		
		OPENCV_ASSERT (err == noErr, "icvDataProc_QT_Cam", "couldnt begin decompression sequence");
		DisposeHandle ((Handle) description);
	}

	// okay, we have a decompression sequence -> decompress!
	err = DecompressSequenceFrameS (capture->sequence, raw_data, len, 0, &ignore, nil);
	if (err != noErr)
	{
		fprintf (stderr, "icvDataProc_QT_Cam: couldn't decompress frame - %d\n", (int) err);
		return err;
	}

	// check if we dropped a frame
	/*#ifndef NDEBUG
		if (capture->got_frame)
			fprintf (stderr, "icvDataProc_QT_Cam: frame was dropped\n");
	#endif*/

	// everything worked as expected
	capture->got_frame = true;
	return noErr;
}


static int icvOpenCamera_QT (CvCapture_QT_Cam * capture, const int index)
{
	OPENCV_ASSERT (capture,    "icvOpenCamera_QT", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (index >= 0, "icvOpenCamera_QT", "camera index is negative");

	PixMapHandle  pixmap       = nil;
	OSErr         result       = noErr;

	// open sequence grabber component
	capture->grabber = OpenDefaultComponent (SeqGrabComponentType, 0);
	OPENCV_ASSERT (capture->grabber, "icvOpenCamera_QT", "couldnt create image");

	// initialize sequence grabber component
	result = SGInitialize (capture->grabber);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt initialize sequence grabber");
	result = SGSetDataRef (capture->grabber, 0, 0, seqGrabDontMakeMovie);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set data reference of sequence grabber");

	// set up video channel
	result = SGNewChannel (capture->grabber, VideoMediaType, & (capture->channel));
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt create new video channel");

    // select the camera indicated by index
    SGDeviceList device_list = 0;
    result = SGGetChannelDeviceList (capture->channel, 0, & device_list);
    OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt get channel device list");
    for (int i = 0, current_index = 1; i < (*device_list)->count; i++)
    {
        SGDeviceName device = (*device_list)->entry[i];
        if (device.flags == 0)
        {
            if (current_index == index)
            {
                result = SGSetChannelDevice (capture->channel, device.name);
                OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set the channel video device");
                break;
            }
            current_index++;
        }
    }   
    result = SGDisposeDeviceList (capture->grabber, device_list);
    OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt dispose the channel device list");
    
	// query natural camera resolution -- this will be wrong, but will be an upper
	// bound on the actual resolution -- the actual resolution is set below
	// after starting the frame grabber
	result = SGGetSrcVideoBounds (capture->channel, & (capture->bounds));
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set video channel bounds");

	// create offscreen GWorld
	result = QTNewGWorld (& (capture->gworld), k32ARGBPixelFormat, & (capture->bounds), 0, 0, 0);
	result = SGSetGWorld (capture->grabber, capture->gworld, 0);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set GWorld for sequence grabber");
	result = SGSetChannelBounds (capture->channel, & (capture->bounds));
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set video channel bounds");
	result = SGSetChannelUsage (capture->channel, seqGrabRecord);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set channel usage");

    // start recording so we can size
	result = SGStartRecord (capture->grabber);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt start recording");

	// don't know *actual* resolution until now
	ImageDescriptionHandle imageDesc = (ImageDescriptionHandle)NewHandle(0);
	result = SGGetChannelSampleDescription(capture->channel, (Handle)imageDesc);
	OPENCV_ASSERT( result == noErr, "icvOpenCamera_QT", "couldn't get image size");
	capture->bounds.right = (**imageDesc).width;
	capture->bounds.bottom = (**imageDesc).height;
	DisposeHandle ((Handle) imageDesc);

	// stop grabber so that we can reset the parameters to the right size
	result = SGStop (capture->grabber);
	OPENCV_ASSERT (result == noErr, "icveClose_QT_Cam", "couldnt stop recording");

	// reset GWorld to correct image size
	GWorldPtr tmpgworld;
	result = QTNewGWorld( &tmpgworld, k32ARGBPixelFormat, &(capture->bounds), 0, 0, 0);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt create offscreen GWorld");
	result = SGSetGWorld( capture->grabber, tmpgworld, 0);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set GWorld for sequence grabber");
	DisposeGWorld( capture->gworld );
	capture->gworld = tmpgworld;

	result = SGSetChannelBounds (capture->channel, & (capture->bounds));
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set video channel bounds");

	// allocate images
	capture->size = cvSize (capture->bounds.right - capture->bounds.left, capture->bounds.bottom - capture->bounds.top);

	// build IplImage header that points to the PixMap of the Movie's GWorld.
	// unfortunately, cvCvtColor doesn't know ARGB, the QuickTime pixel format,
	// so we shift the base address by one byte.
	// ATTENTION: don't access the last pixel's alpha entry, it's inexistant
	capture->image_rgb = cvCreateImageHeader (capture->size, IPL_DEPTH_8U, 4);
	OPENCV_ASSERT (capture->image_rgb, "icvOpenCamera_QT", "couldnt create image header");
	pixmap = GetGWorldPixMap (capture->gworld);
	OPENCV_ASSERT (pixmap, "icvOpenCamera_QT", "didn't get GWorld PixMap handle");
	LockPixels (pixmap);
	cvSetData (capture->image_rgb, GetPixBaseAddr (pixmap) + 1, GetPixRowBytes (pixmap));

	// create IplImage that hold correctly formatted result
	capture->image_bgr = cvCreateImage (capture->size, IPL_DEPTH_8U, 3);
	OPENCV_ASSERT (capture->image_bgr, "icvOpenCamera_QT", "couldnt create image");


	// tell the sequence grabber to invoke our data proc
	result = SGSetDataProc (capture->grabber, NewSGDataUPP (icvDataProc_QT_Cam), (long) capture);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt set data proc");

	// start recording
	result = SGStartRecord (capture->grabber);
	OPENCV_ASSERT (result == noErr, "icvOpenCamera_QT", "couldnt start recording");

	return 1;
}


static int icvClose_QT_Cam (CvCapture_QT_Cam * capture)
{
	OPENCV_ASSERT (capture, "icvClose_QT_Cam", "'capture' is a NULL-pointer");

	OSErr  result = noErr;


	// stop recording
	result = SGStop (capture->grabber);
	OPENCV_ASSERT (result == noErr, "icveClose_QT_Cam", "couldnt stop recording");

	// close sequence grabber component
	result = CloseComponent (capture->grabber);
	OPENCV_ASSERT (result == noErr, "icveClose_QT_Cam", "couldnt close sequence grabber component");

	// end decompression sequence
	CDSequenceEnd (capture->sequence);

	// free memory
	cvReleaseImage (& capture->image_bgr);
	cvReleaseImageHeader (& capture->image_rgb);
	DisposeGWorld (capture->gworld);

	// sucessful
	return 1;
}

static int icvGrabFrame_QT_Cam (CvCapture_QT_Cam * capture)
{
	OPENCV_ASSERT (capture,          "icvGrabFrame_QT_Cam", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (capture->grabber, "icvGrabFrame_QT_Cam", "'grabber' is a NULL-pointer");

	ComponentResult	result = noErr;


	// grab one frame
	result = SGIdle (capture->grabber);
	if (result != noErr)
	{
		fprintf (stderr, "SGIdle failed in icvGrabFrame_QT_Cam with error %d\n", (int) result);
		return 0;
	}

	// successful
	return 1;
}

static const void * icvRetrieveFrame_QT_Cam (CvCapture_QT_Cam * capture, int)
{
	OPENCV_ASSERT (capture,            "icvRetrieveFrame_QT_Cam", "'capture' is a NULL-pointer");
	OPENCV_ASSERT (capture->image_rgb, "icvRetrieveFrame_QT_Cam", "invalid source image");
	OPENCV_ASSERT (capture->image_bgr, "icvRetrieveFrame_QT_Cam", "invalid destination image");

	OSErr         myErr          = noErr;


	// service active sequence grabbers (= redraw immediately)
	while (! capture->got_frame)
	{
		myErr = SGIdle (capture->grabber);
		if (myErr != noErr)
		{
			fprintf (stderr, "SGIdle() didn't succeed in icvRetrieveFrame_QT_Cam().\n");
			return 0;
		}
	}

	// covert RGB of GWorld to BGR
	cvCvtColor (capture->image_rgb, capture->image_bgr, CV_RGBA2BGR);

	// reset grabbing status
	capture->got_frame = false;

    // always return the same image pointer
	return capture->image_bgr;
}

#endif


typedef struct CvVideoWriter_QT {

    DataHandler data_handler;
    Movie movie;
    Track track;
    Media video;

    ICMCompressionSessionRef compression_session_ref;

    TimeValue duration_per_sample;
} CvVideoWriter_QT;


static TimeScale const TIME_SCALE = 600;

static OSStatus icvEncodedFrameOutputCallback(
    void* writer,
    ICMCompressionSessionRef compression_session_ref,
    OSStatus error,
    ICMEncodedFrameRef encoded_frame_ref,
    void* reserved
);

static void icvSourceTrackingCallback(
    void *source_tracking_ref_con,
    ICMSourceTrackingFlags source_tracking_flags,
    void *source_frame_ref_con,
    void *reserved
);

static int icvWriteFrame_QT(
    CvVideoWriter_QT * video_writer,
    const IplImage * image
) {
    CVPixelBufferRef pixel_buffer_ref = NULL;
    CVReturn retval =
        CVPixelBufferCreate(
            kCFAllocatorDefault,
            image->width, image->height, k24RGBPixelFormat,
            NULL /* pixel_buffer_attributes */,
            &pixel_buffer_ref
        );

    // convert BGR IPL image to RGB pixel buffer
    IplImage* image_rgb =
        cvCreateImageHeader(
            cvSize( image->width, image->height ),
            IPL_DEPTH_8U,
            3
        );

    retval = CVPixelBufferLockBaseAddress( pixel_buffer_ref, 0 );

    void* base_address = CVPixelBufferGetBaseAddress( pixel_buffer_ref );
    size_t bytes_per_row = CVPixelBufferGetBytesPerRow( pixel_buffer_ref );
    cvSetData( image_rgb, base_address, bytes_per_row );

    cvConvertImage( image, image_rgb, CV_CVTIMG_SWAP_RB );

    retval = CVPixelBufferUnlockBaseAddress( pixel_buffer_ref, 0 );

    cvReleaseImageHeader( &image_rgb );

    ICMSourceTrackingCallbackRecord source_tracking_callback_record;
    source_tracking_callback_record.sourceTrackingCallback =
        icvSourceTrackingCallback;
    source_tracking_callback_record.sourceTrackingRefCon = NULL;

    OSStatus status =
        ICMCompressionSessionEncodeFrame(
            video_writer->compression_session_ref,
            pixel_buffer_ref,
            0,
            video_writer->duration_per_sample,
            kICMValidTime_DisplayDurationIsValid,
            NULL,
            &source_tracking_callback_record,
            static_cast<void*>( &pixel_buffer_ref )
        );

    return 0;
}

static void icvReleaseVideoWriter_QT( CvVideoWriter_QT ** writer ) {
    if ( ( writer != NULL ) && ( *writer != NULL ) ) {
        CvVideoWriter_QT* video_writer = *writer;

        // force compression session to complete encoding of outstanding source
        // frames
        ICMCompressionSessionCompleteFrames(
            video_writer->compression_session_ref, TRUE, 0, 0
        );

        EndMediaEdits( video_writer->video );

        ICMCompressionSessionRelease( video_writer->compression_session_ref );

        InsertMediaIntoTrack(
            video_writer->track,
            0,
            0,
            GetMediaDuration( video_writer->video ),
            FixRatio( 1, 1 )
        );

        UpdateMovieInStorage( video_writer->movie, video_writer->data_handler );

        CloseMovieStorage( video_writer->data_handler );

/*
        // export to AVI
        Handle data_ref;
        OSType data_ref_type;
        QTNewDataReferenceFromFullPathCFString(
            CFSTR( "/Users/seibert/Desktop/test.avi" ), kQTPOSIXPathStyle, 0,
            &data_ref, &data_ref_type
        );

        ConvertMovieToDataRef( video_writer->movie, NULL, data_ref,
            data_ref_type, kQTFileTypeAVI, 'TVOD', 0, NULL );

        DisposeHandle( data_ref );
*/

        DisposeMovie( video_writer->movie );

        cvFree( writer );
    }
}

static OSStatus icvEncodedFrameOutputCallback(
    void* writer,
    ICMCompressionSessionRef compression_session_ref,
    OSStatus error,
    ICMEncodedFrameRef encoded_frame_ref,
    void* reserved
) {
    CvVideoWriter_QT* video_writer = static_cast<CvVideoWriter_QT*>( writer );

    OSStatus err = AddMediaSampleFromEncodedFrame( video_writer->video,
        encoded_frame_ref, NULL );

    return err;
}

static void icvSourceTrackingCallback(
    void *source_tracking_ref_con,
    ICMSourceTrackingFlags source_tracking_flags,
    void *source_frame_ref_con,
    void *reserved
) {
    if ( source_tracking_flags & kICMSourceTracking_ReleasedPixelBuffer ) {
        CVPixelBufferRelease(
            *static_cast<CVPixelBufferRef*>( source_frame_ref_con )
        );
    }
}


static CvVideoWriter_QT* icvCreateVideoWriter_QT(
    const char * filename,
    int fourcc,
    double fps,
    CvSize frame_size,
    int is_color
) {
    CV_FUNCNAME( "icvCreateVideoWriter" );

    CvVideoWriter_QT* video_writer =
        static_cast<CvVideoWriter_QT*>( cvAlloc( sizeof( CvVideoWriter_QT ) ) );
    memset( video_writer, 0, sizeof( CvVideoWriter_QT ) );

    Handle                            data_ref     = NULL;
    OSType                            data_ref_type;
    DataHandler                       data_handler = NULL;
    Movie                             movie        = NULL;
    ICMCompressionSessionOptionsRef   options_ref  = NULL;
    ICMCompressionSessionRef          compression_session_ref = NULL;
    CFStringRef                       out_path     = nil;
    Track                             video_track  = nil;
    Media                             video        = nil;
    OSErr                             err          = noErr;

    __BEGIN__

    // validate input arguments
    if ( filename == NULL ) {
        CV_ERROR( CV_StsBadArg, "Video file name must not be NULL" );
    }
    if ( fps <= 0.0 ) {
        CV_ERROR( CV_StsBadArg, "FPS must be larger than 0.0" );
    }
    if ( ( frame_size.width <= 0 ) || ( frame_size.height <= 0 ) ) {
        CV_ERROR( CV_StsBadArg,
            "Frame width and height must be larger than 0" );
    }

    // initialize QuickTime
    if ( !did_enter_movies ) {
        err = EnterMovies();
        if ( err != noErr ) {
            CV_ERROR( CV_StsInternal, "Unable to initialize QuickTime" );
        }
        did_enter_movies = 1;
    }

    // convert the file name into a data reference
    out_path = CFStringCreateWithCString( kCFAllocatorDefault, filename, kCFStringEncodingISOLatin1 );
    CV_ASSERT( out_path != nil );
    err = QTNewDataReferenceFromFullPathCFString( out_path, kQTPOSIXPathStyle,
        0, &data_ref, &data_ref_type );
    CFRelease( out_path );
    if ( err != noErr ) {
        CV_ERROR( CV_StsInternal,
            "Cannot create data reference from file name" );
    }

    // create a new movie on disk
    err = CreateMovieStorage( data_ref, data_ref_type, 'TVOD',
        smCurrentScript, newMovieActive, &data_handler, &movie );

    if ( err != noErr ) {
        CV_ERROR( CV_StsInternal, "Cannot create movie storage" );
    }

    // create a track with video
    video_track = NewMovieTrack (movie,
            FixRatio( frame_size.width, 1 ),
            FixRatio( frame_size.height, 1 ),
            kNoVolume);
    err = GetMoviesError();
    if ( err != noErr ) {
        CV_ERROR( CV_StsInternal, "Cannot create video track" );
    }
    video = NewTrackMedia( video_track, VideoMediaType, TIME_SCALE, nil, 0 );
    err = GetMoviesError();
    if ( err != noErr ) {
        CV_ERROR( CV_StsInternal, "Cannot create video media" );
    }

    CodecType codecType;
    switch ( fourcc ) {
        case CV_FOURCC( 'D', 'I', 'B', ' ' ):
            codecType = kRawCodecType;
            break;
        default:
            codecType = kRawCodecType;
            break;
    }

    // start a compression session
    err = ICMCompressionSessionOptionsCreate( kCFAllocatorDefault,
        &options_ref );
    if ( err != noErr ) {
        CV_ERROR( CV_StsInternal, "Cannot create compression session options" );
    }
    err = ICMCompressionSessionOptionsSetAllowTemporalCompression( options_ref,
        true );
    if ( err != noErr) {
        CV_ERROR( CV_StsInternal, "Cannot enable temporal compression" );
    }
    err = ICMCompressionSessionOptionsSetAllowFrameReordering( options_ref,
        true );
    if ( err != noErr) {
        CV_ERROR( CV_StsInternal, "Cannot enable frame reordering" );
    }

    ICMEncodedFrameOutputRecord encoded_frame_output_record;
    encoded_frame_output_record.encodedFrameOutputCallback =
        icvEncodedFrameOutputCallback;
    encoded_frame_output_record.encodedFrameOutputRefCon =
        static_cast<void*>( video_writer );
    encoded_frame_output_record.frameDataAllocator = NULL;

    err = ICMCompressionSessionCreate( kCFAllocatorDefault, frame_size.width,
        frame_size.height, codecType, TIME_SCALE, options_ref,
        NULL /*source_pixel_buffer_attributes*/, &encoded_frame_output_record,
        &compression_session_ref );
    ICMCompressionSessionOptionsRelease( options_ref );
    if ( err != noErr ) {
        CV_ERROR( CV_StsInternal, "Cannot create compression session" );
    }

    err = BeginMediaEdits( video );
    if ( err != noErr ) {
        CV_ERROR( CV_StsInternal, "Cannot begin media edits" );
    }

    // fill in the video writer structure
    video_writer->data_handler = data_handler;
    video_writer->movie = movie;
    video_writer->track = video_track;
    video_writer->video = video;
    video_writer->compression_session_ref = compression_session_ref;
    video_writer->duration_per_sample =
        static_cast<TimeValue>( static_cast<double>( TIME_SCALE ) / fps );

    __END__

    // clean up in case of error (unless error processing mode is
    // CV_ErrModeLeaf)
    if ( err != noErr ) {
        if ( options_ref != NULL ) {
            ICMCompressionSessionOptionsRelease( options_ref );
        }
        if ( compression_session_ref != NULL ) {
            ICMCompressionSessionRelease( compression_session_ref );
        }
        if ( data_handler != NULL ) {
            CloseMovieStorage( data_handler );
        }
        if ( movie != NULL ) {
            DisposeMovie( movie );
        }
        if ( data_ref != NULL ) {
            DeleteMovieStorage( data_ref, data_ref_type );
            DisposeHandle( data_ref );
        }
        cvFree( reinterpret_cast<void**>( &video_writer ) );
        video_writer = NULL;
    }

    return video_writer;
}


/**
*
*   Wrappers for the new C++ CvCapture & CvVideoWriter structures
*
*/

class CvCapture_QT_Movie_CPP : public CvCapture
{
public:
    CvCapture_QT_Movie_CPP() { captureQT = 0; }
    virtual ~CvCapture_QT_Movie_CPP() { close(); }

    virtual bool open( const char* filename );
    virtual void close();

    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
	virtual int getCaptureDomain() { return CV_CAP_QT; } // Return the type of the capture object: CV_CAP_VFW, etc...
protected:

    CvCapture_QT_Movie* captureQT;
};

bool CvCapture_QT_Movie_CPP::open( const char* filename )
{
    close();
    captureQT = icvCaptureFromFile_QT( filename );
    return captureQT != 0;
}

void CvCapture_QT_Movie_CPP::close()
{
    if( captureQT )
    {
        icvClose_QT_Movie( captureQT );
        cvFree( &captureQT );
    }
}

bool CvCapture_QT_Movie_CPP::grabFrame()
{
    return captureQT ? icvGrabFrame_QT_Movie( captureQT ) != 0 : false;
}

IplImage* CvCapture_QT_Movie_CPP::retrieveFrame(int)
{
    return captureQT ? (IplImage*)icvRetrieveFrame_QT_Movie( captureQT, 0 ) : 0;
}

double CvCapture_QT_Movie_CPP::getProperty( int propId )
{
    return captureQT ? icvGetProperty_QT_Movie( captureQT, propId ) : 0;
}

bool CvCapture_QT_Movie_CPP::setProperty( int propId, double value )
{
    return captureQT ? icvSetProperty_QT_Movie( captureQT, propId, value ) != 0 : false;
}

CvCapture* cvCreateFileCapture_QT( const char* filename )
{
    CvCapture_QT_Movie_CPP* capture = new CvCapture_QT_Movie_CPP;

    if( capture->open( filename ))
        return capture;

    delete capture;
    return 0;
}


/////////////////////////////////////

class CvCapture_QT_Cam_CPP : public CvCapture
{
public:
    CvCapture_QT_Cam_CPP() { captureQT = 0; }
    virtual ~CvCapture_QT_Cam_CPP() { close(); }

    virtual bool open( int index );
    virtual void close();

    virtual double getProperty(int);
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);
	virtual int getCaptureDomain() { return CV_CAP_QT; } // Return the type of the capture object: CV_CAP_VFW, etc...
protected:

    CvCapture_QT_Cam* captureQT;
};

bool CvCapture_QT_Cam_CPP::open( int index )
{
    close();
    captureQT = icvCaptureFromCam_QT( index );
    return captureQT != 0;
}

void CvCapture_QT_Cam_CPP::close()
{
    if( captureQT )
    {
        icvClose_QT_Cam( captureQT );
        cvFree( &captureQT );
    }
}

bool CvCapture_QT_Cam_CPP::grabFrame()
{
    return captureQT ? icvGrabFrame_QT_Cam( captureQT ) != 0 : false;
}

IplImage* CvCapture_QT_Cam_CPP::retrieveFrame(int)
{
    return captureQT ? (IplImage*)icvRetrieveFrame_QT_Cam( captureQT, 0 ) : 0;
}

double CvCapture_QT_Cam_CPP::getProperty( int propId )
{
    return captureQT ? icvGetProperty_QT_Cam( captureQT, propId ) : 0;
}

bool CvCapture_QT_Cam_CPP::setProperty( int propId, double value )
{
    return captureQT ? icvSetProperty_QT_Cam( captureQT, propId, value ) != 0 : false;
}

CvCapture* cvCreateCameraCapture_QT( int index )
{
    CvCapture_QT_Cam_CPP* capture = new CvCapture_QT_Cam_CPP;

    if( capture->open( index ))
        return capture;

    delete capture;
    return 0;
}

/////////////////////////////////

class CvVideoWriter_QT_CPP : public CvVideoWriter
{
public:
    CvVideoWriter_QT_CPP() { writerQT = 0; }
    virtual ~CvVideoWriter_QT_CPP() { close(); }

    virtual bool open( const char* filename, int fourcc,
                       double fps, CvSize frameSize, bool isColor );
    virtual void close();
    virtual bool writeFrame( const IplImage* );

protected:
    CvVideoWriter_QT* writerQT;
};

bool CvVideoWriter_QT_CPP::open( const char* filename, int fourcc,
                       double fps, CvSize frameSize, bool isColor )
{
    close();
    writerQT = icvCreateVideoWriter_QT( filename, fourcc, fps, frameSize, isColor );
    return writerQT != 0;
}

void CvVideoWriter_QT_CPP::close()
{
    if( writerQT )
    {
        icvReleaseVideoWriter_QT( &writerQT );
        writerQT = 0;
    }
}

bool CvVideoWriter_QT_CPP::writeFrame( const IplImage* image )
{
    if( !writerQT || !image )
        return false;
    return icvWriteFrame_QT( writerQT, image ) >= 0;
}

CvVideoWriter* cvCreateVideoWriter_QT( const char* filename, int fourcc,
                                       double fps, CvSize frameSize, int isColor )
{
    CvVideoWriter_QT_CPP* writer = new CvVideoWriter_QT_CPP;
    if( writer->open( filename, fourcc, fps, frameSize, isColor != 0 ))
        return writer;
    delete writer;
    return 0;
}
