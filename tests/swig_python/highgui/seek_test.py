"""
This script will test highgui's seek functionality
for different video formats
"""

# import the necessary things for OpenCV and comparson routine
import os
#import python
#from python.highgui import *
#from python.cv import *
import match
from highgui import *
from cv import *

# path to videos and images we need
PREFIX=os.path.join(os.environ["srcdir"],"../../opencv_extra/testdata/python/")

# this is the folder with the videos and images
# and name of output window
IMAGES		= PREFIX+"images/"
VIDEOS		= PREFIX+"videos/"



show_frames=False

# testing routine, seeks through file and compares read images with frames in frames.QCIF[]
def seek_frame_ok(FILENAME,ERRORS):
  # create a video reader using the tiny videofile VIDEOS+FILENAME
  video=cvCreateFileCapture(VIDEOS+FILENAME)

  if video is None:
    # couldn't open video (FAIL)
    return 1

  if show_frames:
    cvNamedWindow("test", CV_WINDOW_AUTOSIZE)
  
  # skip 2 frames and read 3rd frame each until EOF and check if the read image is ok
  for k in [0,3,6,9,12,15,18,21,24,27]:
    cvSetCaptureProperty(video, CV_CAP_PROP_POS_FRAMES, k)

    # try to query frame
    image=cvQueryFrame(video)

    if image is None:
      # returned image is NULL (FAIL)
      return 1

    compresult = match.match(image,k,ERRORS[k])
    if not compresult:
      return 1

    if show_frames:
      cvShowImage("test",image)
      cvWaitKey(200)

  # same as above, just backwards...
  for k in [27,24,21,18,15,12,9,6,3,0]:

    cvSetCaptureProperty(video, CV_CAP_PROP_POS_FRAMES, k)

    # try to query frame
    image=cvQueryFrame(video)

    if image is None:
    # returned image is NULL (FAIL)
      return 1

    compresult = match.match(image,k,ERRORS[k])
    if not compresult:
      return 1

    if show_frames:
      cvShowImage("test",image)
      cvWaitKey(200)

  # ATTENTION: We do not release the video reader, window or any image.
  # This is bad manners, but Python and OpenCV don't care,
  # the whole memory segment will be freed on finish anyway...

  del video
  # everything is fine (PASS)
  return 0


# testing routine, seeks through file and compares read images with frames in frames.QCIF[]
def seek_time_ok(FILENAME,ERRORS):

  # create a video reader using the tiny videofile VIDEOS+FILENAME
  video=cvCreateFileCapture(VIDEOS+FILENAME)

  if video is None:
    # couldn't open video (FAIL)
    return 1

  if show_frames:
    cvNamedWindow("test", CV_WINDOW_AUTOSIZE)

  # skip 2 frames and read 3rd frame each until EOF and check if the read image is ok
  for k in [0,3,6,9,12,15,18,21,24,27]:

    cvSetCaptureProperty(video, CV_CAP_PROP_POS_MSEC, k*40)

    # try to query frame
    image=cvQueryFrame(video)

    if image is None:
    # returned image is NULL (FAIL)
      return 1

    compresult = match.match(image,k,ERRORS[k])
    if not compresult:
      return 1

    if show_frames:
      cvShowImage("test",image)
      cvWaitKey(200)

  # same as above, just backwards...
  for k in [27,24,21,18,15,12,9,6,3,0]:

    cvSetCaptureProperty(video, CV_CAP_PROP_POS_MSEC, k*40)

    # try to query frame
    image=cvQueryFrame(video)

    if image is None:
    # returned image is NULL (FAIL)
      return 1

    compresult = match.match(image,k,ERRORS[k])
    if not compresult:
      return 1

    if show_frames:
      cvShowImage("test",image)
      cvWaitKey(200)

  # ATTENTION: We do not release the video reader, window or any image.
  # This is bad manners, but Python and OpenCV don't care,
  # the whole memory segment will be freed on finish anyway...

  del video
  # everything is fine (PASS)
  return 0
