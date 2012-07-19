# EXAMPLES OF HOW TO RUN THE PROGRAMS ASSUMING YOU ARE IN THEIR DIRECTORY
# On Linux, make the files using "make" (which will invoke Makefile in the directory)
# If you just copy programs from the pdf of the book, watch out that the 
# quote characters aren't special characters. 
#
# I put a calibration pattern, OpenCV_Chessboard.png in this directory
# print it out and tape it to any rigid object like a book or binder. It's
# for use with the camera calibration programs in Chapter 11.
#

#------------------------------------------------------------------------#
#CHAPTER 2 PROGRAMS:
./ch2_ex2_1 stuff.jpg
./ch2_ex2_2 test.avi
./ch2_ex2_3 test.avi
./ch2_ex2_4 stuff.jpg 
./ch2_ex2_5 stuff.jpg 
./ch2_ex2_6 stuff.jpg 
./ch2_ex2_7 stuff.jpg 
./ch2_ex2_8 stuff.jpg 
  
  # With video camera: 
./ch2_ex2_9     

  # Using movie file: 
./ch2_ex2_9 test.avi

  # Writing won't work if you are using gstreamer. Use ffmpeg but might have to have developer libs from ffmpeg 
./ch2_ex2_10 test.avi log_polar_out.avi


#------------------------------------------------------------------------#
#CHAPTER 3 PROGRAMS:   
  ch3_ex3_1.txt  [not a program]
  ch3_ex3_2.txt  [not a program]
./ch3_ex3_3
./ch3_ex3_4
./ch3_ex3_5
  ch3_ex3_6.txt  [not a program]
  ch3_ex3_7.txt  [not a program]
  ch3_ex3_8.txt  [not a program]  
./ch3_ex3_9
  ch3_ex3_10.txt [not a program]
./ch3_ex3_11 stuff.jpg
./ch3_ex3_12 stuff.jpg 50 100 300 200 100
./ch3_ex3_13 stuff.jpg 50 100 300 200 100
./ch3_ex3_14 stuff.jpg adrian.jpg 50 100 300 200 0.5 0.5
./ch3_ex3_15
  ch3_ex3_16.txt [not a program]
./ch3_ex3_17
  ch3_ex3_18.xml [not a program]
./ch3_ex3_19
./ch3_ex3_20


#------------------------------------------------------------------------#
#CHAPTER 4 PROGRAMS
./ch4_ex4_1 
./ch4_ex4_2 test.avi
./ch4_ex4_3 test.avi


#------------------------------------------------------------------------#
#CHAPTER 5 PROGRAMS
./ch5_ex5_1 stuff.jpg 
./ch5_ex5_2 adrian.jpg
./ch5_ex5_3 adrian.jpg
./ch5_ex5_4 15 1 1 71 15 adrian.jpg 

#------------------------------------------------------------------------#
#CHAPTER 6
./ch6_ex6_1 stuff.jpg
./ch6_ex6_2 stuff.jpg 
./ch6_ex6_3 adrian.jpg 
./ch6_ex6_4 stuff.jpg 100
./ch6_ex6_5 


#------------------------------------------------------------------------#
#CHAPTER 7
./ch7_ex7_1 adrian.jpg 
  ch7_ex7_2.txt  [not a program]
./ch7_ex7_3_expanded HandIndoorColor.jpg HandOutdoorSunColor.jpg HandOutdoorColor.jpg adrian.jpg 
  ch7_ex7_4.txt  [not a program]
./ch7_ex7_5_HistBackProj BlueCup.jpg adrian.jpg 0
./ch7_ex7_5 faceTemplate.jpg faceScene.jpg 


#------------------------------------------------------------------------#
#CHAPTER 8
  ch8_ex8_1.txt  [not a program]
./ch8_ex8_2 adrian.jpg 
./ch8_ex8_3 faceTemplate.jpg 


#------------------------------------------------------------------------#
#CHAPTER 9
./ch9_ex9_1 test.avi 

  # The background demo shows 2 background modeling methods: Averaging and YUV Codebook
  # I've set the parameters to work well with tree.avi == we learn the model for 50 frames
  # The parameters following tree.avi below are optional, but pretty good for that sequence
  # You can adjust them on the fly.  Best to pause "p", adjust, single step a bit with "s"
  # and then resume running with "r" or "p".  To adjust, the video window and NOT the consul
  # window must be active.  I put a built in pause in the short tree.avi sequence so you
  # can experiment with parmeters as you single step "s", adjust" and step "s" along. -- Gary
./ch9_background 1 50 tree.avi 9 2 35 16 2 15 11 16

  # Copy of watershed.cpp in samples/c directory is also here
./ch9_watershed fruits.jpg 

  #This one isn't a stand alone program, it is a function that does frame differencing as in the book
ch9_backgroundDiff.cpp  [not built as a program]

  #This is a potentially add on function to ch9_backgroundAVG.cpp -- it clears stale codebook entries
ch9_ClearStaleCB_Entries.cpp [not built as a program]


#------------------------------------------------------------------------#
#CHAPTER 10
 ./ch10_ex10_1 
 
   # Decided to do the same example, except using the Horn-Schunck algorithm
./ch10_ex10_1b_Horn_Schunck

  # For fun, here's motion template code copied from samples/c.  Uses a video camera.  Point the camera
  # away from you.  Start the program, wait a few seconds until it goes black.  Then move in front
  # of the camera.
./ch10_motempl 

./ch10_ex10_2


#------------------------------------------------------------------------#
#CHAPTER 11
  # Calibration using video camera
  #  Parameters are chessboard_width height num_to_collect  wait_frames
  #  The num_to_collect is how many valid chessboard captures we should use to calibrate
  #  The wait_frames is how many frames to skip before attempting to find a chessboard (this allows you to move it around).
  #  NOTE: The "9 6" work with OpenCV_Chessboard.png which you can print out and use for this.
./ch11_ex11_1 9 6 10 60

  # We can also calibrate from a list of images on disk.  You will need to run this in order for the birdseye demo to work.
./ch11_ex11_1_fromdisk 12 12 ch11_chessboards.txt


#------------------------------------------------------------------------#
#CHAPTER 12
./ch12_ex12_1 12 12 Intrinsics.xml Distortion.xml ch12_birdseye.jpg 

  # Just a belabored example of computing the fundamental matrix
  #  NOTE: The "9 6" work with OpenCV_Chessboard.png which you can print out and use for this.
./ch12_ex12_2 9 6 8 40

  # Stereo calibration, rectification and depth
./ch12_ex12_3

./ch12_ex12_4


#------------------------------------------------------------------------#
#CHAPTER 13
./ch13_ex13_1
./ch13_ex13_2
./ch13_ex13_3
./ch13_ex13_4





#------------------------------------------------------------------------#
#CHAPTER 14
  # You fill in these





  
