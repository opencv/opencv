#! /usr/bin/env octave

cv;
highgui;

global g;
g.image0 = [];
g.image1 = [];
g.threshold1 = 255;
g.threshold2 = 30;
g.l = g.level = 4;
g.block_size = 1000;
g.filter = CV_GAUSSIAN_5x5;
g.storage = [];
g.min_comp = CvConnectedComp();

function set_thresh1( val )
  global g;
  g.threshold1 = val;
  ON_SEGMENT();
endfunction

function set_thresh2( val )
  global g;
  g.threshold2 = val;
  ON_SEGMENT()
endfunction

function ON_SEGMENT()
  global g;
  global cv;
  g
  swig_this(g.image0)
  swig_this(g.image1)
  swig_this(g.storage)
  g.level
  g.threshold1
  g.threshold2
  comp = cv.cvPyrSegmentation(g.image0, g.image1, g.storage, g.level, g.threshold1+1, g.threshold2+1);
  cvShowImage("Segmentation", g.image1);
endfunction

filename = "../c/fruits.jpg";
if (size(argv, 2) >= 1)
  filename = argv(){1};
endif
g.image0 = cvLoadImage( filename, 1);
if (! swig_this(g.image0))
  printf("Error opening %s\n",filename);
  exit(-1);
endif

cvNamedWindow("Source", 0);
cvShowImage("Source", g.image0);
cvNamedWindow("Segmentation", 0);
g.storage = cvCreateMemStorage ( g.block_size );
new_width = bitshift(g.image0.width, -g.level);
new_height = bitshift(g.image0.height, -g.level);
g.image0 = cvCreateImage( cvSize(new_width,new_height), g.image0.depth, g.image0.nChannels );
g.image1 = cvCreateImage( cvSize(new_width,new_height), g.image0.depth, g.image0.nChannels );
## segmentation of the color image
g.l = 1;
g.threshold1 =255;
g.threshold2 =30;
ON_SEGMENT();
g.sthreshold1 = cvCreateTrackbar("Threshold1", "Segmentation", g.threshold1, 255, @set_thresh1);
g.sthreshold2 = cvCreateTrackbar("Threshold2", "Segmentation",  g.threshold2, 255, @set_thresh2);
cvShowImage("Segmentation", image1);
cvWaitKey(0);
cvDestroyWindow("Segmentation");
cvDestroyWindow("Source");
