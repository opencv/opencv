swig -DSKIP_INCLUDES -python -small highgui.i
gcc -I/usr/include/python2.3/ -I../../cxcore/include -D CV_NO_BACKWARD_COMPATIBILITY -c highgui_wrap.c
