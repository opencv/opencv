#ifndef __CL_PROCESSOR_HPP__
#define __CL_PROCESSOR_HPP__

int initCL();
void closeCL();
void processFrame(int tex1, int tex2, int w, int h, int mode);

#endif
