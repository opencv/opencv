#define __STDC_CONSTANT_MACROS

#include <stdlib.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

#define CALC_FFMPEG_VERSION(a,b,c) ( a<<16 | b<<8 | c )

static void test()
{
  AVFormatContext* c = 0;
  AVCodec* avcodec = 0;
  AVFrame* frame = 0;

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(52, 111, 0)
  int err = avformat_open_input(&c, "", NULL, NULL);
#else
  int err = av_open_input_file(&c, "", NULL, 0, NULL);
#endif
}

int main() { test(); return 0; }
