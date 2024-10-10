#include <webp/mux.h>

int main()
{
  WebPAnimEncoderOptions anim_config;
  WebPAnimEncoderOptionsInit(&anim_config);
  WebPAnimEncoder* anim_encoder = WebPAnimEncoderNew(320, 240, &anim_config);
  WebPAnimEncoderDelete(anim_encoder);
  return 0;
}
