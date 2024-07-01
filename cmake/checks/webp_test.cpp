#include <webp/mux.h>

int main()
{
  WebPAnimEncoderOptions anim_config;
  WebPAnimEncoder* anim_encoder = WebPAnimEncoderNew(320, 240, &anim_config);
  return 0;
}
