#include <webnn/webnn_cpp.h>
#include <webnn/webnn.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webnn.h>
#else
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>
#endif


int main(int /*argc*/, char** /*argv*/)
{
#ifdef __EMSCRIPTEN__
    ml::Context ml_context = ml::Context(emscripten_webnn_create_context());
#else
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    webnnProcSetProcs(&backendProcs);
    ml::Context ml_context = ml::Context(webnn_native::CreateContext());
#endif
    return 0;
}