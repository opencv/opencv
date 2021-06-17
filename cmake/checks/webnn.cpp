#include <webnn/webnn_cpp.h>
#include <webnn/webnn.h>
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>


int main(int /*argc*/, char** /*argv*/)
{
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    webnnProcSetProcs(&backendProcs);
    ml::Context ml_context = ml::Context(webnn_native::CreateContext());
    return 0;
}