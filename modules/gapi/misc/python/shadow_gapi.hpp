#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
   GAPI_EXPORTS_W GCompileArgs compile_args(gapi::GKernelPackage pkg);
   class GAPI_EXPORTS_W_SIMPLE GProtoInputArgs {};
   class GAPI_EXPORTS_W_SIMPLE GProtoOutputArgs {};
   class GAPI_EXPORTS_W_SIMPLE GRunArg {};
} // namespace cv
