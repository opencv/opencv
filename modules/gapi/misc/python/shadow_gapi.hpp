#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
   GAPI_EXPORTS_W GCompileArgs compile_args(gapi::GKernelPackage pkg);
   class GAPI_EXPORTS_W GProtoArg { };
   class GAPI_EXPORTS_W GProtoInputArgs { };
   class GAPI_EXPORTS_W GProtoOutputArgs { };
   // class GAPI_EXPORTS_W GIOProtoArgs
   //class GAPI_EXPORTS_W GProtoArg { };
} // namespace cv
