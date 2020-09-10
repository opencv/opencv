#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
   GAPI_EXPORTS_W GCompileArgs compile_args(gapi::GKernelPackage pkg);
   class GAPI_EXPORTS_W_SIMPLE GProtoArg { };
   class GAPI_EXPORTS_W_SIMPLE GProtoInputArgs { };
   class GAPI_EXPORTS_W_SIMPLE GProtoOutputArgs { };

   using GProtoInputArgs  = GIOProtoArgs<In_Tag>;
   using GProtoOutputArgs = GIOProtoArgs<Out_Tag>;
} // namespace cv
