#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
   struct GAPI_EXPORTS_W_SIMPLE GCompileArg { };

   GAPI_EXPORTS_W GCompileArgs compile_args(gapi::GKernelPackage pkg);
   GAPI_EXPORTS_W GCompileArgs compile_args(gapi::GNetPackage pkg);

   // NB: This classes doesn't exist in *.so
   // HACK: Mark them as a class to force python wrapper generate code for this entities
   class GAPI_EXPORTS_W_SIMPLE GProtoArg { };
   class GAPI_EXPORTS_W_SIMPLE GProtoInputArgs { };
   class GAPI_EXPORTS_W_SIMPLE GProtoOutputArgs { };
   class GAPI_EXPORTS_W_SIMPLE GRunArg { };
   class GAPI_EXPORTS_W_SIMPLE GMetaArg { };

   class GAPI_EXPORTS_W_SIMPLE GArrayP2f { };

   using GProtoInputArgs  = GIOProtoArgs<In_Tag>;
   using GProtoOutputArgs = GIOProtoArgs<Out_Tag>;

   namespace gapi
   {
       GAPI_EXPORTS_W gapi::GNetPackage networks(const cv::gapi::ie::PyParams& params);
       namespace wip
       {
           class GAPI_EXPORTS_W IStreamSource { };
       } // namespace wip
   } // namespace gapi
} // namespace cv
