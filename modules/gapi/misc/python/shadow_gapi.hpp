#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
   GAPI_EXPORTS_W GCompileArgs compile_args(gapi::GKernelPackage pkg);
} // namespace cv
