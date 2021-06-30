#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace cv
{
   struct GAPI_EXPORTS_W_SIMPLE GCompileArg {
       GAPI_WRAP GCompileArg(gapi::GKernelPackage pkg);
       GAPI_WRAP GCompileArg(gapi::GNetPackage pkg);
   };

   class GAPI_EXPORTS_W_SIMPLE GInferInputs
   {
   public:
       GAPI_WRAP GInferInputs();
       GAPI_WRAP GInferInputs& setInput(const std::string& name, const cv::GMat&   value);
       GAPI_WRAP GInferInputs& setInput(const std::string& name, const cv::GFrame& value);
   };

   class GAPI_EXPORTS_W_SIMPLE GInferListInputs
   {
   public:
       GAPI_WRAP GInferListInputs();
       GAPI_WRAP GInferListInputs setInput(const std::string& name, const cv::GArray<cv::GMat>& value);
       GAPI_WRAP GInferListInputs setInput(const std::string& name, const cv::GArray<cv::Rect>& value);
   };

   class GAPI_EXPORTS_W_SIMPLE GInferOutputs
   {
   public:
       GAPI_WRAP GInferOutputs();
       GAPI_WRAP cv::GMat at(const std::string& name);
   };

   class GAPI_EXPORTS_W_SIMPLE GInferListOutputs
   {
   public:
       GAPI_WRAP GInferListOutputs();
       GAPI_WRAP cv::GArray<cv::GMat> at(const std::string& name);
   };

   namespace detail
   {
       gapi::GNetParam GAPI_EXPORTS_W strip(gapi::ie::PyParams params);
   } // namespace detail

   namespace gapi
   {
       namespace streaming
       {
           // FIXME: Extend to work with an arbitrary G-type.
           cv::GOpaque<int64_t> GAPI_EXPORTS_W timestamp(cv::GMat);
           cv::GOpaque<int64_t> GAPI_EXPORTS_W seqNo(cv::GMat);
           cv::GOpaque<int64_t> GAPI_EXPORTS_W seq_id(cv::GMat);
       } // namespace streaming
       namespace wip
       {
           class GAPI_EXPORTS_W IStreamSource { };
       } // namespace wip
   } // namespace gapi
} // namespace cv
