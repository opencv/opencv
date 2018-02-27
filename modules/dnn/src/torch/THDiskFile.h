#ifndef TH_DISK_FILE_INC
#define TH_DISK_FILE_INC

#include "THFile.h"
#include <string>

namespace TH
{

TH_API THFile *THDiskFile_new(const std::string &name, const char *mode, int isQuiet);

TH_API int THDiskFile_isLittleEndianCPU(void);
TH_API int THDiskFile_isBigEndianCPU(void);
TH_API void THDiskFile_nativeEndianEncoding(THFile *self);
TH_API void THDiskFile_littleEndianEncoding(THFile *self);
TH_API void THDiskFile_bigEndianEncoding(THFile *self);
TH_API void THDiskFile_longSize(THFile *self, int size);
TH_API void THDiskFile_noBuffer(THFile *self);

} // namespace

#endif
