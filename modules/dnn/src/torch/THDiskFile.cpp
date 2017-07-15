#include "../precomp.hpp"
#if defined(ENABLE_TORCH_IMPORTER) && ENABLE_TORCH_IMPORTER
#include "THGeneral.h"
#include "THDiskFile.h"
#include "THFilePrivate.h"

namespace TH
{

typedef struct THDiskFile__
{
    THFile file;

    FILE *handle;
    char *name;
    int isNativeEncoding;
    int longSize;

} THDiskFile;

static int THDiskFile_isOpened(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)self;
  return (dfself->handle != NULL);
}

const char *THDiskFile_name(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)self;
  return dfself->name;
}

/* workaround mac osx lion ***insane*** fread bug */
#ifdef __APPLE__
static size_t fread__(void *ptr, size_t size, size_t nitems, FILE *stream)
{
  size_t nread = 0;
  while(!feof(stream) && !ferror(stream) && (nread < nitems))
    nread += fread((char*)ptr+nread*size, size, std::min<size_t>(2147483648UL/size, nitems-nread), stream);
  return nread;
}
#else
#define fread__ fread
#endif

#define READ_WRITE_METHODS(TYPE, TYPEC, ASCII_READ_ELEM, ASCII_WRITE_ELEM) \
  static long THDiskFile_read##TYPEC(THFile *self, TYPE *data, long n)  \
  {                                                                     \
    THDiskFile *dfself = (THDiskFile*)(self);                           \
    long nread = 0L;                                                    \
                                                                        \
    THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file"); \
    THArgCheck(dfself->file.isReadable, 1, "attempt to read in a write-only file"); \
                                                                        \
    if(dfself->file.isBinary)                                           \
    {                                                                   \
      nread = fread__(data, sizeof(TYPE), n, dfself->handle);           \
      if(!dfself->isNativeEncoding && (sizeof(TYPE) > 1) && (nread > 0)) \
        THDiskFile_reverseMemory(data, data, sizeof(TYPE), nread);      \
    }                                                                   \
    else                                                                \
    {                                                                   \
      long i;                                                           \
      for(i = 0; i < n; i++)                                            \
      {                                                                 \
        ASCII_READ_ELEM; /* increment here result and break if wrong */ \
      }                                                                 \
      if(dfself->file.isAutoSpacing && (n > 0))                         \
      {                                                                 \
        int c = fgetc(dfself->handle);                                  \
        if( (c != '\n') && (c != EOF) )                                 \
          ungetc(c, dfself->handle);                                    \
      }                                                                 \
    }                                                                   \
                                                                        \
    if(nread != n)                                                      \
    {                                                                   \
      dfself->file.hasError = 1; /* shouldn't we put hasError to 0 all the time ? */ \
      if(!dfself->file.isQuiet)                                         \
        THError("read error: read %d blocks instead of %d", nread, n);  \
    }                                                                   \
                                                                        \
    return nread;                                                       \
  }

static int THDiskFile_mode(const char *mode, int *isReadable, int *isWritable)
{
  *isReadable = 0;
  *isWritable = 0;
  if(strlen(mode) == 1)
  {
    if(*mode == 'r')
    {
      *isReadable = 1;
      return 1;
    }
    else if(*mode == 'w')
    {
      *isWritable = 1;
      return 1;
    }
  }
  else if(strlen(mode) == 2)
  {
    if(mode[0] == 'r' && mode[1] == 'w')
    {
      *isReadable = 1;
      *isWritable = 1;
      return 1;
    }
  }
  return 0;
}

static void THDiskFile_seek(THFile *self, long position)
{
  THDiskFile *dfself = (THDiskFile*)(self);

  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");

#if defined(_WIN64)
  if(_fseeki64(dfself->handle, (__int64)position, SEEK_SET) < 0)
#elif defined(_WIN32)
  if(fseek(dfself->handle, (long)position, SEEK_SET) < 0)
#else
  if(fseeko(dfself->handle, (off_t)position, SEEK_SET) < 0)
#endif
  {
    dfself->file.hasError = 1;
    if(!dfself->file.isQuiet)
      THError("unable to seek at position %d", position);
  }
}

static void THDiskFile_seekEnd(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)(self);

  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");

#if defined(_WIN64)
  if(_fseeki64(dfself->handle, 0L, SEEK_END) < 0)
#elif defined(_WIN32)
  if(fseek(dfself->handle, 0L, SEEK_END) < 0)
#else
  if(fseeko(dfself->handle, 0L, SEEK_END) < 0)
#endif
  {
    dfself->file.hasError = 1;
    if(!dfself->file.isQuiet)
      THError("unable to seek at end of file");
  }
}

static long THDiskFile_position(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");

#if defined(_WIN64)
  __int64 offset = _ftelli64(dfself->handle);
#elif defined(_WIN32)
  long offset = ftell(dfself->handle);
#else
  off_t offset = ftello(dfself->handle);
#endif
  if (offset > -1)
      return (long)offset;
  else if(!dfself->file.isQuiet)
      THError("unable to obtain disk file offset (maybe a long overflow occurred)");

  return 0;
}

static void THDiskFile_close(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");
  fclose(dfself->handle);
  dfself->handle = NULL;
}

/* Little and Big Endian */

static void THDiskFile_reverseMemory(void *dst, const void *src, long blockSize, long numBlocks)
{
  if(blockSize != 1)
  {
    long halfBlockSize = blockSize/2;
    char *charSrc = (char*)src;
    char *charDst = (char*)dst;
    long b, i;
    for(b = 0; b < numBlocks; b++)
    {
      for(i = 0; i < halfBlockSize; i++)
      {
        char z = charSrc[i];
        charDst[i] = charSrc[blockSize-1-i];
        charDst[blockSize-1-i] = z;
      }
      charSrc += blockSize;
      charDst += blockSize;
    }
  }
}

int THDiskFile_isLittleEndianCPU(void)
{
  int x = 7;
  char *ptr = (char *)&x;

  if(ptr[0] == 0)
    return 0;
  else
    return 1;
}

int THDiskFile_isBigEndianCPU(void)
{
  return(!THDiskFile_isLittleEndianCPU());
}

void THDiskFile_nativeEndianEncoding(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");
  dfself->isNativeEncoding = 1;
}

void THDiskFile_littleEndianEncoding(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");
  dfself->isNativeEncoding = THDiskFile_isLittleEndianCPU();
}

void THDiskFile_bigEndianEncoding(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");
  dfself->isNativeEncoding = !THDiskFile_isLittleEndianCPU();
}

/* End of Little and Big Endian Stuff */

void THDiskFile_longSize(THFile *self, int size)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");
  THArgCheck(size == 0 || size == 4 || size == 8, 1, "Invalid long size specified");
  dfself->longSize = size;
}

void THDiskFile_noBuffer(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");
  if (setvbuf(dfself->handle, NULL, _IONBF, 0)) {
    THError("error: cannot disable buffer");
  }
}

static void THDiskFile_free(THFile *self)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  if(dfself->handle)
    fclose(dfself->handle);
  THFree(dfself->name);
  THFree(dfself);
}

/* Note that we do a trick */
READ_WRITE_METHODS(unsigned char, Byte,
                   nread = fread(data, 1, n, dfself->handle); break,
                   nwrite = fwrite(data, 1, n, dfself->handle); break)

READ_WRITE_METHODS(char, Char,
                   nread = fread(data, 1, n, dfself->handle); break,
                   nwrite = fwrite(data, 1, n, dfself->handle); break)

READ_WRITE_METHODS(short, Short,
                   int ret = fscanf(dfself->handle, "%hd", &data[i]); if(ret <= 0) break; else nread++,
                   int ret = fprintf(dfself->handle, "%hd", data[i]); if(ret <= 0) break; else nwrite++)

READ_WRITE_METHODS(int, Int,
                   int ret = fscanf(dfself->handle, "%d\n\r", &data[i]); if(ret <= 0) break; else nread++,
                   int ret = fprintf(dfself->handle, "%d", data[i]); if(ret <= 0) break; else nwrite++)

/*READ_WRITE_METHODS(long, Long,
                   int ret = fscanf(dfself->handle, "%ld", &data[i]); if(ret <= 0) break; else nread++,
                   int ret = fprintf(dfself->handle, "%ld", data[i]); if(ret <= 0) break; else nwrite++)*/

READ_WRITE_METHODS(float, Float,
                   int ret = fscanf(dfself->handle, "%g", &data[i]); if(ret <= 0) break; else nread++,
                   int ret = fprintf(dfself->handle, "%.9g", data[i]); if(ret <= 0) break; else nwrite++)

READ_WRITE_METHODS(double, Double,
                   int ret = fscanf(dfself->handle, "%lg", &data[i]); if(ret <= 0) break; else nread++,
                   int ret = fprintf(dfself->handle, "%.17g", data[i]); if(ret <= 0) break; else nwrite++)


/* For Long we need to rewrite everything, because of the special management of longSize */
static long THDiskFile_readLong(THFile *self, int64 *data, long n)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  long nread = 0L;

  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");
  THArgCheck(dfself->file.isReadable, 1, "attempt to read in a write-only file");

  if(dfself->file.isBinary)
  {
    if(dfself->longSize == 0 || dfself->longSize == sizeof(int64))
    {
      nread = fread__(data, sizeof(int64), n, dfself->handle);
      if(!dfself->isNativeEncoding && (sizeof(int64) > 1) && (nread > 0))
        THDiskFile_reverseMemory(data, data, sizeof(int64), nread);
    } else if(dfself->longSize == 4)
    {
      nread = fread__(data, 4, n, dfself->handle);
      if(!dfself->isNativeEncoding && (nread > 0))
        THDiskFile_reverseMemory(data, data, 4, nread);
      long i;
      for(i = nread; i > 0; i--)
        data[i-1] = ((int *)data)[i-1];
    }
    else /* if(dfself->longSize == 8) */
    {
      int big_endian = !THDiskFile_isLittleEndianCPU();
      int32_t *buffer = (int32_t*)THAlloc(8*n);
      if (!buffer)
          THError("can not allocate buffer");
      nread = fread__(buffer, 8, n, dfself->handle);
      long i;
      for(i = nread; i > 0; i--)
        data[i-1] = buffer[2*(i-1) + big_endian];
      THFree(buffer);
      if(!dfself->isNativeEncoding && (nread > 0))
        THDiskFile_reverseMemory(data, data, 4, nread);
     }
  }
  else
  {
    long i;
    for(i = 0; i < n; i++)
    {
      long d;
      int ret = fscanf(dfself->handle, "%ld", &d); if(ret <= 0) break; else nread++;
      data[i] = d;
    }
    if(dfself->file.isAutoSpacing && (n > 0))
    {
      int c = fgetc(dfself->handle);
      if( (c != '\n') && (c != EOF) )
        ungetc(c, dfself->handle);
    }
  }

  if(nread != n)
  {
    dfself->file.hasError = 1; /* shouldn't we put hasError to 0 all the time ? */
    if(!dfself->file.isQuiet)
      THError("read error: read %d blocks instead of %d", nread, n);
  }

  return nread;
}


static long THDiskFile_readString(THFile *self, const char *format, char **str_)
{
  THDiskFile *dfself = (THDiskFile*)(self);
  THArgCheck(dfself->handle != NULL, 1, "attempt to use a closed file");
  THArgCheck(dfself->file.isReadable, 1, "attempt to read in a write-only file");
  THArgCheck((strlen(format) >= 2 ? (format[0] == '*') && (format[1] == 'a' || format[1] == 'l') : 0), 2, "format must be '*a' or '*l'");

/* note: the string won't survive long, as it is copied into lua */
/* so 1024 is not that big... */
#define TBRS_BSZ 1024L

  if(format[1] == 'a')
  {
    char *p = (char*)THAlloc(TBRS_BSZ);
    long total = TBRS_BSZ;
    long pos = 0L;

    for (;;)
    {
      if(total-pos == 0) /* we need more space! */
      {
        total += TBRS_BSZ;
        p = (char*)THRealloc(p, total);
      }
      if (p == NULL)
        THError("read error: failed to allocate buffer");
      pos += fread(p+pos, 1, total-pos, dfself->handle);
      if (pos < total) /* eof? */
      {
        if(pos == 0L)
        {
          THFree(p);
          dfself->file.hasError = 1;
          if(!dfself->file.isQuiet)
            THError("read error: read 0 blocks instead of 1");

          *str_ = NULL;
          return 0;
        }
        *str_ = p;
        return pos;
      }
    }
  }
  else
  {
    char *p = (char*)THAlloc(TBRS_BSZ);
    long total = TBRS_BSZ;
    long pos = 0L;
    long size;

    for (;;)
    {
      if(total-pos <= 1) /* we can only write '\0' in there! */
      {
        total += TBRS_BSZ;
        p = (char*)THRealloc(p, total);
      }
      if (p == NULL)
        THError("read error: failed to allocate buffer");
      if (fgets(p+pos, total-pos, dfself->handle) == NULL) /* eof? */
      {
        if(pos == 0L)
        {
          THFree(p);
          dfself->file.hasError = 1;
          if(!dfself->file.isQuiet)
            THError("read error: read 0 blocks instead of 1");

          *str_ = NULL;
          return 0;
        }
        *str_ = p;
        return pos;
      }
      size = strlen(p+pos);
      if (size == 0L || (p+pos)[size-1] != '\n')
      {
        pos += size;
      }
      else
      {
        pos += size-1L; /* do not include `eol' */
        *str_ = p;
        return pos;
      }
    }
  }

  *str_ = NULL;
  return 0;
}


THFile *THDiskFile_new(const char *name, const char *mode, int isQuiet)
{
  static struct THFileVTable vtable = {
    THDiskFile_isOpened,

    THDiskFile_readByte,
    THDiskFile_readChar,
    THDiskFile_readShort,
    THDiskFile_readInt,
    THDiskFile_readLong,
    THDiskFile_readFloat,
    THDiskFile_readDouble,
    THDiskFile_readString,

    THDiskFile_seek,
    THDiskFile_seekEnd,
    THDiskFile_position,
    THDiskFile_close,
    THDiskFile_free
  };

  int isReadable;
  int isWritable;
  FILE *handle;
  THDiskFile *self;

  THArgCheck(THDiskFile_mode(mode, &isReadable, &isWritable), 2, "file mode should be 'r','w' or 'rw'");

  CV_Assert(isReadable && !isWritable);
#ifdef _MSC_VER
  if (fopen_s(&handle, name, "rb") != 0)
      handle = NULL;
#else
  handle = fopen(name,"rb");
#endif

  if(!handle)
  {
    if(isQuiet)
      return 0;
    else
      THError("cannot open <%s> in mode %c%c", name, (isReadable ? 'r' : ' '), (isWritable ? 'w' : ' '));
  }

  self = (THDiskFile*)THAlloc(sizeof(THDiskFile));
  if (!self)
      THError("cannot allocate memory for self");

  self->handle = handle;
  self->name = (char*)THAlloc(strlen(name)+1);
  if (!self->name)
      THError("cannot allocate memory for self->name");
  strcpy(self->name, name);
  self->isNativeEncoding = 1;
  self->longSize = 0;

  self->file.vtable = &vtable;
  self->file.isQuiet = isQuiet;
  self->file.isReadable = isReadable;
  self->file.isWritable = isWritable;
  self->file.isBinary = 0;
  self->file.isAutoSpacing = 1;
  self->file.hasError = 0;

  return (THFile*)self;
}

}
#endif
