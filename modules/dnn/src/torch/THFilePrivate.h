namespace TH {

struct THFile__
{
    struct THFileVTable *vtable;

    int isQuiet;
    int isReadable;
    int isWritable;
    int isBinary;
    int isAutoSpacing;
    int hasError;
};

/* virtual table definition */

struct THFileVTable
{
    int (*isOpened)(THFile *self);

    long (*readByte)(THFile *self, unsigned char *data, long n);
    long (*readChar)(THFile *self, char *data, long n);
    long (*readShort)(THFile *self, short *data, long n);
    long (*readInt)(THFile *self, int *data, long n);
    long (*readLong)(THFile *self, int64 *data, long n);
    long (*readFloat)(THFile *self, float *data, long n);
    long (*readDouble)(THFile *self, double *data, long n);
    long (*readString)(THFile *self, const char *format, char **str_);

    void (*seek)(THFile *self, long position);
    void (*seekEnd)(THFile *self);
    long (*position)(THFile *self);
    void (*close)(THFile *self);
    void (*free)(THFile *self);
};

} // namespace
