#ifndef _BDATYPES_H
#define _BDATYPES_H
#if __GNUC__ >= 3
#pragma GCC system_header
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--- DirectShow Reference - DirectShow Enumerated Types */
typedef enum {
	MEDIA_TRANSPORT_PACKET,
	MEDIA_ELEMENTARY_STREAM,
	MEDIA_MPEG2_PSI,
	MEDIA_TRANSPORT_PAYLOAD
} MEDIA_SAMPLE_CONTENT;
/*--- DirectShow Reference - DirectShow Structures */
typedef struct {
	DWORD dwOffset;
	DWORD dwPacketLength;
	DWORD dwStride;
} MPEG2_TRANSPORT_STRIDE;
typedef struct {
	ULONG ulPID;
	MEDIA_SAMPLE_CONTENT MediaSampleContent ;
} PID_MAP;

#ifdef __cplusplus
}
#endif
#endif
