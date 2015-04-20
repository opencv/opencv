/* Copyright (c) 1998, 1999 Thai Open Source Software Center Ltd
   See the file COPYING for copying permission.
*/

#define STRICT 1
#define WIN32_LEAN_AND_MEAN 1

#ifdef XML_UNICODE_WCHAR_T
#ifndef XML_UNICODE
#define XML_UNICODE
#endif
#endif

#ifdef XML_UNICODE
#define UNICODE
#define _UNICODE
#endif /* XML_UNICODE */
#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include "filemap.h"

static void win32perror(const TCHAR *);

int
filemap(const TCHAR *name,
        void (*processor)(const void *, size_t, const TCHAR *, void *arg),
        void *arg)
{
  HANDLE f;
  HANDLE m;
  DWORD size;
  DWORD sizeHi;
  void *p;

  f = CreateFile(name, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                          FILE_FLAG_SEQUENTIAL_SCAN, NULL);
  if (f == INVALID_HANDLE_VALUE) {
    win32perror(name);
    return 0;
  }
  size = GetFileSize(f, &sizeHi);
  if (size == (DWORD)-1) {
    win32perror(name);
    return 0;
  }
  if (sizeHi) {
    _ftprintf(stderr, _T("%s: bigger than 2Gb\n"), name);
    return 0;
  }
  /* CreateFileMapping barfs on zero length files */
  if (size == 0) {
    static const char c = '\0';
    processor(&c, 0, name, arg);
    CloseHandle(f);
    return 1;
  }
  m = CreateFileMapping(f, NULL, PAGE_READONLY, 0, 0, NULL);
  if (m == NULL) {
    win32perror(name);
    CloseHandle(f);
    return 0;
  }
  p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  if (p == NULL) {
    win32perror(name);
    CloseHandle(m);
    CloseHandle(f);
    return 0;
  }
  processor(p, size, name, arg); 
  UnmapViewOfFile(p);
  CloseHandle(m);
  CloseHandle(f);
  return 1;
}

static void
win32perror(const TCHAR *s)
{
  LPVOID buf;
  if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
                    | FORMAT_MESSAGE_FROM_SYSTEM,
                    NULL,
                    GetLastError(),
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    (LPTSTR) &buf,
                    0,
                    NULL)) {
    _ftprintf(stderr, _T("%s: %s"), s, buf);
    fflush(stderr);
    LocalFree(buf);
  }
  else
    _ftprintf(stderr, _T("%s: unknown Windows error\n"), s);
}
