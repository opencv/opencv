/* Copyright (c) 1998, 1999 Thai Open Source Software Center Ltd
   See the file COPYING for copying permission.
*/

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

#ifndef MAP_FILE
#define MAP_FILE 0
#endif

#include "filemap.h"

int
filemap(const char *name,
        void (*processor)(const void *, size_t, const char *, void *arg),
        void *arg)
{
  int fd;
  size_t nbytes;
  struct stat sb;
  void *p;

  fd = open(name, O_RDONLY);
  if (fd < 0) {
    perror(name);
    return 0;
  }
  if (fstat(fd, &sb) < 0) {
    perror(name);
    close(fd);
    return 0;
  }
  if (!S_ISREG(sb.st_mode)) {
    close(fd);
    fprintf(stderr, "%s: not a regular file\n", name);
    return 0;
  }

  nbytes = sb.st_size;
  p = (void *)mmap((caddr_t)0, (size_t)nbytes, PROT_READ,
                   MAP_FILE|MAP_PRIVATE, fd, (off_t)0);
  if (p == (void *)-1) {
    perror(name);
    close(fd);
    return 0;
  }
  processor(p, nbytes, name, arg);
  munmap((caddr_t)p, nbytes);
  close(fd);
  return 1;
}
