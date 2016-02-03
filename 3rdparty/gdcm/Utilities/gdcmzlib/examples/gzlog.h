/* gzlog.h
  Copyright (C) 2004 Mark Adler, all rights reserved
  version 1.0, 26 Nov 2004

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the author be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  Mark Adler    madler@alumni.caltech.edu
 */

/*
   The gzlog object allows writing short messages to a gzipped log file,
   opening the log file locked for small bursts, and then closing it.  The log
   object works by appending stored data to the gzip file until 1 MB has been
   accumulated.  At that time, the stored data is compressed, and replaces the
   uncompressed data in the file.  The log file is truncated to its new size at
   that time.  After closing, the log file is always valid gzip file that can
   decompressed to recover what was written.

   A gzip header "extra" field contains two file offsets for appending.  The
   first points to just after the last compressed data.  The second points to
   the last stored block in the deflate stream, which is empty.  All of the
   data between those pointers is uncompressed.
 */

/* Open a gzlog object, creating the log file if it does not exist.  Return
   NULL on error.  Note that gzlog_open() could take a long time to return if
   there is difficulty in locking the file. */
void *gzlog_open(char *path);

/* Write to a gzlog object.  Return non-zero on error.  This function will
   simply write data to the file uncompressed.  Compression of the data
   will not occur until gzlog_close() is called.  It is expected that
   gzlog_write() is used for a short message, and then gzlog_close() is
   called.  If a large amount of data is to be written, then the application
   should write no more than 1 MB at a time with gzlog_write() before
   calling gzlog_close() and then gzlog_open() again. */
int gzlog_write(void *log, char *data, size_t len);

/* Close a gzlog object.  Return non-zero on error.  The log file is locked
   until this function is called.  This function will compress stored data
   at the end of the gzip file if at least 1 MB has been accumulated.  Note
   that the file will not be a valid gzip file until this function completes.
 */
int gzlog_close(void *log);
