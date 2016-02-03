#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "expat.h"

#ifdef XML_LARGE_SIZE
#define XML_FMT_INT_MOD "ll"
#else
#define XML_FMT_INT_MOD "l"
#endif

static void
usage(const char *prog, int rc)
{
  fprintf(stderr,
          "usage: %s [-n] filename bufferSize nr_of_loops\n", prog);
  exit(rc);
}

#ifdef AMIGA_SHARED_LIB
#include <proto/expat.h>
int
amiga_main(int argc, char *argv[])
#else
int main (int argc, char *argv[]) 
#endif
{
  XML_Parser  parser;
  char        *XMLBuf, *XMLBufEnd, *XMLBufPtr;
  FILE        *fd;
  struct stat fileAttr;
  int         nrOfLoops, bufferSize, fileSize, i, isFinal;
  int         j = 0, ns = 0;
  clock_t     tstart, tend;
  double      cpuTime = 0.0;

  if (argc > 1) {
    if (argv[1][0] == '-') {
      if (argv[1][1] == 'n' && argv[1][2] == '\0') {
        ns = 1;
        j = 1;
      }
      else
        usage(argv[0], 1);
    }
  }

  if (argc != j + 4)
    usage(argv[0], 1);

  if (stat (argv[j + 1], &fileAttr) != 0) {
    fprintf (stderr, "could not access file '%s'\n", argv[j + 1]);
    return 2;
  }
  
  fd = fopen (argv[j + 1], "r");
  if (!fd) {
    fprintf (stderr, "could not open file '%s'\n", argv[j + 1]);
    exit(2);
  }
  
  bufferSize = atoi (argv[j + 2]);
  nrOfLoops = atoi (argv[j + 3]);
  if (bufferSize <= 0 || nrOfLoops <= 0) {
    fprintf (stderr, 
             "buffer size and nr of loops must be greater than zero.\n");
    exit(3);
  }

  XMLBuf = malloc (fileAttr.st_size);
  fileSize = fread (XMLBuf, sizeof (char), fileAttr.st_size, fd);
  fclose (fd);
  
  i = 0;
  XMLBufEnd = XMLBuf + fileSize;
  while (i < nrOfLoops) {
    XMLBufPtr = XMLBuf;
    isFinal = 0;
    if (ns)
      parser = XML_ParserCreateNS(NULL, '!');
    else
      parser = XML_ParserCreate(NULL);
    tstart = clock();
    do {
      int parseBufferSize = XMLBufEnd - XMLBufPtr;
      if (parseBufferSize <= bufferSize)
        isFinal = 1;
      else
        parseBufferSize = bufferSize;
      if (!XML_Parse (parser, XMLBufPtr, parseBufferSize, isFinal)) {
        fprintf (stderr, "error '%s' at line %" XML_FMT_INT_MOD \
                     "u character %" XML_FMT_INT_MOD "u\n",
                 XML_ErrorString (XML_GetErrorCode (parser)),
                 XML_GetCurrentLineNumber (parser),
                 XML_GetCurrentColumnNumber (parser));
        free (XMLBuf);
        XML_ParserFree (parser);
        exit (4);
      }
      XMLBufPtr += bufferSize;
    } while (!isFinal);
    tend = clock();
    cpuTime += ((double) (tend - tstart)) / CLOCKS_PER_SEC;
    XML_ParserFree (parser);
    i++;
  }

  free (XMLBuf);
      
  printf ("%d loops, with buffer size %d. Average time per loop: %f\n", 
          nrOfLoops, bufferSize, cpuTime / (double) nrOfLoops);
  return 0;
}
