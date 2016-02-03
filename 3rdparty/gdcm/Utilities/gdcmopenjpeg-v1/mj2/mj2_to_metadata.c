/* mj2_to_metadata.c */
/* Dump MJ2, JP2 metadata (partial so far) to xml file */
/* Contributed to Open JPEG by Glenn Pearson, contract software developer, U.S. National Library of Medicine.

The base code in this file was developed by the author as part of a video archiving
project for the U.S. National Library of Medicine, Bethesda, MD. 
It is the policy of NLM (and U.S. government) to not assert copyright.

A non-exclusive copy of this code has been contributed to the Open JPEG project.
Except for copyright, inclusion of the code within Open JPEG for distribution and use
can be bound by the Open JPEG open-source license and disclaimer, expressed elsewhere.
*/

#include "../libopenjpeg/opj_includes.h"
#include "mj2.h"

#include "mj2_to_metadata.h"
#include <string.h>
#include "getopt.h"

/* -------------------------------------------------------------------------- */

/**
sample error callback expecting a FILE* client object
*/
void error_callback(const char *msg, void *client_data) {
	FILE *stream = (FILE*)client_data;
	fprintf(stream, "[ERROR] %s", msg);
}
/**
sample warning callback expecting a FILE* client object
*/
void warning_callback(const char *msg, void *client_data) {
	FILE *stream = (FILE*)client_data;
	fprintf(stream, "[WARNING] %s", msg);
}
/**
sample debug callback expecting a FILE* client object
*/
void info_callback(const char *msg, void *client_data) {
	FILE *stream = (FILE*)client_data;
	fprintf(stream, "[INFO] %s", msg);
}

/* -------------------------------------------------------------------------- */



/* ------------- */

void help_display()
{
  /*             "1234567890123456789012345678901234567890123456789012345678901234567890123456789" */
  fprintf(stdout,"                Help for the 'mj2_to_metadata' Program\n");
  fprintf(stdout,"                ======================================\n");
  fprintf(stdout,"The -h option displays this information on screen.\n\n");
  
  fprintf(stdout,"mj2_to_metadata generates an XML file from a Motion JPEG 2000 file.\n");
  fprintf(stdout,"The generated XML shows the structural, but not (yet) curatorial,\n");
  fprintf(stdout,"metadata from the movie header and from the JPEG 2000 image and tile\n");
  fprintf(stdout,"headers of a sample frame.  Excluded: low-level packed-bits image data.\n\n");

  fprintf(stdout,"By Default\n");
  fprintf(stdout,"----------\n");
  fprintf(stdout,"The metadata includes the jp2 image and tile headers of the first frame.\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"Metadata values are shown in 'raw' form (e.g., hexidecimal) as stored in the\n");
  fprintf(stdout,"file, and, if apt, in a 'derived' form that is more quickly grasped.\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"Notes explaining the XML are embedded as terse comments.  These include\n");
  fprintf(stdout,"   meaning of non-obvious tag abbreviations;\n");
  fprintf(stdout,"   range and precision of valid values;\n");
  fprintf(stdout,"   interpretations of values, such as enumerations; and\n");
  fprintf(stdout,"   current implementation limitations.\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"The sample-size and chunk-offset tables, each with 1 row per frame, are not reported.\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"The file is self-contained and no verification (e.g., against a DTD) is requested.\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"Required Parameters (except with -h)\n");
  fprintf(stdout,"------------------------------------\n");
  fprintf(stdout,"[Caution: file strings that contain spaces should be wrapped with quotes.]\n");
  fprintf(stdout,"-i input.mj2  : where 'input' is any source file name or path.\n");
  fprintf(stdout,"                MJ2 files created with 'frames_to_mj2' are supported so far.\n");
  fprintf(stdout,"                These are silent, single-track, 'MJ2 Simple Profile' videos.\n");
  fprintf(stdout,"-o output.xml : where 'output' is any destination file name or path.\n");
  fprintf(stdout,"\n");
  fprintf(stdout,"Optional Parameters\n");
  fprintf(stdout,"-------------------\n");
  fprintf(stdout,"-h            : Display this help information.\n");
  fprintf(stdout,"-n            : Suppress all mj2_to_metadata notes.\n");
  fprintf(stdout,"-t            : Include sample-size and chunk-offset tables.\n");
  fprintf(stdout,"-f n          : where n > 0.  Include jp2 header info for frame n [default=1].\n");
  fprintf(stdout,"-f 0          : No jp2 header info.\n");
  fprintf(stdout,"-r            : Suppress all 'raw' data for which a 'derived' form exists.\n");
  fprintf(stdout,"-d            : Suppress all 'derived' data.\n");
  fprintf(stdout,"                (If both -r and -d given, -r will be ignored.)\n");
  fprintf(stdout,"-v string     : Verify against the DTD file located by the string.\n");
  fprintf(stdout,"                Prepend quoted 'string' with either SYSTEM or PUBLIC keyword.\n");
  fprintf(stdout,"                Thus, for the distributed DTD placed in the same directory as\n");
  fprintf(stdout,"                the output file: -v \"SYSTEM mj2_to_metadata.dtd\"\n");
  fprintf(stdout,"                \"PUBLIC\" is used with an access protocol (e.g., http:) + URL.\n");
  /* More to come */
  fprintf(stdout,"\n");
  /*             "1234567890123456789012345678901234567890123456789012345678901234567890123456789" */
}

/* ------------- */

int main(int argc, char *argv[]) {

	opj_dinfo_t* dinfo; 
	opj_event_mgr_t event_mgr;		/* event manager */

  FILE *file, *xmlout;
/*  char xmloutname[50]; */
  opj_mj2_t *movie;

  char* infile = 0;
  char* outfile = 0;
  char* s, S1, S2, S3;
  int len;
  unsigned int sampleframe = 1; /* First frame */
  char* stringDTD = NULL;
  BOOL notes = TRUE;
  BOOL sampletables = FALSE;
  BOOL raw = TRUE;
  BOOL derived = TRUE;
	mj2_dparameters_t parameters;

  while (TRUE) {
	/* ':' after letter means it takes an argument */
    int c = getopt(argc, argv, "i:o:f:v:hntrd");
	/* FUTURE:  Reserve 'p' for pruning file (which will probably make -t redundant) */
    if (c == -1)
      break;
    switch (c) {
    case 'i':			/* IN file */
      infile = optarg;
      s = optarg;
      while (*s) { s++; } /* Run to filename end */
      s--;
      S3 = *s;
      s--;
      S2 = *s;
      s--;
      S1 = *s;
      
      if ((S1 == 'm' && S2 == 'j' && S3 == '2')
      || (S1 == 'M' && S2 == 'J' && S3 == '2')) {
       break;
      }
      fprintf(stderr, "Input file name must have .mj2 extension, not .%c%c%c.\n", S1, S2, S3);
      return 1;

      /* ----------------------------------------------------- */
    case 'o':			/* OUT file */
      outfile = optarg;
      while (*outfile) { outfile++; } /* Run to filename end */
      outfile--;
      S3 = *outfile;
      outfile--;
      S2 = *outfile;
      outfile--;
      S1 = *outfile;
      
      outfile = optarg;
      
      if ((S1 == 'x' && S2 == 'm' && S3 == 'l')
	  || (S1 == 'X' && S2 == 'M' && S3 == 'L'))
        break;
    
      fprintf(stderr,
	  "Output file name must have .xml extension, not .%c%c%c\n", S1, S2, S3);
	  return 1;

      /* ----------------------------------------------------- */
    case 'f':			/* Choose sample frame.  0 = none */
      sscanf(optarg, "%u", &sampleframe);
      break;

      /* ----------------------------------------------------- */
    case 'v':			/* Verification by DTD. */
      stringDTD = optarg;
	  /* We will not insist upon last 3 chars being "dtd", since non-file
	  access protocol may be used. */
	  if(strchr(stringDTD,'"') != NULL) {
        fprintf(stderr, "-D's string must not contain any embedded double-quote characters.\n");
	    return 1;
	  }

      if (strncmp(stringDTD,"PUBLIC ",7) == 0 || strncmp(stringDTD,"SYSTEM ",7) == 0)
        break;
    
      fprintf(stderr, "-D's string must start with \"PUBLIC \" or \"SYSTEM \"\n");
	  return 1;

    /* ----------------------------------------------------- */
    case 'n':			/* Suppress comments */
      notes = FALSE;
      break;

    /* ----------------------------------------------------- */
    case 't':			/* Show sample size and chunk offset tables */
      sampletables = TRUE;
      break;

    /* ----------------------------------------------------- */
    case 'h':			/* Display an help description */
      help_display();
      return 0;

    /* ----------------------------------------------------- */
    case 'r':			/* Suppress raw data */
      raw = FALSE;
      break;

    /* ----------------------------------------------------- */
    case 'd':			/* Suppress derived data */
      derived = FALSE;
      break;

   /* ----------------------------------------------------- */
    default:
      return 1;
    } /* switch */
  } /* while */

  if(!raw && !derived)
	  raw = TRUE; /* At least one of 'raw' and 'derived' must be true */

    /* Error messages */
  /* -------------- */
  if (!infile || !outfile) {
    fprintf(stderr,"Correct usage: mj2_to_metadata -i mj2-file -o xml-file (plus options)\n");
    return 1;
  }

/* was:
  if (argc != 3) {
    printf("Bad syntax: Usage: MJ2_to_metadata inputfile.mj2 outputfile.xml\n"); 
    printf("Example: MJ2_to_metadata foreman.mj2 foreman.xml\n");
    return 1;
  }
*/
  len = strlen(infile);
  if(infile[0] == ' ')
  {
    infile++; /* There may be a leading blank if user put space after -i */
  }
  
  file = fopen(infile, "rb"); /* was: argv[1] */
  
  if (!file) {
    fprintf(stderr, "Failed to open %s for reading.\n", infile); /* was: argv[1] */
    return 1;
  }

  len = strlen(outfile);
  if(outfile[0] == ' ')
  {
    outfile++; /* There may be a leading blank if user put space after -o */
  }

  // Checking output file
  xmlout = fopen(outfile, "w"); /* was: argv[2] */
  if (!xmlout) {
    fprintf(stderr, "Failed to open %s for writing.\n", outfile); /* was: argv[2] */
    return 1;
  }
  // Leave it open

	/*
	configure the event callbacks (not required)
	setting of each callback is optionnal
	*/
	memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
	event_mgr.error_handler = error_callback;
	event_mgr.warning_handler = warning_callback;
	event_mgr.info_handler = info_callback;

	/* get a MJ2 decompressor handle */
	dinfo = mj2_create_decompress();

	/* catch events using our callbacks and give a local context */
	opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, stderr);		

	/* setup the decoder decoding parameters using user parameters */
	movie = (opj_mj2_t*) dinfo->mj2_handle;
	mj2_setup_decoder(dinfo->mj2_handle, &parameters);

  if (mj2_read_struct(file, movie)) // Creating the movie structure
  {
    fclose(xmlout);
    return 1;
  }

  xml_write_init(notes, sampletables, raw, derived);
  xml_write_struct(file, xmlout, movie, sampleframe, stringDTD, &event_mgr);
  fclose(xmlout);

	fprintf(stderr,"Metadata correctly extracted to XML file \n");;	

	/* free remaining structures */
	if(dinfo) {
		mj2_destroy_decompress((opj_mj2_t*)dinfo->mj2_handle);
	}

  return 0;
}


