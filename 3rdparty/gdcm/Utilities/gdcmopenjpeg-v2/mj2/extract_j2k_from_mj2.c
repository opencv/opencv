/*
 * Copyright (c) 2002-2007, Communications and Remote Sensing Laboratory, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2007, Professor Benoit Macq
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openjpeg.h"
#include "j2k.h"
#include "jp2.h"
#include "mj2.h"

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


int main(int argc, char *argv[]) {
  opj_dinfo_t* dinfo;
  opj_event_mgr_t event_mgr;    /* event manager */
  int tnum;
  unsigned int snum;
  opj_mj2_t *movie;
  mj2_tk_t *track;
  mj2_sample_t *sample;
  unsigned char* frame_codestream;
  FILE *file, *outfile;
  char outfilename[50];
  mj2_dparameters_t parameters;

  if (argc != 3) {
    printf("Bad syntax: Usage: MJ2_extractor mj2filename output_location\n");
    printf("Example: MJ2_extractor foreman.mj2 output/foreman\n");
    return 1;
  }

  file = fopen(argv[1], "rb");

  if (!file) {
    fprintf(stderr, "failed to open %s for reading\n", argv[1]);
    return 1;
  }

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
    return 1;

  // Decode first video track
  tnum = 0;
  while (movie->tk[tnum].track_type != 0)
    tnum ++;

  track = &movie->tk[tnum];

  fprintf(stdout,"Extracting %d frames from file...\n",track->num_samples);

  for (snum=0; snum < track->num_samples; snum++)
  {
    sample = &track->sample[snum];
    frame_codestream = (unsigned char*) malloc (sample->sample_size-8); // Skipping JP2C marker
    fseek(file,sample->offset+8,SEEK_SET);
    fread(frame_codestream,sample->sample_size-8,1, file);  // Assuming that jp and ftyp markers size do

    sprintf(outfilename,"%s_%05d.j2k",argv[2],snum);
    outfile = fopen(outfilename, "wb");
    if (!outfile) {
      fprintf(stderr, "failed to open %s for writing\n",outfilename);
      return 1;
    }
    fwrite(frame_codestream,sample->sample_size-8,1,outfile);
    fclose(outfile);
    free(frame_codestream);
    }
  fclose(file);
  fprintf(stdout, "%d frames correctly extracted\n", snum);

  /* free remaining structures */
  if(dinfo) {
    mj2_destroy_decompress((opj_mj2_t*)dinfo->mj2_handle);
  }

  return 0;
}
