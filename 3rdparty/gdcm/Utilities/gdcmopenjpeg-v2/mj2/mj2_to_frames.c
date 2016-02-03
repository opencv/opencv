/*
* Copyright (c) 2003-2004, François-Olivier Devaux
* Copyright (c) 2002-2004,  Communications and remote sensing Laboratory, Universite catholique de Louvain, Belgium
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
#include "j2k_lib.h"
#include "j2k.h"
#include "jp2.h"
#include "mj2.h"
#include "mj2_convert.h"

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
  mj2_dparameters_t mj2_parameters;      /* decompression parameters */
  opj_dinfo_t* dinfo;
  opj_event_mgr_t event_mgr;    /* event manager */
  opj_cio_t *cio = NULL;
  unsigned int tnum, snum;
  opj_mj2_t *movie;
  mj2_tk_t *track;
  mj2_sample_t *sample;
  unsigned char* frame_codestream;
  FILE *file, *outfile;
  char outfilename[50];
  opj_image_t *img = NULL;
  unsigned int max_codstrm_size = 0;
  double total_time = 0;
  unsigned int numframes = 0;

  if (argc != 3) {
    printf("Bad syntax: Usage: mj2_to_frames inputfile.mj2 outputfile.yuv\n");
    printf("Example: MJ2_decoder foreman.mj2 foreman.yuv\n");
    return 1;
  }

  file = fopen(argv[1], "rb");

  if (!file) {
    fprintf(stderr, "failed to open %s for reading\n", argv[1]);
    return 1;
  }

  // Checking output file
  outfile = fopen(argv[2], "w");
  if (!file) {
    fprintf(stderr, "failed to open %s for writing\n", argv[2]);
    return 1;
  }
  fclose(outfile);

  /*
  configure the event callbacks (not required)
  setting of each callback is optionnal
  */
  memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
  event_mgr.error_handler = error_callback;
  event_mgr.warning_handler = warning_callback;
  event_mgr.info_handler = NULL;

  /* get a MJ2 decompressor handle */
  dinfo = mj2_create_decompress();
  movie = dinfo->mj2_handle;

  /* catch events using our callbacks and give a local context */
  opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, stderr);

  /* set J2K decoding parameters to default values */
  opj_set_default_decoder_parameters(&mj2_parameters.j2k_parameters);

  /* setup the decoder decoding parameters using user parameters */
  mj2_setup_decoder(dinfo->mj2_handle, &mj2_parameters);

  if (mj2_read_struct(file, movie)) // Creating the movie structure
    return 1;

  // Decode first video track
  for (tnum=0; tnum < (unsigned int)(movie->num_htk + movie->num_stk + movie->num_vtk); tnum++) {
    if (movie->tk[tnum].track_type == 0)
      break;
  }

  if (movie->tk[tnum].track_type != 0) {
    printf("Error. Movie does not contain any video track\n");
    return 1;
  }

  track = &movie->tk[tnum];

  // Output info on first video tracl
  fprintf(stdout,"The first video track contains %d frames.\nWidth: %d, Height: %d \n\n",
    track->num_samples, track->w, track->h);

  max_codstrm_size = track->sample[0].sample_size-8;
  frame_codestream = (unsigned char*) malloc(max_codstrm_size * sizeof(unsigned char));

  numframes = track->num_samples;

  for (snum=0; snum < numframes; snum++)
  {
    double init_time = opj_clock();
    double elapsed_time;

    sample = &track->sample[snum];
    if (sample->sample_size-8 > max_codstrm_size) {
      max_codstrm_size =  sample->sample_size-8;
      if ((frame_codestream = realloc(frame_codestream, max_codstrm_size)) == NULL) {
        printf("Error reallocation memory\n");
        return 1;
      };
    }
    fseek(file,sample->offset+8,SEEK_SET);
    fread(frame_codestream, sample->sample_size-8, 1, file);  // Assuming that jp and ftyp markers size do

    /* open a byte stream */
    cio = opj_cio_open((opj_common_ptr)dinfo, frame_codestream, sample->sample_size-8);

    img = opj_decode(dinfo, cio); // Decode J2K to image

    if (((img->numcomps == 3) && (img->comps[0].dx == img->comps[1].dx / 2)
      && (img->comps[0].dx == img->comps[2].dx / 2 ) && (img->comps[0].dx == 1))
      || (img->numcomps == 1)) {

      if (!imagetoyuv(img, argv[2]))  // Convert image to YUV
        return 1;
    }
    else if ((img->numcomps == 3) &&
      (img->comps[0].dx == 1) && (img->comps[1].dx == 1)&&
      (img->comps[2].dx == 1))// If YUV 4:4:4 input --> to bmp
    {
      fprintf(stdout,"The frames will be output in a bmp format (output_1.bmp, ...)\n");
      sprintf(outfilename,"output_%d.bmp",snum);
      if (imagetobmp(img, outfilename))  // Convert image to BMP
        return 1;

    }
    else {
      fprintf(stdout,"Image component dimensions are unknown. Unable to output image\n");
      fprintf(stdout,"The frames will be output in a j2k file (output_1.j2k, ...)\n");

      sprintf(outfilename,"output_%d.j2k",snum);
      outfile = fopen(outfilename, "wb");
      if (!outfile) {
        fprintf(stderr, "failed to open %s for writing\n",outfilename);
        return 1;
      }
      fwrite(frame_codestream,sample->sample_size-8,1,outfile);
      fclose(outfile);
    }
    /* close the byte stream */
    opj_cio_close(cio);
    /* free image data structure */
    opj_image_destroy(img);
    elapsed_time = opj_clock()-init_time;
    fprintf(stderr, "Frame number %d/%d decoded in %.2f mseconds\n", snum + 1, numframes, elapsed_time*1000);
    total_time += elapsed_time;

  }

  free(frame_codestream);
  fclose(file);

  /* free remaining structures */
  if(dinfo) {
    mj2_destroy_decompress((opj_mj2_t*)dinfo->mj2_handle);
  }
  free(dinfo);

  fprintf(stdout, "%d frame(s) correctly decompressed\n", snum);
  fprintf(stdout,"Total decoding time: %.2f seconds (%.1f fps)\n", total_time, (float)numframes/total_time);

  return 0;
}
