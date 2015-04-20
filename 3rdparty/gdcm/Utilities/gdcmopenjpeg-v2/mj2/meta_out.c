/* meta_out.c */
/* Dump MJ2, JP2 metadata (partial so far) to xml file */
/* Callable from mj2_to_metadata */
/* Contributed to Open JPEG by Glenn Pearson, contract software developer, U.S. National Library of Medicine.

The base code in this file was developed by the author as part of a video archiving
project for the U.S. National Library of Medicine, Bethesda, MD.
It is the policy of NLM (and U.S. government) to not assert copyright.

A non-exclusive copy of this code has been contributed to the Open JPEG project.
Except for copyright, inclusion of the code within Open JPEG for distribution and use
can be bound by the Open JPEG open-source license and disclaimer, expressed elsewhere.
*/

#include <windows.h> /* for time functions */

#include "opj_includes.h"
#include "mj2.h"

#include <time.h>
#include "meta_out.h"

static BOOL notes = TRUE;
static BOOL sampletables = FALSE;
static BOOL raw = TRUE;
static BOOL derived = TRUE;

opj_tcp_t *j2k_default_tcp;

/* Forwards */
int xml_write_overall_header(FILE *file, FILE *xmlout, opj_mj2_t * movie, unsigned int sampleframe, opj_event_mgr_t *event_mgr);
int xml_write_moov(FILE *file, FILE *xmlout, opj_mj2_t * movie, unsigned int sampleframe, opj_event_mgr_t *event_mgr);

void uint_to_chars(unsigned int value, char* buf);

void xml_write_trak(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum, unsigned int sampleframe, opj_event_mgr_t *event_mgr);
void xml_write_tkhd(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum);
void xml_write_udta(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum);
void xml_write_mdia(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum);
void xml_write_stbl(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum);

void UnixTimeToFileTime(time_t t, LPFILETIME pft);
void UnixTimeToSystemTime(time_t t, LPSYSTEMTIME pst);
void xml_time_out(FILE* xmlout, time_t t);

void int16_to_3packedchars(short int value, char* buf);

void xml_write_moov_udta(FILE* xmlout, opj_mj2_t * movie);
void xml_write_free_and_skip(FILE* xmlout, opj_mj2_t * movie);
void xml_write_uuid(FILE* xmlout, opj_mj2_t * movie);

int xml_out_frame(FILE* file, FILE* xmlout, mj2_sample_t *sample, unsigned int snum, opj_event_mgr_t *event_mgr);

void xml_out_frame_siz(FILE* xmlout, opj_image_t *img, opj_cp_t *cp);
void xml_out_frame_cod(FILE* xmlout, opj_tcp_t *tcp);
void xml_out_frame_coc(FILE* xmlout, opj_tcp_t *tcp, int numcomps); /* opj_image_t *img); */
BOOL same_component_style(opj_tccp_t *tccp1, opj_tccp_t *tccp2);
void xml_out_frame_qcd(FILE* xmlout, opj_tcp_t *tcp);
void xml_out_frame_qcc(FILE* xmlout, opj_tcp_t *tcp, int numcomps); /* opj_image_t *img); */
BOOL same_component_quantization(opj_tccp_t *tccp1, opj_tccp_t *tccp2);
void xml_out_frame_rgn(FILE* xmlout, opj_tcp_t *tcp, int numcomps);/* opj_image_t *img);*/
void xml_out_frame_poc(FILE* xmlout, opj_tcp_t *tcp);
void xml_out_frame_ppm(FILE* xmlout, opj_cp_t *cp);
void xml_out_frame_ppt(FILE* xmlout, opj_tcp_t *tcp);
void xml_out_frame_tlm(FILE* xmlout); /* j2k_default_tcp is passed globally */ /* NO-OP.  TLM NOT SAVED IN DATA STRUCTURE */
void xml_out_frame_plm(FILE* xmlout); /* j2k_default_tcp is passed globally */ /* NO-OP.  PLM NOT SAVED IN DATA STRUCTURE.  opt in main; can be used in conjunction with PLT */
void xml_out_frame_plt(FILE* xmlout, opj_tcp_t *tcp); /* NO-OP.  PLM NOT SAVED IN DATA STRUCTURE.  opt in main; can be used in conjunction with PLT */
void xml_out_frame_crg(FILE* xmlout); /* j2k_default_tcp is passed globally */ /* opt in main; */
void xml_out_frame_com(FILE* xmlout, opj_tcp_t *tcp); /* NO-OP.  COM NOT SAVED IN DATA STRUCTURE */ /* opt in main; */
void xml_out_dump_hex(FILE* xmlout, char *data, int data_len, char* s);
void xml_out_dump_hex_and_ascii(FILE* xmlout, char *data, int data_len, char* s);
void xml_out_frame_jp2h(FILE* xmlout, opj_jp2_t *jp2_struct);
#ifdef NOTYET
/* Shown with cp, extended, as data structure... but it could be a new different one */
void xml_out_frame_jp2i(FILE* xmlout, opj_cp_t *cp);/* IntellectualProperty 'jp2i' (no restrictions on location) */
void xml_out_frame_xml(FILE* xmlout, opj_cp_t *cp); /* XML 'xml\040' (0x786d6c20).  Can appear multiply */
void xml_out_frame_uuid(FILE* xmlout, opj_cp_t *cp); /* UUID 'uuid' (top level only) */
void xml_out_frame_uinf(FILE* xmlout, opj_cp_t *cp); /* UUIDInfo 'uinf', includes UUIDList 'ulst' and URL 'url\40' */
void xml_out_frame_unknown_type(FILE* xmlout, opj_cp_t *cp);
#endif


void xml_write_init(BOOL n, BOOL t, BOOL r, BOOL d)
{
  /* Init file globals */
  notes = n;
  sampletables = t;
  raw = r;
  derived = d;
}

int xml_write_struct(FILE* file, FILE *xmlout, opj_mj2_t * movie, unsigned int sampleframe, char* stringDTD, opj_event_mgr_t *event_mgr) {

  if(stringDTD != NULL)
  {
    fprintf(xmlout,"<?xml version=\"1.0\" standalone=\"no\"?>\n");
  /* stringDTD is known to start with "SYSTEM " or "PUBLIC " */
  /* typical: SYSTEM mj2_to_metadata.dtd */
  stringDTD[6] = '\0'; /* Break into two strings at space, so quotes can be inserted. */
    fprintf(xmlout,"<!DOCTYPE MJ2_File %s \"%s\">\n", stringDTD, stringDTD+7);
  stringDTD[6] = ' '; /* restore for sake of debugger or memory allocator */
  } else
    fprintf(xmlout,"<?xml version=\"1.0\" standalone=\"yes\"?>\n");

  fprintf(xmlout, "<MJ2_File>\n");
  xml_write_overall_header(file, xmlout, movie, sampleframe, event_mgr);
  fprintf(xmlout, "</MJ2_File>");
  return 0;
}

/* ------------- */

int xml_write_overall_header(FILE *file, FILE *xmlout, opj_mj2_t * movie, unsigned int sampleframe, opj_event_mgr_t *event_mgr)
{
  int i;
  char buf[5];
  buf[4] = '\0';

  fprintf(xmlout,   "  <JP2 BoxType=\"jP[space][space]\" Signature=\"0x0d0a870a\" />\n");
  // Called after structure initialized by mj2_read_ftyp
  fprintf(xmlout,   "  <FileType BoxType=\"ftyp\">\n");
  uint_to_chars(movie->brand, buf);
  fprintf(xmlout,   "    <Brand>%s</Brand>\n", buf);    /* 4 character; BR              */
  fprintf(xmlout,   "    <MinorVersion>%u</MinorVersion>\n", movie->minversion);    /* 4 char; MinV            */
  fprintf(xmlout,   "    <CompatibilityList Count=\"%d\">\n",movie->num_cl);
  for (i = movie->num_cl - 1; i > -1; i--) /* read routine stored in reverse order, so let's undo damage */
  {
    uint_to_chars(movie->cl[i], buf);
    fprintf(xmlout, "      <CompatibleBrand>%s</CompatibleBrand>\n", buf);    /*4 characters, each CLi */
  }
  fprintf(xmlout,   "    </CompatibilityList>\n");
  fprintf(xmlout,   "  </FileType>\n");
  xml_write_moov(file, xmlout, movie, sampleframe, event_mgr);
  // To come?              <mdat>  // This is the container for media data that can also be accessed through track structures,
                                   // so is redundant, and simply not of interest as metadata
  //                       <moof>  // Allows incremental build up of movie.  Probably not in Simple Profile
  xml_write_free_and_skip(xmlout, movie); /* NO OP so far */ /* May be a place where user squirrels metadata */
  xml_write_uuid(xmlout, movie); /* NO OP so far */ /* May be a place where user squirrels metadata */
  return 0;
}

/* ------------- */

int xml_write_moov(FILE *file, FILE *xmlout, opj_mj2_t * movie, unsigned int sampleframe, opj_event_mgr_t *event_mgr)
{
  unsigned int tnum;
  mj2_tk_t *track;

  fprintf(xmlout,   "  <MovieBox BoxType=\"moov\">\n");
  fprintf(xmlout,   "    <MovieHeader BoxType=\"mvhd\">\n");
  fprintf(xmlout,   "      <CreationTime>\n");
  if(raw)
    fprintf(xmlout, "        <InSeconds>%u</InSeconds>\n", movie->creation_time);
  if(notes)
    fprintf(xmlout, "        <!-- Seconds since start of Jan. 1, 1904 UTC (Greenwich) -->\n");
  /*  2082844800 = seconds between 1/1/04 and 1/1/70 */
  /* There's still a time zone offset problem not solved... but spec is ambigous as to whether stored time
     should be local or UTC */
  if(derived) {
    fprintf(xmlout, "        <AsLocalTime>");
                             xml_time_out(xmlout, movie->creation_time - 2082844800);
                                                     fprintf(xmlout,"</AsLocalTime>\n");
  }
  fprintf(xmlout,   "      </CreationTime>\n");
  fprintf(xmlout,   "      <ModificationTime>\n");
  if(raw)
    fprintf(xmlout, "        <InSeconds>%u</InSeconds>\n", movie->modification_time);
  if(derived) {
    fprintf(xmlout, "        <AsLocalTime>");
                             xml_time_out(xmlout, movie->modification_time - 2082844800);
                                                     fprintf(xmlout,"</AsLocalTime>\n");
  }
  fprintf(xmlout,   "      </ModificationTime>\n");
  fprintf(xmlout,   "      <Timescale>%d</Timescale>\n", movie->timescale);
  if(notes)
    fprintf(xmlout, "      <!-- Timescale defines time units in one second -->\n");
  fprintf(xmlout,   "      <Rate>\n");        /* Rate to play presentation  (default = 0x00010000)          */
  if(notes) {
    fprintf(xmlout, "      <!-- Rate to play presentation is stored as fixed-point binary 16.16 value. Decimal value is approximation. -->\n");
    fprintf(xmlout, "      <!-- Rate is expressed relative to normal (default) value of 0x00010000 (1.0) -->\n");
  }
  if(raw)
    fprintf(xmlout, "        <AsHex>0x%08x</AsHex>\n", movie->rate);
  if(derived)
    fprintf(xmlout, "        <AsDecimal>%12.6f</AsDecimal>\n", (double)movie->rate/(double)0x00010000);
  fprintf(xmlout,   "      </Rate>\n");
  fprintf(xmlout,   "      <Duration>\n");
  if(raw)
    fprintf(xmlout, "        <InTimeUnits>%u</InTimeUnits>\n", movie->duration);
  if(derived)
    fprintf(xmlout, "        <InSeconds>%12.3f</InSeconds>\n", (double)movie->duration/(double)movie->timescale);    // Make this double later to get fractional seconds
  fprintf(xmlout,   "      </Duration>\n");
#ifdef CURRENTSTRUCT
  movie->volume = movie->volume << 8;
#endif
  fprintf(xmlout,   "      <Volume>\n");
  if(notes) {
    fprintf(xmlout, "      <!-- Audio volume stored as fixed-point binary 8.8 value. Decimal value is approximation. -->\n");
    fprintf(xmlout, "      <!-- Full, normal (default) value is 0x0100 (1.0) -->\n");
  }
  if(raw)
    fprintf(xmlout, "        <AsHex>0x%04x</AsHex>\n", movie->volume);
  if(derived)
    fprintf(xmlout, "        <AsDecimal>%6.3f</AsDecimal>\n", (double)movie->volume/(double)0x0100);
  fprintf(xmlout,   "      </Volume>\n");
#ifdef CURRENTSTRUCT
  if(notes)
    fprintf(xmlout, "      <!-- Current m2j_to_metadata implementation always shows bits to right of decimal as zeroed. -->\n");
  movie->volume = movie->volume >> 8;
#endif
  /* Transformation matrix for video                            */
  fprintf(xmlout,   "      <TransformationMatrix>\n");
  if(notes) {
    fprintf(xmlout, "      <!-- 3 x 3 Video Transformation Matrix {a,b,u,c,d,v,x,y,w}.  Required: u=0, v=0, w=1 -->\n");
    fprintf(xmlout, "      <!-- Maps decompressed point (p,q) to rendered point (ap + cq + x, bp + dq + y) -->\n");
    fprintf(xmlout, "      <!-- Stored as Fixed Point Hex: all are binary 16.16, except u,v,w are 2.30 -->\n");
    fprintf(xmlout, "      <!-- Unity = 0x00010000,0,0,0,0x00010000,0,0,0,0x40000000 -->\n");
  }
  fprintf(xmlout,   "        <TMa>0x%08x</TMa>\n", movie->trans_matrix[0]);
  fprintf(xmlout,   "        <TMb>0x%08x</TMb>\n", movie->trans_matrix[1]);
  fprintf(xmlout,   "        <TMu>0x%08x</TMu>\n", movie->trans_matrix[2]);
  fprintf(xmlout,   "        <TMc>0x%08x</TMc>\n", movie->trans_matrix[3]);
  fprintf(xmlout,   "        <TMd>0x%08x</TMd>\n", movie->trans_matrix[4]);
  fprintf(xmlout,   "        <TMv>0x%08x</TMv>\n", movie->trans_matrix[5]);
  fprintf(xmlout,   "        <TMx>0x%08x</TMx>\n", movie->trans_matrix[6]);
  fprintf(xmlout,   "        <TMy>0x%08x</TMy>\n", movie->trans_matrix[7]);
  fprintf(xmlout,   "        <TMw>0x%08x</TMw>\n", movie->trans_matrix[8]);
  fprintf(xmlout,   "      </TransformationMatrix>\n");
  fprintf(xmlout,   "    </MovieHeader>\n");

  fprintf(xmlout,   "    <Statistics>\n");
  fprintf(xmlout,   "      <TracksFound>\n");
  fprintf(xmlout,   "        <Video>%d</Video>\n", movie->num_vtk);
  fprintf(xmlout,   "        <Audio>%d</Audio>\n", movie->num_stk);
  fprintf(xmlout,   "        <Hint>%d</Hint>\n", movie->num_htk);
  if(notes)
    fprintf(xmlout, "        <!-- Hint tracks for streaming video are not part of MJ2, but are a defined extension. -->\n");
  /* See Part 3 Amend 2 Section 4.2 for relation of MJ2 to Part 12 Sections 7 and 10 hints */
  fprintf(xmlout,   "      </TracksFound>\n");
  fprintf(xmlout,   "    </Statistics>\n");
  /* Idea for the future:  It would be possible to add code to verify that the file values:
    1) are legal and self-consistent
  2) comply with particular JP2 and/or MJ2 profiles.
  This could be reported here as additional XML elements */

  // Find first video track
  tnum = 0;
  while (movie->tk[tnum].track_type != 0)
    tnum ++;

  track = &(movie->tk[tnum]);
  // For now, output info on first video track
  xml_write_trak(file, xmlout, track, tnum, sampleframe, event_mgr);

  // to come:                <MovieExtends mvek> // possibly not in Simple Profile
  xml_write_moov_udta(xmlout, movie); /* NO OP so far */ /* <UserDataBox udta> contains <CopyrightBox cprt> */
  fprintf(xmlout,   "  </MovieBox>\n");
  return 0;
}

/* --------------- */

void uint_to_chars(unsigned int value, char* buf)
{
  /* buf is at least char[5] */
    int i;
    for (i = 3; i >= 0; i--)
    {
        buf[i] = (value & 0x000000ff);
        value = (value >> 8);
    }
  buf[4] = '\0'; /* Precautionary */
}

/* ------------- */

/* WINDOWS SPECIFIC */

void UnixTimeToFileTime(time_t t, LPFILETIME pft)
{
  /* Windows specific.  From MS Q167296 */
  /* 'time_t' represents seconds since midnight January 1, 1970 UTC (coordinated universal time). */
  /* 64-bit FILETIME structure represents the number of 100-nanosecond intervals since January 1, 1601 UTC (coordinate universal time). */
  LONGLONG ll; /* LONGLONG is a 64-bit value. */
  ll = Int32x32To64(t, 10000000) + 116444736000000000;
  pft->dwLowDateTime = (DWORD)ll;
  /* pft->dwLowDateTime = (DWORD)(0x00000000ffffffff & ll); */
  pft->dwHighDateTime = (DWORD)(ll >> 32);
}
// Once the UNIX time is converted to a FILETIME structure,
// other Win32 time formats can be easily obtained by using Win32 functions such
// as FileTimeToSystemTime() and FileTimeToDosDateTime().

/* ------------- */

void UnixTimeToSystemTime(time_t t, LPSYSTEMTIME pst)
{
  /* Windows specific */
  FILETIME ft;
  UnixTimeToFileTime(t, &ft);
  FileTimeToLocalFileTime( &ft, &ft ); /* Adjust from UTC to local time zone */
  FileTimeToSystemTime(&ft, pst);
}

/* ------------- */

void xml_time_out(FILE* xmlout, time_t t)
{
  /* Windows specific */
  SYSTEMTIME st;
  char szLocalDate[255], szLocalTime[255];
  UnixTimeToSystemTime( t, &st );
  GetDateFormat( LOCALE_USER_DEFAULT, DATE_LONGDATE, &st, NULL, szLocalDate, 255 );
  GetTimeFormat( LOCALE_USER_DEFAULT, 0, &st, NULL, szLocalTime, 255 );
  fprintf(xmlout, "%s %s", szLocalDate, szLocalTime );
}

/* END WINDOWS SPECIFIC */

/* ------------- */

void xml_write_moov_udta(FILE* xmlout, opj_mj2_t * movie) {
  /* Compare with xml_write_udta */
#ifdef NOTYET
  /* NO-OP so far.  Optional UserData 'udta' (zero or one in moov or each trak)
     can contain multiple Copyright 'cprt' with different language codes */
  /* There may be nested non-standard boxes within udta */
  IMAGINE movie->udta, movie->copyright_count, movie->copyright_language[i] (array of 16bit ints), movie->copyright_notice[i] (array of buffers)
  PROBABLY ALSO NEED movie->udta_len or special handler for non-standard boxes
  char buf[5];
  int i;

  if(movie->udta != 1)
    return; /* Not present */

  fprintf(xmlout,    "    <UserData BoxType=\"udta\">\n");
  for(i = 0; i < movie->copyright_count; i++) {
    fprintf(xmlout,  "      <Copyright BoxType=\"cprt\"> Instance=\"%d\">\n", i+1);
    int16_to_3packedchars((short int)movie->copyright_languages[i], buf);
    fprintf(xmlout,  "        <Language>%s</Language>\n", buf);    /* 3 chars */
    fprintf(xmlout,  "        <Notice>%s</Notice>\n",movie->copyright_notices[i]);
    fprintf(xmlout,  "      </Copyright>\n", i+1);
  }
  /* TO DO: Non-standard boxes */
  fprintf(xmlout,    "    </UserData>\n");
#endif
}

void xml_write_free_and_skip(FILE* xmlout, opj_mj2_t * movie) {
#ifdef NOTYET
  /* NO-OP so far.  There can be zero or more instances of free and/or skip
     at the top level of the file.  This may be a place where the user squirrel's metadata.
   Let's assume unstructured, and do a dump */
  IMAGINE movie->free_and_skip, movie->free_and_skip_count, movie->free_and_skip_content[i] (array of buffers),
    movie->free_and_skip_len[i] (array of ints), movie->is_skip[i] (array of BOOL)
  int i;

  if(movie->free_and_skip != 1)
    return; /* Not present */

  for(i = 0; i < movie->free_and_skip_count; i++) {
    if(movie->is_skip[i])
      fprintf(xmlout,    "  <Skip BoxType=\"skip\">\n");
  else
      fprintf(xmlout,    "  <Free BoxType=\"free\">\n");

    xml_out_dump_hex_and_ascii(xmlout, movie->free_and_skip_contents[i], movie->free_and_skip_len[i]);

    if(movie->is_skip[i])
      fprintf(xmlout,    "  </Skip>\n");
  else
      fprintf(xmlout,    "  </Free>\n");
  }
#endif
}

void xml_write_uuid(FILE* xmlout, opj_mj2_t * movie) {
/* Univeral Unique IDs of 16 bytes.  */
#ifdef NOTYET
  /* NO-OP so far.  There can be zero or more instances of private uuid boxes in a file.
     This function supports the top level of the file, but uuid may be elsewhere [not yet supported].
   This may be a place where the user squirrel's metadata.  Let's assume unstructured, and do a dump */
  IMAGINE movie->uuid, movie->uuid_count, movie->uuid_content[i] (array of buffers),
    movie->uuid_len[i] (array of ints), movie->uuid_type[i] (array of 17-byte (16+null termination) buffers)
  int i;

  if(movie->uuid != 1)
    return; /* Not present */

  for(i = 0; i < movie->uuid_count; i++) {
    fprintf(xmlout,    "  <PrivateExtension BoxType=\"uuid\" UUID=\"%s\">\n", movie->uuid_type[i]);
  // See Part III section 5.2.1, 6.1, 6.2
    xml_out_dump_hex_and_ascii(xmlout, movie->uuid_contents[i], movie->uuid_len[i]);
    fprintf(xmlout,    "  </PrivateExtension>\n");
  }
#endif
}

/* ------------- */

void xml_write_trak(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum, unsigned int sampleframe, opj_event_mgr_t *event_mgr)
{
  fprintf(xmlout,    "    <Track BoxType=\"trak\" Instance=\"%d\">\n", tnum);
  xml_write_tkhd(file, xmlout, track, tnum);
  // TO DO: TrackReferenceContainer 'tref'  just used in hint track
  // TO DO: EditListContainer 'edts', contains EditList 'elst' with media-time, segment-duration, media-rate
  xml_write_mdia(file, xmlout, track, tnum);
  xml_write_udta(file, xmlout, track, tnum); // NO-OP so far.  Optional UserData 'udta', can contain multiple Copyright 'cprt'

  if(track->track_type==0) { /* Only do for visual track */
  /* sampleframe is from user option -f.  1 = first frame */
    /* sampleframe of 0 is a user requests: no jp2 header */
  /* Treat out-of-bounds values in the same way */
  if(sampleframe > 0 && sampleframe <= track->num_samples)
    {
      mj2_sample_t *sample;
      unsigned int snum;

      snum = sampleframe-1;
      // Someday maybe do a smart range scan... for (snum=0; snum < track->num_samples; snum++){
      //  fprintf(stdout,"Frame %d: ",snum+1);
      sample = &track->sample[snum];
    if(xml_out_frame(file, xmlout, sample, snum, event_mgr))
      return; /* Not great error handling here */
    }
  }
  fprintf(xmlout,    "    </Track>\n");
}

/* ------------- */

void xml_write_tkhd(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum)
{
  fprintf(xmlout,    "      <TrackHeader BoxType=\"tkhd\">\n");
  if(notes) {
    fprintf(xmlout,  "      <!-- Not shown here: CreationTime, ModificationTime, Duration. -->\n");
    fprintf(xmlout,  "      <!-- These 3 fields are reported under MediaHeader below.   When reading these 3, -->\n");
    fprintf(xmlout,  "      <!-- m2j_to_metadata currently doesn't distinguish between TrackHeader and MediaHeader source. -->\n");
    fprintf(xmlout,  "      <!-- If both found, value read from MediaHeader is used. -->\n");
  }
  fprintf(xmlout,    "        <TrackID>%u</TrackID>\n", track->track_ID);
  if(track->track_type==0) /* For visual track */
  {
    fprintf(xmlout,  "        <TrackLayer>%d</TrackLayer>\n", track->layer);
    if(notes)
      fprintf(xmlout,"        <!-- front-to-back ordering of video tracks. 0 = normal, -1 is closer, etc. -->\n");
  }
  if(track->track_type!=0) /* volume irrelevant for visual track */
  {
#ifdef CURRENTSTRUCT
    track->volume = track->volume << 8;
#endif
    fprintf(xmlout,  "        <Volume>\n");
  if(notes) {
      fprintf(xmlout,"          <!-- Track audio volume stored as fixed-point binary 8.8 value. Decimal value is approximation. -->\n");
      fprintf(xmlout,"          <!-- Full, normal (default) value is 0x0100 (1.0) -->\n");
  }
  if(raw)
      fprintf(xmlout,"          <AsHex>0x%04x</AsHex>\n", track->volume);
  if(derived)
      fprintf(xmlout,"          <AsDecimal>%6.3f</AsDecimal>\n", (double)track->volume/(double)0x0100);
    fprintf(xmlout,  "        </Volume>\n");
#ifdef CURRENTSTRUCT
  if(notes)
    fprintf(xmlout,  "        <!-- Current m2j_to_metadata implementation always shows bits to right of decimal as zeroed. -->\n");
  track->volume = track->volume >> 8;
#endif
  }
  if(track->track_type==0)
  {
    /* Transformation matrix for video                            */
    fprintf(xmlout,  "        <TransformationMatrix>\n");
  if(notes) {
      fprintf(xmlout,"          <!-- Comments about matrix in MovieHeader apply here as well. -->\n");
      fprintf(xmlout,"          <!-- This matrix is applied before MovieHeader one. -->\n");
  }
    fprintf(xmlout,  "          <TMa>0x%08x</TMa>\n", track->trans_matrix[0]);
    fprintf(xmlout,  "          <TMb>0x%08x</TMb>\n", track->trans_matrix[1]);
    fprintf(xmlout,  "          <TMu>0x%08x</TMu>\n", track->trans_matrix[2]);
    fprintf(xmlout,  "          <TMc>0x%08x</TMc>\n", track->trans_matrix[3]);
    fprintf(xmlout,  "          <TMd>0x%08x</TMd>\n", track->trans_matrix[4]);
    fprintf(xmlout,  "          <TMv>0x%08x</TMv>\n", track->trans_matrix[5]);
    fprintf(xmlout,  "          <TMx>0x%08x</TMx>\n", track->trans_matrix[6]);
    fprintf(xmlout,  "          <TMy>0x%08x</TMy>\n", track->trans_matrix[7]);
    fprintf(xmlout,  "          <TMw>0x%08x</TMw>\n", track->trans_matrix[8]);
    fprintf(xmlout,  "        </TransformationMatrix>\n");
  }
#ifdef CURRENTSTRUCT
  track->w = track->w << 16;
  track->h = track->h << 16;
#endif
  if(notes) {
    fprintf(xmlout,  "        <!-- Width and Height in pixels are for the presentation; frames will be scaled to this. -->\n");
    fprintf(xmlout,  "        <!-- Both stored as fixed-point binary 16.16 values. Decimal values are approximations. -->\n");
  }
  fprintf(xmlout,    "        <Width>\n");
  if(raw)
    fprintf(xmlout,  "          <AsHex>0x%08x</AsHex>\n", track->w);
  if(derived)
    fprintf(xmlout,  "          <AsDecimal>%12.6f</AsDecimal>\n", (double)track->w/(double)0x00010000);        /* Rate to play presentation  (default = 0x00010000)          */
  fprintf(xmlout,    "        </Width>\n");
  fprintf(xmlout,    "        <Height>\n");
  if(raw)
    fprintf(xmlout,  "          <AsHex>0x%08x</AsHex>\n", track->h);
  if(derived)
    fprintf(xmlout,  "          <AsDecimal>%12.6f</AsDecimal>\n", (double)track->h/(double)0x00010000);        /* Rate to play presentation  (default = 0x00010000)          */
  fprintf(xmlout,    "        </Height>\n");
#ifdef CURRENTSTRUCT
  if(notes) {
    fprintf(xmlout,  "        <!-- Current m2j_to_metadata implementation always shows bits to right of decimal as zeroed. -->\n");
    fprintf(xmlout,  "        <!-- Also, width and height values shown here will actually be those read from track's <VisualSampleEntry> if given. -->\n");
  }
  track->w = track->w >> 16;
  track->h = track->h >> 16;
#endif
  fprintf(xmlout,    "      </TrackHeader>\n");
}

/* ------------- */

void xml_write_udta(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum) {
  /* NO-OP so far.  Optional UserData 'udta' (zero or one in moov or each trak)
     can contain multiple Copyright 'cprt' with different language codes */
  /* There may be nested non-standard boxes within udta */
#ifdef NOTYET
  IMAGINE track->udta, track->copyright_count, track->copyright_language[i] (array of 16bit ints), track->copyright_notice[i] (array of buffers)
  PROBABLY ALSO NEED track->udta_len or special handler for non-standard boxes
  char buf[5];
  int i;

  if(track->udta != 1)
    return; /* Not present */

  fprintf(xmlout,    "      <UserData BoxType=\"udta\">\n");
  for(i = 0; i < track->copyright_count; i++) {
    fprintf(xmlout,  "        <Copyright BoxType=\"cprt\"> Instance=\"%d\">\n", i+1);
    int16_to_3packedchars((short int)track->copyright_languages[i], buf);
    fprintf(xmlout,  "          <Language>%s</Language>\n", buf);    /* 3 chars */
    fprintf(xmlout,  "          <Notice>%s</Notice>\n",track->copyright_notices[i]);
    fprintf(xmlout,  "        </Copyright>\n", i+1);
  }
  /* TO DO: Non-standard boxes */
  fprintf(xmlout,    "      </UserData>\n");
#endif
}

/* ------------- */

void xml_write_mdia(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum)
{
  char buf[5];
  int i, k;
  buf[4] = '\0';

  fprintf(xmlout,    "      <Media BoxType=\"mdia\">\n");
  fprintf(xmlout,    "        <MediaHeader BoxType=\"mdhd\">\n");
  fprintf(xmlout,    "          <CreationTime>\n");
  if(raw)
    fprintf(xmlout,  "            <InSeconds>%u</InSeconds>\n", track->creation_time);
  if(notes)
    fprintf(xmlout,  "            <!-- Seconds since start of Jan. 1, 1904 UTC (Greenwich) -->\n");
  /*  2082844800 = seconds between 1/1/04 and 1/1/70 */
  /* There's still a time zone offset problem not solved... but spec is ambigous as to whether stored time
     should be local or UTC */
  if(derived) {
    fprintf(xmlout,  "            <AsLocalTime>");
                                xml_time_out(xmlout, track->creation_time - 2082844800);
                                                     fprintf(xmlout,"</AsLocalTime>\n");
  }
  fprintf(xmlout,    "          </CreationTime>\n");
  fprintf(xmlout,    "          <ModificationTime>\n");
  if(raw)
    fprintf(xmlout,  "            <InSeconds>%u</InSeconds>\n", track->modification_time);
  if(derived) {
    fprintf(xmlout,  "            <AsLocalTime>");
                                xml_time_out(xmlout, track->modification_time - 2082844800);
                                                     fprintf(xmlout,"</AsLocalTime>\n");
  }
  fprintf(xmlout,    "          </ModificationTime>\n");
  fprintf(xmlout,    "          <Timescale>%d</Timescale>\n", track->timescale);
  if(notes)
    fprintf(xmlout,  "          <!-- Timescale defines time units in one second -->\n");
  fprintf(xmlout,    "          <Duration>\n");
  if(raw)
    fprintf(xmlout,  "            <InTimeUnits>%u</InTimeUnits>\n", track->duration);
  if(derived)
    fprintf(xmlout,  "            <InSeconds>%12.3f</InSeconds>\n", (double)track->duration/(double)track->timescale);    // Make this double later to get fractional seconds
  fprintf(xmlout,    "          </Duration>\n");
  int16_to_3packedchars((short int)track->language, buf);
  fprintf(xmlout,    "          <Language>%s</Language>\n", buf);    /* 3 chars */
  fprintf(xmlout,    "        </MediaHeader>\n");
  fprintf(xmlout,    "        <HandlerReference BoxType=\"hdlr\">\n");
  switch(track->track_type)
  {
  case 0:
    fprintf(xmlout,  "          <HandlerType Code=\"vide\">video media track</HandlerType>\n"); break;
  case 1:
    fprintf(xmlout,  "          <HandlerType Code=\"soun\">Sound</HandlerType>\n"); break;
  case 2:
    fprintf(xmlout,  "          <HandlerType Code=\"hint\">Hint</HandlerType>\n"); break;
  }
  if(notes) {
    fprintf(xmlout,  "          <!-- String value shown is not actually read from file. -->\n");
    fprintf(xmlout,  "          <!-- Shown value is one used for our encode. -->\n");
  }
  fprintf(xmlout,    "        </HandlerReference>\n");
  fprintf(xmlout,    "        <MediaInfoContainer BoxType=\"minf\">\n");
  switch(track->track_type)
  {
  case 0:
    fprintf(xmlout,  "          <VideoMediaHeader BoxType=\"vmhd\">\n");
    fprintf(xmlout,  "            <GraphicsMode>0x%02x</GraphicsMode>\n", track->graphicsmode);
  if(notes) {
      fprintf(xmlout,"            <!-- Enumerated values of graphics mode: -->\n");
      fprintf(xmlout,"            <!--  0x00 = copy (over existing image); -->\n");
      fprintf(xmlout,"            <!--  0x24 = transparent; 'blue-screen' this image using opcolor; -->\n");
      fprintf(xmlout,"            <!--  0x100 = alpha; alpha-blend this image -->\n");
/*    fprintf(xmlout,"            <!--  0x101 = whitealpha; alpha-blend this image, which has been blended with white; -->\n"); This was evidently dropped upon amendment */
      fprintf(xmlout,"            <!--  0x102 = pre-multiplied black alpha; image has been already been alpha-blended with black. -->\n");
      fprintf(xmlout,"            <!--  0x110 = component alpha; blend alpha channel(s) and color channels individually. -->\n");
  }
    fprintf(xmlout,  "            <Opcolor>\n");
    fprintf(xmlout,  "              <Red>0x%02x</Red>\n", track->opcolor[0]);
    fprintf(xmlout,  "              <Green>0x%02x</Green>\n",track->opcolor[1]);
    fprintf(xmlout,  "              <Blue>0x%02x</Blue>\n",track->opcolor[2]);
    fprintf(xmlout,  "            </Opcolor>\n");
    fprintf(xmlout,  "          </VideoMediaHeader>\n");
    break;
  case 1:
    fprintf(xmlout,  "          <SoundMediaHeader BoxType=\"smhd\">\n");
#ifdef CURRENTSTRUCT
  track->balance = track->balance << 8;
#endif
    fprintf(xmlout,  "            <Balance>\n");
  if(notes) {
      fprintf(xmlout,"              <!-- Track audio balance fixes mono track in stereo space. -->\n");
      fprintf(xmlout,"              <!-- Stored as fixed-point binary 8.8 value. Decimal value is approximation. -->\n");
      fprintf(xmlout,"              <!-- 0.0 = center, -1.0 = full left, 1.0 = full right -->\n");
  }
  if(raw)
      fprintf(xmlout,"              <AsHex>0x%04x</AsHex>\n", track->balance);
    if(derived)
    fprintf(xmlout,"              <AsDecimal>%6.3f</AsDecimal>\n", (double)track->balance/(double)0x0100);
    fprintf(xmlout,  "            </Balance>\n");
#ifdef CURRENTSTRUCT
    if(notes)
    fprintf(xmlout,"            <!-- Current m2j_to_metadata implementation always shows bits to right of decimal as zeroed. -->\n");
  track->balance = track->balance >> 8;
#endif
    fprintf(xmlout,  "          </SoundMediaHeader>\n");
    break;
  case 2:
    fprintf(xmlout,  "          <HintMediaHeader BoxType=\"hmhd\">\n");
    fprintf(xmlout,  "            <MaxPDU_Size>%d</MaxPDU_Size>\n", track->maxPDUsize);
    if(notes)
      fprintf(xmlout,"            <!-- Size in bytes of largest PDU in this hint stream. -->\n");
    fprintf(xmlout,  "            <AvgPDU_Size>%d</AvgPDU_Size>\n", track->avgPDUsize);
    if(notes)
      fprintf(xmlout,"            <!-- Average size in bytes of a PDU over the entire presentation. -->\n");
    fprintf(xmlout,  "            <MaxBitRate>%d</MaxBitRate>\n", track->maxbitrate);
    if(notes)
      fprintf(xmlout,"            <!-- Maximum rate in bits per second over any window of 1 second. -->\n");
    fprintf(xmlout,  "            <AvgBitRate>%d</AvgBitRate>\n", track->avgbitrate);
    if(notes)
      fprintf(xmlout,"            <!-- Averate rate in bits per second over the entire presentation. -->\n");
    fprintf(xmlout,  "            <SlidingAvgBit>%d</SlidingAvgBitRate>\n", track->slidingavgbitrate);
    if(notes)
      fprintf(xmlout,"            <!-- Maximum rate in bits per second over any window of one minute. -->\n");
    fprintf(xmlout,  "          </HintMediaHeader>\n");
    break;
  }
  fprintf(xmlout,    "          <DataInfo BoxType=\"dinf\">\n");
  fprintf(xmlout,    "            <DataReference BoxType=\"dref\"  URL_Count=\"%d\" URN_Count=\"%d\">\n", track->num_url, track->num_urn); // table w. flags, URLs, URNs
  // Data structure does not distinguish between single URL, single URN, or DREF table or URLs & URNs.
  // We could infer those, but for now just present everything as a DREF table.
  if(notes)
    fprintf(xmlout,  "              <!-- No entries here mean that file is self-contained, as required by Simple Profile. -->\n");
  for(k = 0; k < track->num_url; k++) {
    fprintf(xmlout,  "            <DataEntryUrlBox BoxType=\"url[space]\">\n"); // table w. flags, URLs, URNs
    if(notes)
      fprintf(xmlout,"              <!-- Only the first 16 bytes of URL location are recorded in mj2_to_metadata data structure. -->\n");
    for(i = 0; i < 4; i++) {
      uint_to_chars(track->url[track->num_url].location[i], buf);
    fprintf(xmlout,  "              <Location>%s</Location>\n");
    }
    fprintf(xmlout,  "            </DataEntryUrlBox>\n"); // table w. flags, URLs, URNs
  }
  for(k = 0; k < track->num_urn; k++) {
    fprintf(xmlout,"            <DataEntryUrnBox BoxType=\"urn[space]\">\n"); // table w. flags, URLs, URNs
    // Only the first 16 bytes are recorded in the data structure currently.
    if(notes)
      fprintf(xmlout,"              <!-- Only the first 16 bytes each of URN name and optional location are recorded in mj2_to_metadata data structure. -->\n");
    fprintf(xmlout,  "              <Name>");
    for(i = 0; i < 4; i++) {
      uint_to_chars(track->urn[track->num_urn].name[i], buf);
      fprintf(xmlout,"%s", buf);
    }
    fprintf(xmlout,  "</Name>\n");
    fprintf(xmlout,  "              <Location>");
    for(i = 0; i < 4; i++) {
      uint_to_chars(track->urn[track->num_urn].location[i], buf);
      fprintf(xmlout,"%s");
    }
    fprintf(xmlout,  "</Location>\n");
    fprintf(xmlout,  "            </DataEntryUrnBox>\n");
  }
  fprintf(xmlout,    "            </DataReference>\n");
  fprintf(xmlout,    "          </DataInfo>\n");

  xml_write_stbl(file, xmlout, track, tnum); /* SampleTable */

  fprintf(xmlout,    "        </MediaInfoContainer>\n");
  fprintf(xmlout,    "      </Media>\n");
}

/* ------------- */

void xml_write_stbl(FILE* file, FILE* xmlout, mj2_tk_t *track, unsigned int tnum)
{
  char buf[5], buf33[33];
  int i, len;
  buf[4] = '\0';

  fprintf(xmlout,      "          <SampleTable BoxType=\"stbl\">\n");
  if(notes)
    fprintf(xmlout,    "            <!-- What follows are specific instances of generic SampleDescription BoxType=\"stsd\" -->\n");
  switch(track->track_type)
  {
  case 0:
    // There could be multiple instances of this, but "entry_count" is just a local at read-time.
    // And it's used wrong, too, as count of just visual type, when it's really all 3 types.
    // This is referred to as "smj2" within mj2.c
    fprintf(xmlout,    "            <VisualSampleEntry BoxType=\"mjp2\">\n");
  if(notes) {
      fprintf(xmlout,  "            <!-- If multiple instances of this box, only first is shown here. -->\n");
    fprintf(xmlout,  "            <!-- Width and Height are in pixels.  Unlike the Track Header, there is no fractional part. -->\n");
    fprintf(xmlout,  "            <!-- In mj2_to_metadata implementation, the values are not represented separately from Track Header's values. -->\n");
  }
  /* No shifting required.  If CURRENTSTRUCT gets changed, then may need to revisit treatment of these */
    fprintf(xmlout,    "              <WidthAsInteger>%d</WidthAsInteger>\n", track->w);
    fprintf(xmlout,    "              <HeightAsInteger>%d</HeightAsInteger>\n", track->h);
// Horizresolution and vertresolution don't require shifting, already stored right in CURRENTSTRUCT
    if(notes) {
      fprintf(xmlout,  "              <!-- Resolutions are in pixels per inch, for the highest-resolution component (typically luminance). -->\n");
      fprintf(xmlout,  "              <!-- Both stored as fixed-point binary 16.16 values. Decimal values are approximations. -->\n");
      fprintf(xmlout,  "              <!-- Typical value for both resolutions is 0x00480000  (72.0) -->\n");
  }
    fprintf(xmlout,    "              <HorizontalRes>\n");
  if(raw)
      fprintf(xmlout,  "                <AsHex>0x%08x</AsHex>\n", track->horizresolution);
  if(derived)
      fprintf(xmlout,  "                <AsDecimal>%12.6f</AsDecimal>\n", (double)track->horizresolution/(double)0x00010000);        /* Rate to play presentation  (default = 0x00010000)          */
    fprintf(xmlout,    "              </HorizontalRes>\n");
    fprintf(xmlout,    "              <VerticalRes>\n");
  if(raw)
      fprintf(xmlout,  "                <AsHex>0x%08x</AsHex>\n", track->vertresolution);
  if(derived)
      fprintf(xmlout,  "                <AsDecimal>%12.6f</AsDecimal>\n", (double)track->vertresolution/(double)0x00010000);        /* Rate to play presentation  (default = 0x00010000)          */
    fprintf(xmlout,    "              </VerticalRes>\n");

    buf33[0] = '\0';
    for(i = 0; i < 8; i++) {
      uint_to_chars((unsigned int)track->compressorname[i], buf);
      strcat(buf33, buf); /* This loads up (4 * 8) + 1 chars, but trailing ones are usually junk */
    }
    len = (int)buf33[0]; /* First byte has string length in bytes.  There may be garbage beyond it. */
    buf33[len+1] = '\0'; /* Suppress it */
    fprintf(xmlout,    "              <CompressorName>%s</CompressorName>\n", buf33+1); /* Start beyond first byte */
  if(notes) {
      fprintf(xmlout,  "              <!-- Compressor name for debugging.  Standard restricts max length to 31 bytes. -->\n");
      fprintf(xmlout,  "              <!-- Usually blank or \"Motion JPEG2000\" -->\n");
  }
    fprintf(xmlout,    "              <Depth>0x%02x</Depth>\n",track->depth);
  if(notes) {
      fprintf(xmlout,  "              <!-- Depth is: -->\n");
      fprintf(xmlout,  "              <!--   0x20: alpha channels present (color or grayscale) -->\n");
      fprintf(xmlout,  "              <!--   0x28: grayscale without alpha -->\n");
      fprintf(xmlout,  "              <!--   0x18: color without alpha -->\n");
  }

    xml_out_frame_jp2h(xmlout, &(track->jp2_struct));  /* JP2 Header */

  /* Following subboxes are optional */
    fprintf(xmlout,    "              <FieldCoding BoxType=\"fiel\">\n");
    fprintf(xmlout,    "                <FieldCount>%d</FieldCount>\n", (unsigned int)track->fieldcount); /* uchar as 1 byte uint */
    if(notes)
      fprintf(xmlout,  "                <!-- Must be either 1 or 2 -->\n");
    fprintf(xmlout,    "                <FieldOrder>%d</FieldOrder>\n", (unsigned int)track->fieldorder); /* uchar as 1 byte uint */
  if(notes) {
      fprintf(xmlout,  "                <!-- When FieldCount=2, FieldOrder means: -->\n");
      fprintf(xmlout,  "                <!--   0: Field coding unknown -->\n");
      fprintf(xmlout,  "                <!--   1: Field with topmost line is stored first in sample; fields are in temporal order -->\n");
      fprintf(xmlout,  "                <!--   6: Field with topmost line is stored second in sample; fields are in temporal order -->\n");
      fprintf(xmlout,  "                <!-- Defaults: FieldCount=1, FieldOrder=0 if FieldCoding box not present -->\n");
      fprintf(xmlout,  "                <!-- Current implementation doesn't retain whether box was actually present. -->\n");
  }
    fprintf(xmlout,    "              </FieldCoding>\n");

    fprintf(xmlout,    "              <MJP2_Profile BoxType=\"jp2p\" Count=\"%d\">\n",track->num_br);
    for (i = 0; i < track->num_br; i++) /* read routine stored in reverse order, so let's undo damage */
    {
      uint_to_chars(track->br[i], buf);
      fprintf(xmlout,  "                <CompatibleBrand>%s</CompatibleBrand>\n", buf);    /*4 characters, each CLi */
    }
    fprintf(xmlout,    "              </MJP2_Profile>\n");

    fprintf(xmlout,    "              <MJP2_Prefix BoxType=\"jp2x\" Count=\"%d\">\n",track->num_jp2x);
    for (i = 0; i < track->num_jp2x; i++)
    { // We'll probably need better formatting than this
      fprintf(xmlout,  "                <Data>0x%02x</Data>\n", track->jp2xdata[i]);    /* Each entry is single byte */
    }
    fprintf(xmlout,    "              </MJP2_Prefix>\n");

    fprintf(xmlout,    "              <MJP2_SubSampling BoxType=\"jsub\">\n"); /* These values are all 1 byte */
    if(notes)
    fprintf(xmlout,  "              <!-- Typical subsample value is 2 for 4:2:0 -->\n");
    fprintf(xmlout,    "                <HorizontalSub>%d</HorizontalSub>\n", track->hsub);
    fprintf(xmlout,    "                <VerticalSub>%d</VerticalSub>\n", track->vsub);
    fprintf(xmlout,    "                <HorizontalOffset>%d</HorizontalOffset>\n", track->hoff);
    fprintf(xmlout,    "                <VerticalOffset>%d</VerticalOffset>\n", track->voff);
  if(notes) {
    fprintf(xmlout,  "                <!-- Typical {horizontal, vertical} chroma offset values: -->\n");
    fprintf(xmlout,  "                <!-- 4:2:2 format (CCIR601, H.262, MPEG2, MPEG4, recom. Exif): {0, 0} -->\n");
    fprintf(xmlout,  "                <!-- 4:2:2 format (JFIF):                                      {1, 0} -->\n");
    fprintf(xmlout,  "                <!-- 4:2:0 format (H.262, MPEG2, MPEG4):                       {0, 1} -->\n");
    fprintf(xmlout,  "                <!-- 4:2:0 format (MPEG1, H.261, JFIF, recom. Exif):           {1, 1} -->\n");
  }
    fprintf(xmlout,    "              </MJP2_SubSampling>\n"); /* These values are all 1 byte */

    fprintf(xmlout,    "              <MJP2_OriginalFormat BoxType=\"orfo\">\n"); /* Part III Appx. 2 */
    fprintf(xmlout,    "                <OriginalFieldCount>%u</OriginalFieldCount>\n", (unsigned int)track->or_fieldcount); /* uchar as 1-byte uint */
    if(notes)
      fprintf(xmlout,  "                <!-- In original material before encoding.  Must be either 1 or 2 -->\n");
    fprintf(xmlout,    "                <OriginalFieldOrder>%u</OriginalFieldOrder>\n", (unsigned int)track->or_fieldorder); /* uchar as 1-byte uint */
  if(notes) {
      fprintf(xmlout,  "                <!-- When FieldCount=2, FieldOrder means: -->\n");
      fprintf(xmlout,  "                <!--   0: Field coding unknown -->\n");
      fprintf(xmlout,  "                <!--   11: Topmost line came from the earlier field; -->\n");
      fprintf(xmlout,  "                <!--   16:  Topmost line came form the later field. -->\n");
      fprintf(xmlout,  "                <!-- Defaults: FieldCount=1, FieldOrder=0 if FieldCoding box not present -->\n");
      fprintf(xmlout,  "                <!-- Current implementation doesn't retain whether box was actually present. -->\n");
  }
    fprintf(xmlout,    "              </MJP2_OriginalFormat>\n");
    fprintf(xmlout,    "            </VisualSampleEntry>\n");
    break;
  case 1: case 2:
    if(notes)
      fprintf(xmlout,  "            <!-- mj2_to_metadata's data structure doesn't record this currently. -->\n"); break;
  }
  fprintf(xmlout,      "            <TimeToSample BoxType=\"stts\">\n");
  fprintf(xmlout,      "              <SampleStatistics>\n");
  fprintf(xmlout,      "                <TotalSamples>%d</TotalSamples>\n", track->num_samples);
  if(notes)
    fprintf(xmlout,    "                <!-- For video, gives the total frames in the track, by summing all entries in the Sample Table -->\n");
  fprintf(xmlout,      "              </SampleStatistics>\n");
  fprintf(xmlout,      "              <SampleEntries EntryCount=\"%d\">\n", track->num_tts);
  for (i = 0; i < track->num_tts; i++) {
    fprintf(xmlout,    "                <Table Entry=\"%u\" SampleCount=\"%d\" SampleDelta=\"%u\" />\n",
                                      i+1, track->tts[i].sample_count, track->tts[i].sample_delta);
  }
  fprintf(xmlout,      "              </SampleEntries>\n");
  fprintf(xmlout,      "            </TimeToSample>\n");

  fprintf(xmlout,      "            <SampleToChunk BoxType=\"stsc\" Count=\"%d\">\n", track->num_samplestochunk);
  for (i = 0; i < track->num_samplestochunk; i++) {
    fprintf(xmlout,    "              <FirstChunk>%u</FirstChunk>\n",track->sampletochunk[i].first_chunk); /* 4 bytes */
    fprintf(xmlout,    "              <SamplesPerChunk>%u</SamplesPerChunk>\n",track->sampletochunk[i].samples_per_chunk); /* 4 bytes */
    fprintf(xmlout,    "              <SampleDescrIndex>%u</SampleDescrIndex>\n",track->sampletochunk[i].sample_descr_idx); /* 4 bytes */
  }
  fprintf(xmlout,      "            </SampleToChunk>\n");
  // After reading this info in, track->num_chunks is calculated and a decompressed table established internally.

  fprintf(xmlout,      "            <SampleSize BoxType=\"stsz\">\n");
  if(track->same_sample_size) {
    // all values in track->sample[i].sample_size are equal.  Grab the first one.
    fprintf(xmlout,    "              <Sample_Size>%u</Sample_Size>\n", track->sample[0].sample_size);
  if(notes) {
      fprintf(xmlout,  "              <!-- Non-zero value means all samples have that size. -->\n");
    fprintf(xmlout,  "              <!-- So <Sample_Count> (aka Entry_Count in std.) has no meaning, is suppressed from this output, and no table follows. -->\n");
  }
  } else {
    fprintf(xmlout,    "              <Sample_Size>0</Sample_Size>\n");
    if(notes)
    if(sampletables)
        fprintf(xmlout,"              <!-- Zero value means samples have different sizes, given in table next of length Sample_Count (aka Entry_Count in std). -->\n");
    else
        fprintf(xmlout,"              <!-- Zero value means samples have different sizes, given in table (not shown) of length Sample_Count (aka Entry_Count in std). -->\n");
  fprintf(xmlout,    "              <Sample_Count>%u</Sample_Count>\n", track->num_samples);
  if(sampletables)
     for (i = 0; i < (int)track->num_samples; i++) {
      fprintf(xmlout,  "              <EntrySize Num=\"%u\">%u</EntrySize>\n", i+1, track->sample[i].sample_size);
     }
  }
  fprintf(xmlout,      "            </SampleSize>\n");

  fprintf(xmlout,      "            <ChunkOffset BoxType=\"stco\">\n");
  // Structure not yet - Variant ChunkLargeOffset 'co64'
  fprintf(xmlout,      "              <EntryCount>%u</EntryCount>\n", track->num_chunks);
  if(notes) {
    fprintf(xmlout,    "              <!-- For this implementation, EntryCount shown is one calculated during file read of <SampleToChunk> data. -->\n");
    fprintf(xmlout,    "              <!-- Implementation will report failure during file read of <ChunkOffset> data if read entry-count disagrees. -->\n");
  }
  if(sampletables)
    for (i = 0; i < (int)track->num_chunks; i++)
      fprintf(xmlout,  "              <Chunk_Offset Num=\"%d\">%u</Chunk_Offset>\n", i+1, track->chunk[i].offset);
  fprintf(xmlout,      "            </ChunkOffset>\n");

  fprintf(xmlout,      "          </SampleTable>\n");
}

/* ------------- */

int xml_out_frame(FILE* file, FILE* xmlout, mj2_sample_t *sample, unsigned int snum, opj_event_mgr_t *event_mgr)
{
  opj_dparameters_t parameters;  /* decompression parameters */
  opj_image_t *img;
  opj_cp_t *cp;
  int i;
  int numcomps;
  unsigned char* frame_codestream;
  opj_dinfo_t* dinfo = NULL;  /* handle to a decompressor */
  opj_cio_t *cio = NULL;
  opj_j2k_t *j2k;

  /* JPEG 2000 compressed image data */

  /* get a decoder handle */
  dinfo = opj_create_decompress(CODEC_J2K);

  /* catch events using our callbacks and give a local context */
  opj_set_event_mgr((opj_common_ptr)dinfo, event_mgr, stderr);

  /* setup the decoder decoding parameters using the current image and user parameters */
  parameters.cp_limit_decoding = DECODE_ALL_BUT_PACKETS;
  opj_setup_decoder(dinfo, &parameters);

  frame_codestream = (unsigned char*) malloc (sample->sample_size-8); /* Skipping JP2C marker */
  if(frame_codestream == NULL)
    return 1;

  fseek(file,sample->offset+8,SEEK_SET);
  fread(frame_codestream,sample->sample_size-8,1, file);  /* Assuming that jp and ftyp markers size do */

  /* open a byte stream */
  cio = opj_cio_open((opj_common_ptr)dinfo, frame_codestream, sample->sample_size-8);

  /* Decode J2K to image: */
  img = opj_decode(dinfo, cio);
  if (!img) {
    fprintf(stderr, "ERROR -> j2k_to_image: failed to decode image!\n");
    opj_destroy_decompress(dinfo);
    opj_cio_close(cio);
    return 1;
  }

  j2k = (opj_j2k_t*)dinfo->j2k_handle;
  j2k_default_tcp = j2k->default_tcp;
  cp = j2k->cp;

  numcomps = img->numcomps;
  /*  Alignments:        "      <       To help maintain xml pretty-printing */
  fprintf(xmlout,      "      <JP2_Frame Num=\"%d\">\n", snum+1);
  fprintf(xmlout,      "        <MainHeader>\n");
  /* There can be multiple codestreams; a particular image is entirely within a single codestream */
  /* TO DO:  A frame can be represented by two I-guess-contigious codestreams if its interleaved. */
  fprintf(xmlout,      "          <StartOfCodestream Marker=\"SOC\" />\n");
  /* "cp" stands for "coding parameter"; "tcp" is tile coding parameters, "tccp" is tile-component coding parameters */
  xml_out_frame_siz(xmlout, img, cp); /* reqd in main */
  xml_out_frame_cod(xmlout, j2k_default_tcp); /* reqd in main */
  xml_out_frame_coc(xmlout, j2k_default_tcp, numcomps); /* opt in main, at most 1 per component */
  xml_out_frame_qcd(xmlout, j2k_default_tcp); /* reqd in main */
  xml_out_frame_qcc(xmlout, j2k_default_tcp, numcomps);  /* opt in main, at most 1 per component */
  xml_out_frame_rgn(xmlout, j2k_default_tcp, numcomps); /* opt, at most 1 per component */
  xml_out_frame_poc(xmlout, j2k_default_tcp); /*  opt (but reqd in main or tile for any progression order changes) */
  /* Next four get j2k_default_tcp passed globally: */
#ifdef SUPPRESS_FOR_NOW
  xml_out_frame_ppm(xmlout, cp); /* opt (but either PPM or PPT [distributed in tile headers] or codestream packet header reqd) */
#endif
  xml_out_frame_tlm(xmlout); /* NO-OP.  TLM NOT SAVED IN DATA STRUCTURE */ /* opt */
  xml_out_frame_plm(xmlout); /* NO-OP.  PLM NOT SAVED IN DATA STRUCTURE */ /* opt in main; can be used in conjunction with PLT */
  xml_out_frame_crg(xmlout); /* NO-OP.  CRG NOT SAVED IN DATA STRUCTURE */ /* opt in main; */
  xml_out_frame_com(xmlout, j2k_default_tcp); /* NO-OP.  COM NOT SAVED IN DATA STRUCTURE */ /* opt in main; */

  fprintf(xmlout,      "        </MainHeader>\n");

  /*  TO DO: all the tile headers (sigh)  */
  fprintf(xmlout,      "        <TilePartHeaders Count=\"%d\">\n", cp->tileno_size);    /* size of the vector tileno */
  for(i = 0; i < cp->tileno_size; i++) { /* I think cp->tileno_size will be same number as (cp->tw * cp->th) or as global j2k_curtileno */
    // Standard seems to use zero-based # for tile-part.
    fprintf(xmlout,    "          <TilePartHeader Num=\"%d\" ID=\"%d\">\n", i, cp->tileno[i]);      /* ID number of the tiles present in the codestream */
    fprintf(xmlout,    "            <StartOfTilePart Marker=\"SOT\" />\n");
  /* All markers in tile-part headers (between SOT and SOD) are optional, unless structure requires. */
    if(i == 0) {
      xml_out_frame_cod(xmlout, &(cp->tcps[i])); /* No more than 1 per tile */
      xml_out_frame_coc(xmlout, &(cp->tcps[i]), numcomps); /* No more than 1 per component */
      xml_out_frame_qcd(xmlout, &(cp->tcps[i])); /* No more than 1 per tile */
      xml_out_frame_qcc(xmlout, &(cp->tcps[i]), numcomps);  /* No more than 1 per component */
      xml_out_frame_rgn(xmlout, &(cp->tcps[i]), numcomps); /* No more than 1 per component */
    }
    xml_out_frame_poc(xmlout, &(cp->tcps[i])); /* Reqd only if any progression order changes different from main POC */
#ifdef SUPPRESS_FOR_NOW
    xml_out_frame_ppt(xmlout, &(cp->tcps[i])); /* Either PPT [distributed in tile headers] or PPM or codestream packet header reqd. */
#endif
    xml_out_frame_plt(xmlout, &(cp->tcps[i])); /* NO-OP.  PLT NOT SAVED IN DATA STRUCTURE */ /* Can be used in conjunction with main's PLM */
    xml_out_frame_com(xmlout, &(cp->tcps[i])); /* NO-OP.  COM NOT SAVED IN DATA STRUCTURE */
    /* opj_tcp_t * cp->tcps; "tile coding parameters" */
    /* Maybe not: fprintf(xmlout,  "        <>%d</>, cp->matrice[i];      */ /* Fixed layer    */
    fprintf(xmlout,    "            <StartOfData Marker=\"SOD\" />\n");
    if(notes)
      fprintf(xmlout,  "            <!-- Tile-part bitstream, not shown, follows tile-part header and SOD marker. -->\n");
    fprintf(xmlout,    "          </TilePartHeader>\n");
  }
  fprintf(xmlout,      "        </TilePartHeaders>\n");    /* size of the vector tileno */

#ifdef NOTYET
  IMAGINE the cp object has data to support the following... but we could use an new different data structure instead
  /* I'm unclear if the span of the original fread(frame_codestream...) included the following items if they're trailing. */
  /* ALSO TO DO, BUT DATA STRUCTURE DOESN'T HANDLE YET: boxes (anywhere in file except before the Filetype box): */
  xml_out_frame_jp2i(xmlout, &cp); /* IntellectualProperty 'jp2i' (no restrictions on location) */
  xml_out_frame_xml(xmlout, &cp); /* XML 'xml\040' (0x786d6c20).  Can appear multiply */
  xml_out_frame_uuid(xmlout, &cp); /* UUID 'uuid' (top level only) */
  xml_out_frame_uinf(xmlout, &cp); /* UUIDInfo 'uinf', includes UUIDList 'ulst' and URL 'url\40' */
#endif

  fprintf(xmlout,      "      </JP2_Frame>\n");

  /* Extra commentary: */
  if(notes) {
    fprintf(xmlout,    "      <!-- Given the number and size of components, mj2_to_frame would try to convert this -->\n");
    if (((img->numcomps == 3) && (img->comps[0].dx == img->comps[1].dx / 2)
      && (img->comps[0].dx == img->comps[2].dx / 2 ) && (img->comps[0].dx == 1))
      || (img->numcomps == 1)) {
      fprintf(xmlout,  "      <!-- file to a YUV movie in the normal manner. -->\n");
    }
    else if ((img->numcomps == 3) &&
      (img->comps[0].dx == 1) && (img->comps[1].dx == 1)&&
    (img->comps[2].dx == 1))  {// If YUV 4:4:4 input --> to bmp
    fprintf(xmlout,  "      <!-- YUV 4:4:4 file to a series of .bmp files. -->\n");
    }
    else {
    fprintf(xmlout,  "      <!-- file whose image component dimension are unknown, to a series of .j2k files. -->\n");
    }
  }

  opj_destroy_decompress(dinfo);
  opj_cio_close(cio);
  free(frame_codestream);

  return 0;
}

/* ------------- */

void int16_to_3packedchars(short int value, char* buf)
{
    /* This is to retrieve the 3-letter ASCII language code */
    /* Each char is packed into 5 bits, as difference from 0x60 */
    int i;
    for (i = 2; i >= 0; i--)
    {
        buf[i] = (value & 0x001f) + 0x60;
        value = (value >>5);
    }
    buf[3] = '\0';
}

/* ------------- */

void xml_out_frame_siz(FILE* xmlout, opj_image_t *img, opj_cp_t *cp)
{
  opj_image_comp_t *comp;
  int i;

  fprintf(xmlout,    "          <ImageAndFileSize Marker=\"SIZ\">\n");
  // This is similar to j2k.c's j2k_dump_image.
  // Not of interest: Lsiz, Rsiz
  fprintf(xmlout,    "            <Xsiz>%d</Xsiz>\n", img->x1);
  fprintf(xmlout,    "            <Ysiz>%d</Ysiz>\n", img->y1);
  if(notes)
    fprintf(xmlout,  "            <!-- Xsiz, Ysiz is the size of the reference grid. -->\n");
  fprintf(xmlout,    "            <XOsiz>%d</XOsiz>\n", img->x0);
  fprintf(xmlout,    "            <YOsiz>%d</YOsiz>\n", img->y0);
  if(notes)
    fprintf(xmlout,  "            <!-- XOsiz, YOsiz are offsets from grid origin to image origin. -->\n");
  fprintf(xmlout,    "            <XTsiz>%d</XTsiz>\n", cp->tdx);
  fprintf(xmlout,    "            <YTsiz>%d</YTsiz>\n", cp->tdy);
  if(notes)
    fprintf(xmlout,  "            <!-- XTsiz, YTsiz is the size of one tile with respect to the grid. -->\n");
  fprintf(xmlout,    "            <XTOsiz>%d</XTOsiz>\n", cp->tx0);
  fprintf(xmlout,    "            <YTOsiz>%d</YTOsiz>\n", cp->ty0);
  if(notes)
    fprintf(xmlout,  "            <!-- XTOsiz, YTOsiz are offsets from grid origin to first tile origin. -->\n");
  fprintf(xmlout,    "            <Csiz>%d</Csiz>\n", img->numcomps);
  if(notes) {
    fprintf(xmlout,  "            <!-- Csiz is the number of components in the image. -->\n");
    fprintf(xmlout,  "            <!-- For image components next: -->\n");
    fprintf(xmlout,  "            <!--   XRsiz, YRsiz denote pixel-sample-spacing on the grid, per Part I Annex B. -->\n");
    //fprintf(xmlout,"            <!--   XO, YO is offset of the component compared to the whole image. -->\n");
    fprintf(xmlout,  "            <!--   Bits per pixel (bpp) is the pixel depth. -->\n");
    fprintf(xmlout,  "            <!--   WidthOfData and HeightOfData are calculated values, e.g.: w = roundup((Xsiz - XOsiz)/ XRsiz) -->\n");
  }

  for (i = 0; i < img->numcomps; i++) {/* image-components */
    comp = &(img->comps[i]);
    fprintf(xmlout,  "            <Component Num=\"%d\">\n", i+1);
    fprintf(xmlout,  "              <Ssiz>\n");
  if(raw)
      fprintf(xmlout,"                <AsHex>0x%02x</AsHex>\n", (comp->sgnd << 7) & (comp->prec - 1));
  if(derived) {
      fprintf(xmlout,"                <Signed>%d</Signed>\n", comp->sgnd);
      fprintf(xmlout,"                <PrecisionInBits>%d</PrecisionInBits>\n", comp->prec);
  }
    fprintf(xmlout,  "              </Ssiz>\n");
    fprintf(xmlout,  "              <XRsiz>%d</XRsiz>\n", comp->dx);
    fprintf(xmlout,  "              <YRsiz>%d</YRsiz>\n", comp->dy);
    fprintf(xmlout,  "              <WidthOfData>%d</WidthOfData>\n", comp->w);
    fprintf(xmlout,  "              <HeightOfData>%d</HeightOfData>\n", comp->h);
    /* Rest of these aren't calculated when SIZ is read:
    fprintf(xmlout,  "              <XO>%d</XO>\n", comp->x0);
    fprintf(xmlout,  "              <YO>%d</YO>\n", comp->y0);
  if(notes)
    fprintf(xmlout,"              <!--  XO, YO is offset of the component compared to the whole image. -->\n");
    fprintf(xmlout,  "              <BitsPerPixel>%d</BitsPerPixel>\n", comp->bpp);
    fprintf(xmlout,  "              <NumberOfDecodedResolution>%d</NumberOfDecodedResolution>\n", comp->resno_decoded); */
    // SUPPRESS: n/a to mj2_to_metadata.  fprintf(xmlout,"        <Factor>%d</Factor\n", comp->factor);
    /* factor = number of division by 2 of the out image  compare to the original size of image */
    // TO DO comp->data:  int *data;      /* image-component data      */

    fprintf(xmlout,  "            </Component>\n");
  }
  fprintf(xmlout,    "          </ImageAndFileSize>\n");
}

/* ------------- */

void xml_out_frame_cod(FILE* xmlout, opj_tcp_t *tcp)
{
/* Could be called with tcp = &j2k_default_tcp;
/* Or, for tile-part header, with &j2k_cp->tcps[j2k_curtileno]
/*  Alignment for main:"          < < < <   To help maintain xml pretty-printing */
/*  Alignment for tile:"            < < <   To help maintain xml pretty-printing */
  opj_tccp_t *tccp;
  int i;
  char spaces[13] = "            "; /* 12 spaces if tilepart*/
  char* s = spaces;
  if(tcp == j2k_default_tcp) {
    s++;s++; /* shorten s to 10 spaces if main */
  }
  tccp = &(tcp->tccps[0]);

  fprintf(xmlout,      "%s<CodingStyleDefault Marker=\"COD\">\n",s); /* Required in main header */
  /* Not retained or of interest: Lcod */
  fprintf(xmlout,      "%s  <Scod>0x%02x</Scod>\n", s, tcp->csty); /* 1 byte */
  if(notes) {
    fprintf(xmlout,    "%s  <!-- For Scod, specific bits mean (where bit 0 is lowest or rightmost): -->\n",s);
    fprintf(xmlout,    "%s  <!-- bit 0: Defines entropy coder precincts -->\n",s);
    fprintf(xmlout,    "%s  <!--        0 = (PPx=15, PPy=15); 1 = precincts defined below. -->\n",s);
    fprintf(xmlout,    "%s  <!-- bit 1: 1 = SOP marker may be used; 0 = not. -->\n",s);
    fprintf(xmlout,    "%s  <!-- bit 2: 1 = EPH marker may be used; 0 = not. -->\n",s);
  }
  fprintf(xmlout,      "%s  <SGcod>\n",s);
  fprintf(xmlout,      "%s    <ProgressionOrder>%d</ProgressionOrder>\n", s, tcp->prg); /* 1 byte, SGcod (A) */
  if(notes) {
    fprintf(xmlout,    "%s    <!-- Defined Progression Order Values are: -->\n",s);
    fprintf(xmlout,    "%s    <!-- 0 = LRCP; 1 = RLCP; 2 = RPCL; 3 = PCRL; 4 = CPRL -->\n",s);
    fprintf(xmlout,    "%s    <!-- where L = \"layer\", R = \"resolution level\", C = \"component\", P = \"position\". -->\n",s);
  }
  fprintf(xmlout,      "%s    <NumberOfLayers>%d</NumberOfLayers>\n", s, tcp->numlayers); /* 2 bytes, SGcod (B) */
  fprintf(xmlout,      "%s    <MultipleComponentTransformation>%d</MultipleComponentTransformation>\n", s, tcp->mct); /* 1 byte, SGcod (C).  More or less boolean */
  if(notes)
    fprintf(xmlout,    "%s    <!-- For MCT, 0 = none, 1 = transform first 3 components for efficiency, per Part I Annex G -->\n",s);
  fprintf(xmlout,      "%s  </SGcod>\n",s);
  /* This code will compile only if declaration of j2k_default_tcp is changed from static (to implicit extern) in j2k.c */
  fprintf(xmlout,      "%s  <SPcod>\n",s);
  /* Internal data structure tccp defines separate defaults for each component, but they all get the same values */
  /* So we only have to report the first component's values here. */
  /* Compare j2k_read_cox(...) */
  fprintf(xmlout,      "%s    <NumberOfDecompositionLevels>%d</NumberOfDecompositionLevels>\n", s, tccp->numresolutions - 1);  /* 1 byte, SPcox (D) */
  fprintf(xmlout,      "%s    <CodeblockWidth>%d</CodeblockWidth>\n", s, tccp->cblkw - 2);  /* 1 byte, SPcox (E) */
  fprintf(xmlout,      "%s    <CodeblockHeight>%d</CodeblockHeight>\n", s, tccp->cblkh - 2);  /* 1 byte, SPcox (F) */
  if(notes) {
    fprintf(xmlout,    "%s    <!-- CBW and CBH are non-negative, and summed cannot exceed 8 -->\n",s);
    fprintf(xmlout,    "%s    <!-- Codeblock dimension is 2^(value + 2) -->\n", s);
  }
  fprintf(xmlout,      "%s    <CodeblockStyle>0x%02x</CodeblockStyle>\n", s, tccp->cblksty);  /* 1 byte, SPcox (G) */
  if(notes) {
    fprintf(xmlout,    "%s    <!-- For CodeblockStyle, bits mean (with value 1=feature on, 0=off): -->\n",s);
    fprintf(xmlout,    "%s    <!-- bit 0: Selective arithmetic coding bypass. -->\n",s);
    fprintf(xmlout,    "%s    <!-- bit 1: Reset context probabilities on coding pass boundaries. -->\n",s);
    fprintf(xmlout,    "%s    <!-- bit 2: Termination on each coding pass. -->\n",s);
    fprintf(xmlout,    "%s    <!-- bit 3: Vertically causal context. -->\n",s);
    fprintf(xmlout,    "%s    <!-- bit 4: Predictable termination. -->\n",s);
    fprintf(xmlout,    "%s    <!-- bit 5: Segmentation symbols are used. -->\n",s);
  }
  fprintf(xmlout,      "%s    <Transformation>%d</Transformation>\n", s, tccp->qmfbid);  /* 1 byte, SPcox (H) */
  if(notes)
    fprintf(xmlout,    "%s    <!-- For Transformation, 0=\"9-7 irreversible filter\", 1=\"5-3 reversible filter\" -->\n",s);
  if (tccp->csty & J2K_CP_CSTY_PRT) {
    fprintf(xmlout,    "%s    <PrecinctSize>\n",s); /* 1 byte, SPcox (I_i) */
    if(notes)
      fprintf(xmlout,  "%s    <!-- These are size exponents PPx and PPy. May be zero only for first level (aka N(L)LL subband)-->\n",s);
    for (i = 0; i < tccp->numresolutions; i++) {
      fprintf(xmlout,  "%s      <PrecinctHeightAndWidth  ResolutionLevel=\"%d\">\n", s, i);
    if(raw)
        fprintf(xmlout,"%s        <AsHex>0x%02x</AsHex>\n", s, (tccp->prch[i] << 4) | tccp->prcw[i]);  /* packed into 1 byte, SPcox (G) */
    if(derived) {
        fprintf(xmlout,"%s        <WidthAsDecimal>%d</WidthAsDecimal>\n", s, tccp->prcw[i]);
        fprintf(xmlout,"%s        <HeightAsDecimal>%d</HeightAsDecimal>\n", s, tccp->prch[i]);
    }
      fprintf(xmlout,  "%s      </PrecinctHeightAndWidth>\n", s, i);
    }
    fprintf(xmlout,    "%s    </PrecinctSize>\n",s); /* 1 byte, SPcox (I_i) */
  }
  fprintf(xmlout,      "%s  </SPcod>\n",s);
  fprintf(xmlout,      "%s</CodingStyleDefault>\n",s);
}

/* ------------- */

void xml_out_frame_coc(FILE* xmlout, opj_tcp_t *tcp, int numcomps) /* Optional in main & tile-part headers */
{
/* Uses global j2k_default_tcp */
  opj_tccp_t *tccp, *firstcomp_tccp;
  int i, compno;
  char spaces[13] = "            "; /* 12 spaces if tilepart*/
  char* s = spaces;
  if(tcp == j2k_default_tcp) {
    s++;s++; /* shorten s to 10 spaces if main */
  }

  firstcomp_tccp = &(tcp->tccps[0]);
    /* Internal data structure tccp defines separate defaults for each component, set from main */
  /* default, then selectively overwritten. */
    /* Compare j2k_read_cox(...) */
  /* We don't really know which was the default, and which were not */
  /* Let's pretend that [0] is the default and all others are not */
  if(notes) {
    fprintf(xmlout,    "%s<!-- mj2_to_metadata implementation always reports component[0] as using default COD, -->\n", s);
    if(tcp == j2k_default_tcp)
      fprintf(xmlout,  "%s<!-- and any other component, with main-header style values different from [0], as COC. -->\n", s);
    else
      fprintf(xmlout,  "%s<!-- and any other component, with tile-part-header style values different from [0], as COC. -->\n", s);
  }
  for (compno = 1; compno < numcomps; compno++) /* spec says components are zero-based */
  {
    tccp = &tcp->tccps[compno];
    if(same_component_style(firstcomp_tccp, tccp))
    continue;

/*  Alignments:          "      < < < < <   To help maintain xml pretty-printing */
    fprintf(xmlout,      "%s<CodingStyleComponent Marker=\"COC\">\n", s); /* Optional in main header, at most 1 per component */
    if(notes)
      fprintf(xmlout,    "%s  <!-- See Ccoc below for zero-based component number. -->\n", s);
    /* Overrides the main COD for the specific component */
    /* Not retained or of interest: Lcod */
    fprintf(xmlout,      "%s  <Scoc>0x%02x</Scoc>\n", s, tccp->csty); /* 1 byte */
  if(notes) {
    fprintf(xmlout,    "%s  <!-- Scoc defines entropy coder precincts: -->\n", s);
      fprintf(xmlout,    "%s  <!--   0 = maximum, namely (PPx=15, PPy=15); 1 = precincts defined below. -->\n", s);
  }
    fprintf(xmlout,      "%s  <Ccoc>%d</Ccoc>\n", s, compno); /* 1 or 2 bytes */
    /* Unfortunately compo isn't retained in j2k_read_coc:  compno = cio_read(j2k_img->numcomps <= 256 ? 1 : 2);  /* Ccoc */
    /*if(j2k_img_numcomps <=256)
    component is 1 byte
    else
      compno is 2 byte */

    /* This code will compile only if declaration of j2k_default_tcp is changed from static (to implicit extern) in j2k.c */
    fprintf(xmlout,      "%s  <SPcoc>\n", s);
    fprintf(xmlout,      "%s    <NumberOfDecompositionLevels>%d</NumberOfDecompositionLevels>\n", s, tccp->numresolutions - 1);  /* 1 byte, SPcox (D) */
    fprintf(xmlout,      "%s    <CodeblockWidth>%d</CodeblockWidth>\n", s, tccp->cblkw - 2);  /* 1 byte, SPcox (E) */
    fprintf(xmlout,      "%s    <CodeblockHeight>%d</CodeblockHeight>\n", s, tccp->cblkh - 2);  /* 1 byte, SPcox (F) */
  if(notes) {
      fprintf(xmlout,    "%s    <!-- CBW and CBH are non-negative, and summed cannot exceed 8 -->\n", s);
      fprintf(xmlout,    "%s    <!-- Codeblock dimension is 2^(value + 2) -->\n", s);
  }
    fprintf(xmlout,      "%s    <CodeblockStyle>0x%02x</CodeblockStyle>\n", s, tccp->cblksty);  /* 1 byte, SPcox (G) */
  if(notes) {
      fprintf(xmlout,    "%s    <!-- For CodeblockStyle, bits mean (with value 1=feature on, 0=off): -->\n", s);
      fprintf(xmlout,    "%s    <!-- bit 0: Selective arithmetic coding bypass. -->\n", s);
      fprintf(xmlout,    "%s    <!-- bit 1: Reset context probabilities on coding pass boundaries. -->\n", s);
      fprintf(xmlout,    "%s    <!-- bit 2: Termination on each coding pass. -->\n", s);
      fprintf(xmlout,    "%s    <!-- bit 3: Vertically causal context. -->\n", s);
      fprintf(xmlout,    "%s    <!-- bit 4: Predictable termination. -->\n", s);
      fprintf(xmlout,    "%s    <!-- bit 5: Segmentation symbols are used. -->\n", s);
  }
    fprintf(xmlout,      "%s    <Transformation>%d</Transformation>\n", s, tccp->qmfbid);  /* 1 byte, SPcox (H) */
    if(notes)
      fprintf(xmlout,    "%s    <!-- For Transformation, 0=\"9-7 irreversible filter\", 1=\"5-3 reversible filter\" -->\n", s);
    if (tccp->csty & J2K_CP_CSTY_PRT) {
      fprintf(xmlout,    "%s    <PrecinctSize>\n", s); /* 1 byte, SPcox (I_i) */
      if(notes)
        fprintf(xmlout,  "%s      <!-- These are size exponents PPx and PPy. May be zero only for first level (aka N(L)LL subband)-->\n", s);
      for (i = 0; i < tccp->numresolutions-1; i++) { /* subtract 1 to get # of decomposition levels */
        fprintf(xmlout,  "%s      <PrecinctHeightAndWidth  ResolutionLevel=\"%d\">\n", s, i);
    if(raw)
          fprintf(xmlout,"%s        <AsHex>0x%02x</AsHex>\n", s, (tccp->prch[i] << 4) | tccp->prcw[i]);  /* packed into 1 byte, SPcox (G) */
    if(derived) {
          fprintf(xmlout,"%s        <WidthAsDecimal>%d</WidthAsDecimal>\n", s, tccp->prcw[i]);
          fprintf(xmlout,"%s        <HeightAsDecimal>%d</HeightAsDecimal>\n", s, tccp->prch[i]);
    }
        fprintf(xmlout,  "%s      </PrecinctHeightAndWidth>\n", s, i);
      }
      fprintf(xmlout,    "%s    </PrecinctSize>\n", s); /* 1 byte, SPcox (I_i) */
    }
    fprintf(xmlout,      "%s  </SPcoc>\n", s);
    fprintf(xmlout,      "%s</CodingStyleComponent>\n", s);
  }
}

/* ------------- */

BOOL same_component_style(opj_tccp_t *tccp1, opj_tccp_t *tccp2)
{
  int i;

  if(tccp1->numresolutions != tccp2->numresolutions)
    return FALSE;
  if(tccp1->cblkw != tccp2->cblkw)
    return FALSE;
  if(tccp1->cblkh != tccp2->cblkh)
    return FALSE;
  if(tccp1->cblksty != tccp2->cblksty)
    return FALSE;
  if(tccp1->csty != tccp2->csty)
    return FALSE;

  if (tccp1->csty & J2K_CP_CSTY_PRT) {
      for (i = 0; i < tccp1->numresolutions; i++) {
         if(tccp1->prcw[i] != tccp2->prcw[i] || tccp1->prch[i] != tccp2->prch[i])
       return FALSE;
      }
  }
  return TRUE;
}

/* ------------- */

void xml_out_frame_qcd(FILE* xmlout, opj_tcp_t *tcp)
{
  /* This code will compile only if declaration of j2k_default_tcp is changed from static (to implicit extern) in j2k.c */
  opj_tccp_t *tccp;
  int bandno, numbands;
  char spaces[13] = "            "; /* 12 spaces if tilepart*/
  char* s = spaces;
  if(tcp == j2k_default_tcp) {
    s++;s++; /* shorten s to 10 spaces if main */
  }

  /* Compare j2k_read_qcx */
  fprintf(xmlout,      "%s<QuantizationDefault Marker=\"QCD\">\n", s); /* Required in main header, single occurrence */
  tccp = &(tcp->tccps[0]);
  /* Not retained or of interest: Lqcd */
  fprintf(xmlout,      "%s  <Sqcd>\n", s);    /* 1 byte */
  if(notes)
    fprintf(xmlout,    "%s  <!-- Default quantization style for all components. -->\n", s);
  if(raw)
    fprintf(xmlout,    "%s    <AsHex>0x%02x</AsHex>\n", s, (tccp->numgbits) << 5 | tccp->qntsty);
  if(derived)
    fprintf(xmlout,    "%s    <QuantizationStyle>%d</QuantizationStyle>\n", s, tccp->qntsty);
  if(notes) {
    fprintf(xmlout,    "%s    <!-- Quantization style (in Sqcd's low 5 bits) may be: -->\n", s);
    fprintf(xmlout,    "%s    <!--   0 = No quantization. SPqcd size = 8 bits-->\n", s);
    fprintf(xmlout,    "%s    <!--   1 = Scalar derived (values signaled for N(L)LL subband only). Use Eq. E.5. SPqcd size = 16. -->\n", s);
    fprintf(xmlout,    "%s    <!--   2 = Scalar expounded (values signaled for each subband). SPqcd size = 16. -->\n", s);
  }
  if(derived)
    fprintf(xmlout,    "%s    <NumberOfGuardBits>%d</NumberOfGuardBits>\n", s,  tccp->numgbits);
  if(notes)
    fprintf(xmlout,    "%s    <!-- 0-7 guard bits allowed (stored in Sqcd's high 3 bits) -->\n", s);
  fprintf(xmlout,      "%s  </Sqcd>\n", s);

  /* Problem: numbands in some cases is calculated from len, which is not retained or available here at this time */
  /* So we'll just dump all internal values */
  /* We could calculate it, but I'm having trouble believing the length equations in the standard */

  fprintf(xmlout,      "%s  <SPqcd>\n", s);
  switch(tccp->qntsty) {
  case J2K_CCP_QNTSTY_NOQNT: /* no quantization */
    /* This is what standard says, but I don't believe it: len = 4 + (3*decomp); */
    numbands = J2K_MAXBANDS; /* should be: numbands = len - 1; */
  /* Better: IMAGINE numbands = tccp->stepsize_numbands; */
    /* Instead look for first zero exponent, quit there.  Adequate? */
    fprintf(xmlout,    "%s    <ReversibleStepSizeValue>\n", s);
  if(notes) {
      fprintf(xmlout,  "%s    <!-- Current mj2_to_metadata implementation dumps entire internal table, -->\n", s);
    fprintf(xmlout,  "%s    <!-- until an exponent with zero value is reached. -->\n", s);
    fprintf(xmlout,  "%s    <!-- Exponent epsilon(b) of reversible dynamic range. -->\n", s);
    fprintf(xmlout,  "%s    <!-- Hex value is as stored, in high-order 5 bits. -->\n", s);
  }
    for (bandno = 0; bandno < numbands; bandno++) {
      if(tccp->stepsizes[bandno].expn == 0)
        break; /* Remove when we have real numbands */
      fprintf(xmlout,  "%s      <DynamicRangeExponent Subband=\"%d\">\n", s, bandno);
    if(raw)
        fprintf(xmlout,"%s        <AsHex>0x%02x</AsHex>\n", s, tccp->stepsizes[bandno].expn << 3);
    if(derived)
        fprintf(xmlout,"%s        <AsDecimal>%d</AsDecimal>\n", s, tccp->stepsizes[bandno].expn);
      fprintf(xmlout,  "%s      </DynamicRangeExponent>\n", s);
    }
    fprintf(xmlout,    "%s    </ReversibleStepSizeValue>\n", s);
    break;
  case J2K_CCP_QNTSTY_SIQNT:  /* scalar quantization derived */
    /* This is what standard says.  Should I believe it:: len = 5;
    /* numbands = 1; */
    fprintf(xmlout,    "%s    <QuantizationStepSizeValues>\n", s);
    if(notes)
      fprintf(xmlout,  "%s    <!-- For irreversible transformation only.  See Part I Annex E Equation E.3 -->\n", s);
    fprintf(xmlout,    "%s      <QuantizationValues Subband=\"0\">\n", s);
    if(notes)
      fprintf(xmlout,  "%s      <!-- For N(L)LL subband: >\n", s);
  if(raw)
      fprintf(xmlout,  "%s        <AsHex>0x%02x</AsHex>\n", s, (tccp->stepsizes[0].expn << 11) | tccp->stepsizes[0].mant);
  if(derived) {
      fprintf(xmlout,  "%s        <Exponent>%d</Exponent>\n", s, tccp->stepsizes[0].expn);
      fprintf(xmlout,  "%s        <Mantissa>%d</Mantissa>\n", s, tccp->stepsizes[0].mant);
  }
    fprintf(xmlout,    "%s      </QuantizationValues>\n", s);
  if(notes) {
      fprintf(xmlout,  "%s      <!-- Exponents for subbands beyond 0 are not from header, but calculated per Eq. E.5 -->\n", s);
      fprintf(xmlout,  "%s      <!-- The mantissa for all subbands is the same, given by the value above. -->\n", s);
      fprintf(xmlout,  "%s      <!-- Current mj2_to_metadata implementation dumps entire internal table, -->\n", s);
    fprintf(xmlout,  "%s      <!-- until a subband with exponent of zero value is reached. -->\n", s);
  }

    for (bandno = 1; bandno < J2K_MAXBANDS; bandno++) {
      if(tccp->stepsizes[bandno].expn == 0)
        break;

      fprintf(xmlout,  "%s      <CalculatedExponent Subband=\"%d\">%d</CalculatedExponent>\n", s, bandno, tccp->stepsizes[bandno].expn);
    }

    fprintf(xmlout,    "%s    </QuantizationStepSizeValues>\n", s);
    break;

  default: /* J2K_CCP_QNTSTY_SEQNT */ /* scalar quantization expounded */
    /* This is what standard says, but should I believe it: len = 5 + 6*decomp; */
    numbands = J2K_MAXBANDS; /* should be: (len - 1) / 2;*/
  /* Better: IMAGINE numbands = tccp->stepsize_numbands; */
    fprintf(xmlout,    "%s    <QuantizationStepSizeValues>\n", s);
  if(notes) {
      fprintf(xmlout,  "%s    <!-- For irreversible transformation only.  See Part I Annex E Equation E.3 -->\n", s);
      fprintf(xmlout,  "%s    <!-- Current mj2_to_metadata implementation dumps entire internal table, -->\n", s);
      fprintf(xmlout,  "%s    <!-- until a subband with mantissa and exponent of zero values is reached. -->\n", s);
    }
    for (bandno = 0; bandno < numbands; bandno++) {
      if(tccp->stepsizes[bandno].expn == 0 && tccp->stepsizes[bandno].mant == 0)
        break; /* Remove when we have real numbands */

      fprintf(xmlout,  "%s      <QuantizationValues Subband=\"%d\">\n", s, bandno);
    if(raw)
        fprintf(xmlout,"%s        <AsHex>0x%02x</AsHex>\n", s, (tccp->stepsizes[bandno].expn << 11) | tccp->stepsizes[bandno].mant);
    if(derived) {
        fprintf(xmlout,"%s        <Exponent>%d</Exponent>\n", s, tccp->stepsizes[bandno].expn);
        fprintf(xmlout,"%s        <Mantissa>%d</Mantissa>\n", s, tccp->stepsizes[bandno].mant);
    }
      fprintf(xmlout,  "%s      </QuantizationValues>\n", s);
    }
    fprintf(xmlout,    "%s    </QuantizationStepSizeValues>\n", s);
    break;
  } /* switch */
  fprintf(xmlout,      "%s  </SPqcd>\n", s);
  fprintf(xmlout,      "%s</QuantizationDefault>\n", s);

/*  Alignments:        "    < < < < <   To help maintain xml pretty-printing */
}

/* ------------- */

void xml_out_frame_qcc(FILE* xmlout, opj_tcp_t *tcp, int numcomps)
{
/* Uses global j2k_default_tcp */
  /* This code will compile only if declaration of j2k_default_tcp is changed from static (to implicit extern) in j2k.c */
  opj_tccp_t *tccp, *firstcomp_tccp;
  int bandno, numbands;
  int compno;
  char spaces[13] = "            "; /* 12 spaces if tilepart*/
  char* s = spaces;
  if(tcp == j2k_default_tcp) {
    s++;s++; /* shorten s to 10 spaces if main */
  }

  firstcomp_tccp = &(tcp->tccps[0]);
    /* Internal data structure tccp defines separate defaults for each component, set from main */
  /* default, then selectively overwritten. */
    /* Compare j2k_read_qcx(...) */
  /* We don't really know which was the default, and which were not */
  /* Let's pretend that [0] is the default and all others are not */
  if(notes) {
    fprintf(xmlout,      "%s<!-- mj2_to_metadata implementation always reports component[0] as using default QCD, -->\n", s);
    if(tcp == j2k_default_tcp)
      fprintf(xmlout,    "%s<!-- and any other component, with main-header quantization values different from [0], as QCC. -->\n", s);
    else
      fprintf(xmlout,    "%s<!-- and any other component, with tile-part-header quantization values different from [0], as QCC. -->\n", s);
  }
  for (compno = 1; compno < numcomps; compno++) /* spec says components are zero-based */
  {
    tccp = &(tcp->tccps[compno]);
    if(same_component_quantization(firstcomp_tccp, tccp))
    continue;

    /* Compare j2k_read_qcx */
    fprintf(xmlout,      "%s<QuantizationComponent Marker=\"QCC\" Component=\"%d\">\n", s, compno); /* Required in main header, single occurrence */
    tccp = &j2k_default_tcp->tccps[0];
    /* Not retained or perhaps of interest: Lqcd   It maybe can be calculated.  */
    fprintf(xmlout,      "%s  <Sqcc>\n", s);    /* 1 byte */
    if(notes)
      fprintf(xmlout,    "%s  <!-- Quantization style for this component. -->\n", s);
  if(raw)
      fprintf(xmlout,    "%s    <AsHex>0x%02x</AsHex>\n", s, (tccp->numgbits) << 5 | tccp->qntsty);
  if(derived)
      fprintf(xmlout,    "%s    <QuantizationStyle>%d</QuantizationStyle>\n", s, tccp->qntsty);
  if(notes) {
      fprintf(xmlout,    "%s    <!-- Quantization style (in Sqcc's low 5 bits) may be: -->\n", s);
      fprintf(xmlout,    "%s    <!--   0 = No quantization. SPqcc size = 8 bits-->\n", s);
      fprintf(xmlout,    "%s    <!--   1 = Scalar derived (values signaled for N(L)LL subband only). Use Eq. E.5. SPqcc size = 16. -->\n", s);
      fprintf(xmlout,    "%s    <!--   2 = Scalar expounded (values signaled for each subband). SPqcc size = 16. -->\n", s);
  }
  if(derived)
      fprintf(xmlout,    "%s    <NumberOfGuardBits>%d</NumberOfGuardBits>\n", s,  tccp->numgbits);
    if(notes)
      fprintf(xmlout,    "%s    <!-- 0-7 guard bits allowed (stored in Sqcc's high 3 bits) -->\n", s);
    fprintf(xmlout,      "%s  </Sqcc>\n", s);

    /* Problem: numbands in some cases is calculated from len, which is not retained or available here at this time */
    /* So we'll just dump all internal values */
    fprintf(xmlout,      "%s  <SPqcc>\n", s);
    switch(tccp->qntsty) {
    case J2K_CCP_QNTSTY_NOQNT:
      numbands = J2K_MAXBANDS; /* should be: numbands = len - 1; */
    /* Better: IMAGINE numbands = tccp->stepsize_numbands; */

      /* Instead look for first zero exponent, quit there.  Adequate? */
      fprintf(xmlout,    "%s    <ReversibleStepSizeValue>\n", s);
    if(notes) {
        fprintf(xmlout,  "%s    <!-- Current mj2_to_metadata implementation dumps entire internal table, -->\n", s);
      fprintf(xmlout,  "%s    <!-- until an exponent with zero value is reached. -->\n", s);
      fprintf(xmlout,  "%s    <!-- Exponent epsilon(b) of reversible dynamic range. -->\n", s);
      fprintf(xmlout,  "%s    <!-- Hex value is as stored, in high-order 5 bits. -->\n", s);
    }
      for (bandno = 0; bandno < numbands; bandno++) {
        if(tccp->stepsizes[bandno].expn == 0)
          break; /* Remove this once we have real numbands */
        fprintf(xmlout,  "%s      <Exponent Subband=\"%d\">\n", s, bandno);
    if(raw)
          fprintf(xmlout,"%s        <AsHex>0x%02x</AsHex>\n", s, tccp->stepsizes[bandno].expn << 3);
    if(derived)
          fprintf(xmlout,"%s        <AsDecimal>%d</AsDecimal>\n", s, tccp->stepsizes[bandno].expn);
        fprintf(xmlout,  "%s      </Exponent>\n", s);
      }
      fprintf(xmlout,    "%s    </ReversibleStepSizeValue>\n", s);
      break;
    case J2K_CCP_QNTSTY_SIQNT:
      /* numbands = 1; */
      fprintf(xmlout,    "%s    <QuantizationStepSizeValues>\n", s);
      if(notes)
        fprintf(xmlout,  "%s    <!-- For irreversible transformation only.  See Part I Annex E Equation E.3 -->\n", s);
      fprintf(xmlout,    "%s      <QuantizationValuesForSubband0>\n", s);
      if(notes)
        fprintf(xmlout,  "%s      <!-- For N(L)LL subband: >\n", s);
    if(raw)
        fprintf(xmlout,  "%s        <AsHex>0x%02x</AsHex>\n", s, (tccp->stepsizes[0].expn << 11) | tccp->stepsizes[0].mant);
    if(derived) {
        fprintf(xmlout,  "%s        <Exponent>%d</Exponent>\n", s, tccp->stepsizes[0].expn);
        fprintf(xmlout,  "%s        <Mantissa>%d</Mantissa>\n", s, tccp->stepsizes[0].mant);
    }
      fprintf(xmlout,    "%s      </QuantizationValuesForSubband0>\n", s);
    if(notes) {
        fprintf(xmlout,  "%s      <!-- Exponents for subbands beyond 0 are not from header, but calculated per Eq. E.5 -->\n", s);
        fprintf(xmlout,  "%s      <!-- The mantissa for all subbands is the same, given by the value above. -->\n", s);
        fprintf(xmlout,  "%s      <!-- Current mj2_to_metadata implementation dumps entire internal table, -->\n", s);
      fprintf(xmlout,  "%s      <!-- until a subband with exponent of zero value is reached. -->\n", s);
        }

      for (bandno = 1; bandno < J2K_MAXBANDS; bandno++) {
        if(tccp->stepsizes[bandno].expn == 0)
          break;

        fprintf(xmlout,  "%s      <CalculatedExponent Subband=\"%d\">%d</CalculatedExponent>\n", s, bandno, tccp->stepsizes[bandno].expn);
      }
      fprintf(xmlout,    "%s    </QuantizationStepSizeValues>\n", s);
      break;

    default: /* J2K_CCP_QNTSTY_SEQNT */
      numbands = J2K_MAXBANDS; /* should be: (len - 1) / 2;*/
    /* Better: IMAGINE numbands = tccp->stepsize_numbands; */
      fprintf(xmlout,    "%s    <QuantizationStepSizeValues>\n", s);
      if(notes) {
        fprintf(xmlout,  "%s    <!-- For irreversible transformation only.  See Part I Annex E Equation E.3 -->\n", s);
        fprintf(xmlout,  "%s    <!-- Current mj2_to_metadata implementation dumps entire internal table, -->\n", s);
      fprintf(xmlout,  "%s    <!-- until a subband with mantissa and exponent of zero values is reached. -->\n", s);
    }
      for (bandno = 0; bandno < numbands; bandno++) {
        if(tccp->stepsizes[bandno].expn == 0 && tccp->stepsizes[bandno].mant == 0)
      break; /* Remove this once we have real numbands count */
        fprintf(xmlout,  "%s      <QuantizationValues Subband=\"%d\">\n", s, bandno);
    if(raw)
          fprintf(xmlout,"%s        <AsHex>0x%02x</AsHex>\n", s, (tccp->stepsizes[bandno].expn << 11) | tccp->stepsizes[bandno].mant);
    if(derived) {
          fprintf(xmlout,"%s        <Exponent>%d</Exponent>\n", s, tccp->stepsizes[bandno].expn);
          fprintf(xmlout,"%s        <Mantissa>%d</Mantissa>\n", s, tccp->stepsizes[bandno].mant);
    }
        fprintf(xmlout,  "%s      </QuantizationValues>\n", s);
      }
      fprintf(xmlout,    "%s    </QuantizationStepSizeValues>\n", s);
      break;
    } /* switch */
    fprintf(xmlout,      "%s  </SPqcc>\n", s);
    fprintf(xmlout,      "%s</QuantizationComponent>\n", s);
  }
/*  Alignments:          "    < < < < <   To help maintain xml pretty-printing */
}

/* ------------- */

BOOL same_component_quantization(opj_tccp_t *tccp1, opj_tccp_t *tccp2)
{
  int bandno, numbands;

  if(tccp1->qntsty != tccp2->qntsty)
    return FALSE;
  if(tccp1->numgbits != tccp2->numgbits)
    return FALSE;

  switch(tccp1->qntsty) {
    case J2K_CCP_QNTSTY_NOQNT:
      numbands = J2K_MAXBANDS; /* should be: numbands = len - 1; */
      /* Instead look for first zero exponent, quit there.  Adequate? */
      for (bandno = 0; bandno < numbands; bandno++) {
        if(tccp1->stepsizes[bandno].expn == 0)
          break;
        if(tccp1->stepsizes[bandno].expn != tccp2->stepsizes[bandno].expn)
         return FALSE;
      }
      break;
    case J2K_CCP_QNTSTY_SIQNT:
      /* numbands = 1; */
      if(tccp1->stepsizes[0].expn != tccp2->stepsizes[0].expn || tccp1->stepsizes[0].mant != tccp2->stepsizes[0].mant)
        return FALSE;
    /* Don't need to check remainder, since they are calculated from [0] */
      break;

    default: /* J2K_CCP_QNTSTY_SEQNT */
      numbands = J2K_MAXBANDS; /* should be: (len - 1) / 2;*/
    /* This comparison may cause us problems with trailing junk values. */
      for (bandno = 0; bandno < numbands; bandno++) {
        if(tccp1->stepsizes[bandno].expn != tccp2->stepsizes[bandno].expn || tccp1->stepsizes[bandno].mant != tccp2->stepsizes[bandno].mant);
          return FALSE;
      }
      break;
    } /* switch */
  return TRUE;
}

/* ------------- */

void xml_out_frame_rgn(FILE* xmlout, opj_tcp_t *tcp, int numcomps)
{
  int compno, SPrgn;
  /* MJ2 files can have regions of interest if hybridized with JPX Part II */
  char spaces[13] = "            "; /* 12 spaces if tilepart*/
  char* s = spaces;
  if(tcp == j2k_default_tcp) {
    s++;s++; /* shorten s to 10 spaces if main */
  }

  for(compno = 0; compno < numcomps; compno++) {
    SPrgn = tcp->tccps[compno].roishift;  /* 1 byte; SPrgn */
    if(SPrgn == 0)
    continue; /* Yet another kludge */

    fprintf(xmlout,    "%s<RegionOfInterest Marker=\"RGN\">\n", s); /* Optional in main header, at most 1 per component */
    if(notes)
      fprintf(xmlout,  "%s<!-- See Crgn below for zero-based component number. -->\n", s);
    /* Not retained or of interest: Lrgd */
    fprintf(xmlout,    "%s  <Srgn>0</Srgn>\n", s); /* 1 byte */
    if(notes)
    fprintf(xmlout,  "%s  <!-- Srgn is ROI style.  Only style=0 defined: Implicit ROI (max. shift) -->\n", s);
    fprintf(xmlout,    "%s  <Crgn>%d</Crgn>\n", s, compno); /* 1 or 2 bytes */
    fprintf(xmlout,    "%s  <SPrgn>%d</SPrgn>\n", s, SPrgn); /* 1 byte */
    if(notes)
      fprintf(xmlout,  "%s  <!-- SPrgn is implicit ROI shift, i.e., binary shifting of ROI coefficients above background. -->\n", s);
    fprintf(xmlout,    "</RegionOfInterest\n", s); /* Optional in main header, at most 1 per component */
  }
}

/* ------------- */

void xml_out_frame_poc(FILE* xmlout, opj_tcp_t *tcp) { /* Progression Order Change */
  /* Compare j2k_read_poc() */
  int i;
  opj_poc_t *poc;
  char spaces[13] = "            "; /* 12 spaces if tilepart*/
  char* s = spaces;
  if(tcp == j2k_default_tcp) {
    s++;s++; /* shorten s to 10 spaces if main */
  }

  if(tcp->POC != 1)
    return; /* Not present */

  fprintf(xmlout,    "%s<ProgressionOrderChange Marker=\"POC\">\n", s); /* Optional in main header, at most 1 per component */
  /* j2k_read_poc seems to allow accumulation of default pocs from multiple POC segments, but does
  the spec really allow that? */
  /* 2 bytes, not retained; Lpoc */
  /* I probably didn't get this dump precisely right. */
  for (i = 0; i < tcp->numpocs; i++) {
    poc = &tcp->pocs[i];
    fprintf(xmlout,  "%s  <Progression Num=\"%d\">\n", s, i+1);
    fprintf(xmlout,  "%S    <RSpoc>%d</RSpoc>\n", s, poc->resno0);  /* 1 byte, RSpoc_i */
    if(notes)
    fprintf(xmlout,"%s    <!-- Resolution level index (inclusive) for progression start. Range: 0 to 33 -->\n", s);
    fprintf(xmlout,  "%s    <CSpoc>%d</CSpoc>\n", s, poc->compno0);/* j2k_img->numcomps <= 256 ? 1 byte : 2 bytes; CSpoc_i */
    if(notes)
      fprintf(xmlout,"%s    <!-- Component index (inclusive) for progression start. -->\n", s);
    fprintf(xmlout,  "%s    <LYEpoc>%d</LYEpoc>\n", s, poc->layno1); /* int_min(cio_read(2), tcp->numlayers);  /* 2 bytes; LYEpoc_i */
    if(notes)
      fprintf(xmlout,"%s    <!-- Layer index (exclusive) for progression end. -->\n", s);
    fprintf(xmlout,  "%s    <REpoc>%d</REpoc>\n", s, poc->resno1); /*int_min(cio_read(1), tccp->numresolutions);  /* REpoc_i */
    if(notes)
      fprintf(xmlout,"%s    <!-- Resolution level index (exclusive) for progression end. Range: RSpoc to 33 -->\n", s);
    fprintf(xmlout,  "%s    <CEpoc>%d</CEpoc>\n", s, poc->compno1); /* int_min(cio_read(j2k_img->numcomps <= 256 ? 1 : 2), j2k_img->numcomps);  /* CEpoc_i */
    if(notes)
    fprintf(xmlout,"%s    <!-- Component index (exclusive) for progression end.  Minimum: CSpoc -->\n", s);
    fprintf(xmlout,  "%s    <Ppoc>%d</Ppoc>\n", s, poc->prg); /* 1 byte Ppoc_i */
  if(notes) {
      fprintf(xmlout,"%s    <!-- Defined Progression Order Values are: -->\n", s);
      fprintf(xmlout,"%s    <!-- 0 = LRCP; 1 = RLCP; 2 = RPCL; 3 = PCRL; 4 = CPRL -->\n", s);
      fprintf(xmlout,"%s    <!-- where L = \"layer\", R = \"resolution level\", C = \"component\", P = \"position\". -->\n", s);
  }
    fprintf(xmlout,  "%s  </Progression>\n", s);
  }
  fprintf(xmlout,    "%s</ProgressionOrderChange\n", s);
}

/* ------------- */

#ifdef SUPPRESS_FOR_NOW
/* Suppress PPM and PPT since we're not showing data from the third option, namely within the codestream, and
that's evidently what frames_to_mj2 uses.  And a hex dump isn't so useful anyway */

void xml_out_frame_ppm(FILE *xmlout, opj_cp_t *cp) { /* For main header, not tile-part (which uses PPT instead). */
/* Either the PPM or PPT is required if the packet headers are not distributed in the bit stream */
/* Use of PPM and PPT are mutually exclusive. */
/* Compare j2k_read_ppm() */
  int j;

  if(cp->ppm != 1)
    return; /* Not present */
/* Main header uses indent of 10 spaces */
  fprintf(xmlout,    "          <PackedPacketHeadersMainHeader Marker=\"PPM\">\n"); /* Optional in main header, but if not, must be in PPT or codestream */
  /* 2 bytes Lppm not saved */
  if(notes) {
    fprintf(xmlout,  "          <!-- If there are multiple PPM marker segments in the main header, -->\n");
    fprintf(xmlout,  "          <!-- this mj2_to_metadata implementation will report them as a single consolidated PPM header. -->\n");
    fprintf(xmlout,  "          <!-- The implementation can't currently segregate by tile-part. -->\n");
    fprintf(xmlout,  "          <!-- TO DO? further map the packet headers to xml. -->\n");
  }

  /* 1 byte, not retained ; Zppm is sequence # of this PPM header */
  /* 4 bytes, possibly overwritten multiple times in j2k_cp->ppm_previous: Nppm */
  /* Use j symbol for index instead of i, to make comparable with j2k_read_ppm */
  /* Not real clear whether to use ppm->store or ppm_len as upper bound */
  fprintf(xmlout,    "            <PackedData>\n");
  xml_out_dump_hex(xmlout, cp->ppm_data, cp->ppm_len);
  /* Dump packet headers 1 byte at a time: lppm[i][j] */
  fprintf(xmlout,    "            </PackedData>\n");
  fprintf(xmlout,    "          </PackedPacketHeadersMainHeader>\n"); /* Optional in main header, but if not, must be in PPT or codestream */
}

/* ------------- */

void xml_out_frame_ppt(FILE *xmlout, opj_tcp_t *tcp) { /* For tile-part header, not main (which uses PPM instead). */
/* Either the PPM or PPT is required if the packet headers are not distributed in the bit stream */
/* Use of PPM and PPT are mutually exclusive. */
/* Compare j2k_read_ppt() */
  int j;

  if(tcp->ppt != 1)
    return; /* Not present */

  /* Tile-part indents are 12 spaces */
  fprintf(xmlout,    "            <PackedPacketHeadersTilePartHeader Marker=\"PPT\">\n"); /* Optional in main header, but if not, must be in PPT or codestream */
  /* 2 bytes Lppm not saved */
  if(notes) {
    fprintf(xmlout,  "            <!-- If there are multiple PPT marker segments in the tile-part header, -->\n");
    fprintf(xmlout,  "            <!-- this mj2_to_metadata implementation will report them as a single consolidated PPT header. -->\n");
    fprintf(xmlout,  "            <!-- The implementation can't currently segregate by tile-part. -->\n");
    fprintf(xmlout,  "            <!-- TO DO? further map the packet headers to xml. -->\n");
  }

  /* 1 byte, not retained ; Zppt is sequence # of this PPT header */
  /* 4 bytes, possibly overwritten multiple times in j2k_cp->ppt_previous: Nppt */
  /* Use j symbol for index instead of i, to make comparable with j2k_read_ppt */
  /* Not real clear whether to use ppt->store or ppt_len as upper bound */
  fprintf(xmlout,    "              <PackedData>\n");
  xml_out_dump_hex(xmlout, tcp->ppt_data, tcp->ppt_len);
  /* Dump packet headers 1 byte at a time: lppt[i][j] */
  fprintf(xmlout,    "              </PackedData>\n");
  fprintf(xmlout,    "            </PackedPacketHeadersTileHeader>\n"); /* Optional in tile-part header, but if not, must be in PPM or codestream */
}
#endif SUPPRESS_FOR_NOW

/* ------------- */

void xml_out_frame_tlm(FILE* xmlout) { /* opt, main header only.  May be multiple. */
/* Compare j2k_read_tlm()... which doesn't retain anything! */
/* Plan:  Since this is only called from main header, not tilepart, use global j2k_default_tcp rather than parameter */
/* Main header indents are 10 spaces */
}

/* ------------- */

void xml_out_frame_plm(FILE* xmlout) { /* opt, main header only; can be used in conjunction with tile-part's PLT */
/* NO-OP.  PLM NOT SAVED IN DATA STRUCTURE */
  /* Compare j2k_read_plm()... which doesn't retain anything! */
/* Plan:  Since this is only called from main header, not tilepart, use global j2k_default_tcp rather than parameter */
/* Main header indents are 10 spaces */
}

/* ------------- */

void xml_out_frame_plt(FILE* xmlout, opj_tcp_t *tcp) { /* opt, tile-part headers only; can be used in conjunction with main header's PLM */
/* NO-OP.  PLT NOT SAVED IN DATA STRUCTURE */
  /* Compare j2k_read_plt()... which doesn't retain anything! */
/* Tile-part header indents are 12 spaces */
}

/* ------------- */

void xml_out_frame_crg(FILE* xmlout) { /* NO-OP.  CRG NOT SAVED IN DATA STRUCTURE */ /* opt, main header only; */
/* Compare j2k_read_crg()... which doesn't retain anything! */
/* Plan:  Since this is only called from main header, not tilepart, use global j2k_default_tcp rather than parameter */
#ifdef NOTYET
  THIS PSEUDOCODE IMAGINES THESE EXIST: j2k_default_tcp->crg, j2k_default_tcp->crg_i, j2k_default_tcp->crg_xcrg*, j2k_default_tcp->crg_ycrg*
  (POSSIBLY DON'T NEED crg_i, CAN GET NUMBER OR COMPONENTS FROM ELSEWHERE)
  if(j2k_default_tcp->crg != 1 || j2k_default_tcp->crg_i == 0)
    return; /* Not present */

/* Main header indents are 10 spaces */
  fprintf(xmlout,    "          <ComponentRegistration Marker=\"RG\" Count=\"%d\">\n", j2k_default_tcp->crg_i);
  if(notes) {
    fprintf(xmlout,  "          <!-- Fine tuning of registration of components with respect to each other, -->\n");
    fprintf(xmlout,  "          <!-- not required but potentially helpful for decoder. -->\n");
    fprintf(xmlout,  "          <!-- These supplementary fractional offsets are in units of 1/65536 of the horizontal -->\n");
    fprintf(xmlout,  "          <!-- or vertical separation (e.g., XRsiz[i] or YRsiz[i] for component i). -->\n");
  }
  /* This isn't the most compact form of table, but is OK when number of components is small, as is likely. */
  for (i = 0; i < j2k_default_tcp->crg_i; i++) {
    fprintf(xmlout,  "            <Component Num=\"%d\">\n", i+1);
    fprintf(xmlout,  "              <Xcrg>\n");
  if(raw)
      fprintf(xmlout,"                <AsNumerator>%d</AsNumerator>\n", j2k_default_tcp->crg_xcrg[i]);
  if(derived) {
    /* Calculate n * 100%/65536; 4 digits after decimal point is sufficiently accurate */
      fprintf(xmlout,"                <AsPercentage>%.4f</AsPercentage>\n", ((double)j2k_default_tcp->crg_xcrg[i])/655.36);
    /* We could do another calculation that include XRsiz[i]; maybe later. */
  }
    fprintf(xmlout,  "              </Xcrg>\n");
    fprintf(xmlout,  "              <Ycrg>\n");
  if(raw)
      fprintf(xmlout,"                <AsNumerator>%d</AsNumerator>\n", j2k_default_tcp->crg_ycrg[i]);
  if(derived) {
      fprintf(xmlout,"                <AsPercentage>%f</AsPercentage>\n", ((double)j2k_default_tcp->crg_ycrg[i])/655.36);
  }
    fprintf(xmlout,  "              </Ycrg>\n");
    fprintf(xmlout,  "            </Component>\n");
  }

  fprintf(xmlout,    "          </ComponentRegistration>\n");

#endif
}

/* ------------- */

/* Regrettably from a metadata point of view, j2k_read_com() skips over any comments in main header or tile-part-header */
void xml_out_frame_com(FILE* xmlout, opj_tcp_t *tcp) { /* NO-OP.  COM NOT SAVED IN DATA STRUCTURE */ /* opt in main or tile-part headers; */
/* Compare j2k_read_com()... which doesn't retain anything! */
#ifdef NOTYET
  char spaces[13] = "            "; /* 12 spaces if tilepart*/
  char* s = spaces;
  if(tcp == &j2k_default_tcp) {
    s++;s++; /* shorten s to 10 spaces if main */
  }
  THIS PSEUDOCODE IMAGINES THESE EXIST: tcp->com, tcp->com_len, tcp->com_data array
  if(tcp->com != 1)
    return; /* Not present */

  fprintf(xmlout,    "%s<Comment Marker=\"COM\">\n", s); /* Optional in main or tile-part header */
  xml_out_dump_hex_and_ascii(tcp->com_data, tcp->com_len, s);
  fprintf(xmlout,    "%s</Comment>\n", s);
#endif
}

void xml_out_dump_hex(FILE* xmlout, char *data, int data_len, char* s) {
  /* s is a string of spaces for indent */
  int i;

  /* This is called when raw is true, or there is no appropriate derived form */
  fprintf(xmlout,    "%s<AsHex>\n", s);
  fprintf(xmlout,    "%s  ", s); /* Inadequate for pretty printing */
  for (i = 0; i < data_len; i++) {  /* Dump packet headers */
    fprintf(xmlout,  "%02x", data[i]);
  }
  fprintf(xmlout,    "%s</AsHex>\n", s);
}

/* Define this as an even number: */
#define BYTES_PER_DUMP_LINE 40
/* Current total width for Hex and ASCII is : 11 spaces lead + (3 * BPDL) + 2 spaces + BPDL */
void xml_out_dump_hex_and_ascii(FILE* xmlout, char *data, int data_len, char* s) {
  /* s is a string of spaces for indent */
  int i,j;

  if(raw)
    xml_out_dump_hex(xmlout, data, data_len, s);

  if(derived) {
    fprintf(xmlout,  "%s<AsHexAndASCII>\n", s);
  for (i = 0; i < data_len; ) {
      fprintf(xmlout,"%s ", s); /* Additional leading space added in loop */
    /* First column: hex */
      for (j = 0; j < BYTES_PER_DUMP_LINE; j++)  /* Dump bytes */
        fprintf(xmlout," %02x", data[i+j]);
      /* Space between columns... */ fprintf(xmlout,  "  ");
    /* Second column: ASCII */
    for (j = 0; j < BYTES_PER_DUMP_LINE; j++, i++) {
      if(isprint((int)data[i]) && i < data_len)
          fprintf(xmlout,"%c", data[i]);
      else
        fprintf(xmlout," ");
      }
      /* If we also wanted to output UCS-2 Unicode as a third column, then entire document
      must use fwprintf.  Forget about it for now.  As it stands, if data is UCS-2 format but still
      the ASCII set, then we'll be able to read every other byte as ASCII in column 2.  If
      data is UTF-8 format but still ASCII, then we'll be able to read every byte as ASCII
      in column 2. */
    }
    fprintf(xmlout,  "%s</AsHexAndASCII>\n", s);
  }
}


/* ------------- */

void xml_out_frame_jp2h(FILE* xmlout, opj_jp2_t *jp2_struct) {  /* JP2 Header */
/* Compare jp2_read_jp2h(opj_jp2_t * jp2_struct) */
  int i;

  fprintf(xmlout,      "              <JP2Header BoxType=\"jp2h\">\n");

/* Compare jp2_read_ihdr(jp2_struct)) */
  fprintf(xmlout,      "                <ImageHeader BoxType=\"ihdr\">\n");
  fprintf(xmlout,      "                  <HEIGHT>%d</HEIGHT>\n", jp2_struct->h); /* 4 bytes */
  fprintf(xmlout,      "                  <WIDTH>%d</WIDTH>\n", jp2_struct->w); /* 4 bytes */
  if(notes)
    fprintf(xmlout,    "                  <!-- HEIGHT here, if 2 fields per image, is of total deinterlaced height. -->\n");
  fprintf(xmlout,      "                  <NC>%d</NC>\n", jp2_struct->numcomps); /* 2 bytes */
  if(notes)
    fprintf(xmlout,    "                  <!-- NC is number of components -->\n"); /* 2 bytes */
  fprintf(xmlout,      "                  <BPC>\n"); /* 1 byte */
  if(jp2_struct->bpc == 255) {
    fprintf(xmlout,    "                    <AsHex>0x%02x</AsHex>\n", jp2_struct->bpc); /* 1 byte */
    if(notes)
      fprintf(xmlout,  "                    <!-- BPC = 0xff means bits per pixel varies with component; see table below. -->\n");
  } else { /* Not 0xff */
    if(raw) {
      fprintf(xmlout,  "                    <AsHex>0x%02x</AsHex>\n", jp2_struct->bpc); /* 1 byte */
      if(notes)
        fprintf(xmlout,"                    <!-- BPC = 0xff means bits per pixel varies with component; see table below. -->\n");
  }
    if(derived) {
      fprintf(xmlout,  "                    <BitsPerPixel>%d</BitsPerPixel>\n", jp2_struct->bpc & 0x7f);
      fprintf(xmlout,  "                    <Signed>%d</Signed>\n", jp2_struct->bpc >> 7);
  }
  }
  fprintf(xmlout,      "                  </BPC>\n");
  fprintf(xmlout,      "                  <C>%d</C>\n", jp2_struct->C); /* 1 byte */
  if(notes)
    fprintf(xmlout,    "                  <!-- C is compression type.  Only \"7\" is allowed to date. -->\n"); /* 2 bytes */
  fprintf(xmlout,      "                  <UnkC>%d</UnkC>\n", jp2_struct->UnkC); /* 1 byte */
  if(notes)
    fprintf(xmlout,    "                  <!-- Colourspace Unknown. 1 = unknown, 0 = known (e.g., colourspace spec is accurate) -->\n"); /* 1 byte */
  fprintf(xmlout,      "                  <IPR>%d</IPR>\n", jp2_struct->IPR); /* 1 byte */
  if(notes)
    fprintf(xmlout,    "                  <!-- IPR is 1 if frame contains an Intellectual Property box; 0 otherwise. -->\n"); /* 2 bytes */
  fprintf(xmlout,      "                </ImageHeader>\n");

  if (jp2_struct->bpc == 255)
  {
    fprintf(xmlout,    "                <BitsPerComponent BoxType=\"bpcc\">\n");
    if(notes)
      fprintf(xmlout,  "                <!-- Pixel depth (range 1 to 38) is low 7 bits of hex value + 1 -->\n");
  /* Bits per pixel varies with components */
    /* Compare jp2_read_bpcc(jp2_struct) */
  for (i = 0; i < (int)jp2_struct->numcomps; i++) {
    if(raw)
        fprintf(xmlout,"                  <AsHex>0x%02x</AsHex>\n", jp2_struct->comps[i].bpcc); /* 1 byte */
    if(derived) {
        fprintf(xmlout,"                  <BitsPerPixel>%d</BitsPerPixel>\n", (jp2_struct->comps[i].bpcc & 0x7f)+1);
        fprintf(xmlout,"                  <Signed>%d</Signed>\n", jp2_struct->comps[i].bpcc >> 7);
    }
  }
    fprintf(xmlout,    "                </BitsPerComponent>\n");
  }

  /* Compare jp2_read_colr(jp2_struct) */
  fprintf(xmlout,      "                <ColourSpecification BoxType=\"colr\">\n");
  fprintf(xmlout,      "                  <METH>%d</METH>\n", jp2_struct->meth); /* 1 byte */
  if(notes) {
    fprintf(xmlout,    "                  <!-- Valid values of specification method so far: -->\n");
    fprintf(xmlout,    "                  <!--   1 = Enumerated colourspace, in EnumCS field -->\n");
    fprintf(xmlout,    "                  <!--   2 = Restricted ICC Profile, in PROFILE field -->\n");
  }
  fprintf(xmlout,      "                  <PREC>%d</PREC>\n", jp2_struct->precedence); /* 1 byte */
  if(notes)
    fprintf(xmlout,    "                  <!-- 0 is only valid value of precedence so far. -->\n");
  fprintf(xmlout,      "                  <APPROX>%d</APPROX>\n", jp2_struct->approx); /* 1 byte */
  if(notes)
    fprintf(xmlout,    "                  <!-- 0 is only valid value of colourspace approximation so far. -->\n");

  if (jp2_struct->meth == 1) {
    fprintf(xmlout,    "                  <EnumCS>%d</EnumCS>\n", jp2_struct->enumcs); /* 4 bytes */
  if(notes) {
    fprintf(xmlout,  "                  <!-- Valid values of enumerated MJ2 colourspace so far: -->\n");
    fprintf(xmlout,  "                  <!--   16: sRGB as defined by IEC 61966-2-1. -->\n");
    fprintf(xmlout,  "                  <!--   17: greyscale (related to sRGB). -->\n");
    fprintf(xmlout,  "                  <!--   18: sRGB YCC (from JPEG 2000 Part II). -->\n");
    fprintf(xmlout,  "                  <!-- (Additional JPX values are defined in Part II). -->\n");
  }
  }
  else
    if(notes)
      fprintf(xmlout,  "                  <!-- PROFILE is not handled by current OpenJPEG implementation. -->\n");
    /* only 1 byte is read and nothing stored */
  fprintf(xmlout,      "                </ColourSpecification>\n");

  /* TO DO?  No OpenJPEG support.
  Palette 'pclr'
  ComponentMapping 'cmap'
  ChannelDefinition 'cdef'
  Resolution 'res'
  */
  fprintf(xmlout,      "              </JP2Header>\n");
}
/* ------------- */

#ifdef NOTYET
IMAGE these use cp structure, extended... but we could use a new data structure instead
void xml_out_frame_jp2i(FILE* xmlout, opj_cp_t *cp) {
  /* IntellectualProperty 'jp2i' (no restrictions on location) */
  int i;
  IMAGE cp->jp2i, cp->jp2i_count, cp->jp2i_data (array of chars), cp->cp2i_len (array of ints)
  if(cp->jp2i != 1)
    return; /* Not present */

  for(i = 0; i < cp->jp2i_count; i++)
  {
    fprintf(xmlout,      "            <IntellectualProperty BoxType=\"jp2i\">\n");
  /* I think this can be anything, including binary, so do a dump */
    /* Is it better to indent or not indent this content?  Indent is better for reading, but
    worse for cut/paste. */
    xml_out_dump_hex_and_ascii(xmlout, cp->jp2i_data[i], cp->jp2i_len[i]);
    fprintf(xmlout,      "            </IntellectualProperty>\n");
  }
}

void xml_out_frame_xml(FILE* xmlout, opj_cp_t *cp) {
  /* XML 'xml\040' (0x786d6c20).  Can appear multiply, before or after jp2c codestreams */
  IMAGE cp->xml, cp->xml_count, cp->xml_data (array of chars)
  MAYBE WE DON'T NEED cp->xml_len (array of ints) IF WE ASSUME xml_data IS NULL-TERMINATED.
  ASSUME ASSUME EACH LINE IS ENDED BY \n.
  int i;
  if(cp->xml != 1)
    return; /* Not present */

  for(i = 0; i < cp->xml_count; i++)
  {
    fprintf(xmlout,      "            <TextFormXML BoxType=\"xml[space]" Instance=\"%d\">\n", i+1);
    /* Is it better to indent or not indent this content?  Indent is better for reading, but
    worse for cut/paste. Being lazy, didn't indent here. */
    fprintf(xmlout,cp->xml_data[i]); /* May be multiple lines */ /* Could check if this is well-formed */
    fprintf(xmlout,      "            </TextFormXML>\n");
  }
}

void xml_out_frame_uuid(FILE* xmlout, opj_cp_t *cp) {
  /* UUID 'uuid' (top level only) */
  /* Part I 1.7.2 says: may appear multiply in JP2 file, anywhere except before File Type box */
  /* Part III 5.2.1 says: Private extensions shall be achieved through the 'uuid' type. */
  /* A UUID is a 16-byte value.  There is a conventional string representation for it:
     "0x12345678-9ABC-DEF0-1234-567890ABCDEF".  Let's assume that is what is stored in uuid_value */

  /* Part III 6.1 Any other MJ2 box type could be alternatively written as a 'uuid' box, with value given
     as : 0xXXXXXXXX-0011-0010-8000-00AA00389B71, where the Xs are the boxtype in hex.  However,
     such a file is "not compliant; systems may choose to read [such] objects ... as equivalent to the box of
     the same type, or not."  Here, we choose not to. */
  int i;
  IMAGE cp->uuid, cp->uuid_count, cp->uuid_value (array of uuids... let's say fixed-length strings) cp->uuid_data (array of char buffers), cp->uuid_len (array of ints)
  if(cp->juuid != 1)
    return; /* Not present */

  for(i = 0; i < cp->uuid_count; i++)
  {
    fprintf(xmlout,      "            <UniversalUniqueID BoxType=\"uuid\">
  fprintf(xmlout,      "              <UUID>%s</UUDI>\n", cp->uuid_value[i]);
  fprintf(xmlout,      "              <Data>\n");
  /* I think this can be anything, including binary, so do a dump */
    /* Is it better to indent or not indent this content?  Indent is better for reading, but
    worse for cut/paste. */
    xml_out_dump_hex_and_ascii(xmlout, cp->uuid_data[i], cp->uuid_len[i]);
  fprintf(xmlout,      "              </Data>\n");
    fprintf(xmlout,      "            </UniversalUniqueID>\n");
  }
}

void xml_out_frame_uinf(FILE* xmlout, opj_cp_t *cp) {
  /* UUIDInfo 'uinf', includes UUIDList 'ulst' and URL 'url\40' */
  /* Part I 1.7.3 says: may appear multiply in JP2 file, anywhere at the top level except before File Type box */
  /* So there may be multiple ulst's, and each can have multiple UUIDs listed (with a single URL) */
  /* This is not quite as vendor-specific as UUIDs, or at least is meant to be generally readable */
  /* Assume UUIDs stored in canonical string format */
  int i, j;
  IMAGE cp->uinf, cp->uinf_count, cp->uinf_ulst_nu (array of ints)
    cp->uinf_uuid (2 dimensional array of uuids... let's say fixed-length strings),
    cp->uinf_url (array of char buffers)

  if(cp->uinf != 1)
    return; /* Not present */

  for(i = 0; i < cp->uuid_count; i++)
  {
    fprintf(xmlout,      "            <UUIDInfo BoxType=\"uinf\">\n");
    fprintf(xmlout,      "              <UUIDList BoxType=\"ulst\" Count=\"%d\">\n",cp->cp->uinf_ulst_nu[i]);
  for(j = 0; j < cp->uinf_ulst_nu[i];  j++)
    fprintf(xmlout,    "              <ID Instance=\"%s\">%s</ID>\n", cp->uuif_uuid[i][j], j+1);
    fprintf(xmlout,      "              </UUIDList>\n");
  fprintf(xmlout,      "              <DataEntryURL>\n");
  /* Could add VERS and FLAG here */
  fprintf(xmlout,      "                <LOC>\n");
    fprintf(xmlout,      "                  %s",cp->uinf_url[i]); /* Probably single line, so indent works */ /* In theory, could check if this is well-formed, or good live link */
  fprintf(xmlout,      "                </LOC>\n");
  fprintf(xmlout,      "              </DataEntryURL>\n");
    fprintf(xmlout,      "            </UUIDInfo>\n");
  }
}

IMAGE these use cp structure, extended... but we could use a new data structure instead
void xml_out_frame_unknown_type(FILE* xmlout, opj_cp_t *cp) {
  /* Part III 5.2.1 says "Type fields not defined here are reserved.  Private extensions
     shall be acieved through the 'uuid' type." [This implies an unknown
     type would be an error, but then...] "Boxes not explicitly defined in this standard,
   or otherwise unrecognized by a reader, may be ignored."
   Also, it says  "the following types are not and will not be used, or used only in
   their existing sense, in future versions of this specification, to avoid conflict
   with existing content using earlier pre-standard versions of this format:
     clip, crgn, matt, kmat, pnot, ctab, load, imap;
     track reference types tmcd, chap, sync,scpt, ssrc"
   [But good luck figuring out the mapping.]
   Part III Amend. 2 4.1 is stronger: "All these specifications [of this family, e.g.,
   JP2 Part I, ISO Base format (Part 12) leading to MP4, Quicktime, and possibly including
   MJ2] require that readers ignore objects that are unrecognizable to them".
   */
  int i;
  IMAGE cp->unknown_type, cp->unknown_type_count, cp->unknown_type_boxtype (array of buf[5]s), cp->unknown_type_data (array of chars), cp->unknown_type_len (array of ints)
  if(cp->unknown_type != 1)
    return; /* Not present */

  for(i = 0; i < cp->unknown_type_count; i++)
  {
    fprintf(xmlout,      "            <UnknownType BoxType=\"%s\">\n", cp->unknown_type_boxtype[i]);
    /* Can be anything, including binary, so do a dump */
    /* Is it better to indent or not indent this content?  Indent is better for reading, but
    worse for cut/paste. */
    xml_out_dump_hex_and_ascii(xmlout, cp->unknown_type_data[i], cp->unknown_type_len[i]);
    fprintf(xmlout,      "            </UnknownType>\n");
  }
}

#endif
