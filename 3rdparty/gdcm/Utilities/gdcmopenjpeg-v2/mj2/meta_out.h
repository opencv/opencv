/* meta_out.h */
/* Dump MJ2, JP2 metadata (partial so far) to xml file */
/* Callable from mj2_to_metadata */
/* Contributed to Open JPEG by Glenn Pearson, U.S. National Library of Medicine */

#define BOOL int
#define FALSE 0
#define TRUE 1

void xml_write_init(BOOL n, BOOL t, BOOL r, BOOL d);

int xml_write_struct(FILE *file, FILE *xmlout, opj_mj2_t * movie, unsigned int sampleframe, char* stringDTD, opj_event_mgr_t *event_mgr);
