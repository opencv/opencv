/* last review : october 29th, 2002 */

#ifndef _GETOPT_H_
#define _GETOPT_H_

extern int opterr;
extern int optind;
extern int optopt;
extern int optreset;
extern char *optarg;

extern int getopt(int nargc, char *const *nargv, const char *ostr);

#endif				/* _GETOPT_H_ */
