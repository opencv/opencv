/* last review : october 29th, 2002 */

#ifndef _GETOPT_H_
#define _GETOPT_H_

typedef struct option
{
  const char *name;
  int has_arg;
  int *flag;
  int val;
}option_t;

#define  NO_ARG  0
#define REQ_ARG  1
#define OPT_ARG  2

extern int opterr;
extern int optind;
extern int optopt;
extern int optreset;
extern char *optarg;

extern int getopt(int nargc, char *const *nargv, const char *ostr);
extern int getopt_long(int argc, char * const argv[], const char *optstring,
      const struct option *longopts, int totlen);


#endif        /* _GETOPT_H_ */
