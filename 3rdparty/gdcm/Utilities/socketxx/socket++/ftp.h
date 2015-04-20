// ftp.h
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11

#ifndef FTP_H
#define FTP_H

#include <socket++/protocol.h>
#ifndef WIN32
#	include <sys/param.h>
#else
#define	MAXPATHLEN MAX_PATH
#endif // !WIN32

class ftp: public protocol {
 public:
  enum reptype {
    rt_ascii_nonprint,
    rt_ascii_telnet,
    rt_ascii_asa,
    rt_ebcdic_nonprint,
    rt_ebcdic_telnet,
    rt_ebcdic_asa,
    rt_image,
    rt_local
  };

  enum filestru {
    fs_file,
    fs_record,
    fs_page
  };

  enum transmode {
    tm_stream,
    tm_block,
    tm_comp
  };

  enum replycodea {
    rca_posprelim    = '1',
    rca_poscomp      = '2',
    rca_posinter     = '3',
    rca_error        = '4',
    rca_negtranscomp = '4',
    rca_negpermcomp  = '5'
  };

  enum replycodeb {
    rcb_syntax       = '0',
    rcb_info         = '1',
    rcb_conn         = '2',
    rcb_auth         = '3',
    rcb_unspec       = '4',
    rcb_filesys      = '5'
  };

  class ftpbuf: public protocol::protocolbuf {
    // the following are used when this is used as a server
    char*           usr;
    char*           password;
    char*           account;
    char            cwd [MAXPATHLEN];
    char            parentdir [MAXPATHLEN];
    filestru   fs;
    transmode  tm;
    sockinetaddr    udata; // user will listen at this addr for data conn.
    int             serverportno;
    char            replycode [5];

    std::ostream*           o;

    replycodea send_cmd (const char* cmd, const char* arg=0);
    replycodea ftpdata (int portno, std::istream* i, std::ostream* out,
			     const char* cmd, const char* arg=0);

    ftpbuf (ftpbuf&);
    ftpbuf& operator = (ftpbuf&);
  public:
    ftpbuf (std::ostream* out = 0);

    replycodea get_response ();
    const char*     reply_code () const { return replycode; }

    replycodea help () { return send_cmd ("HELP"); }
    replycodea noop () { return send_cmd ("NOOP"); }
    replycodea quit () { return send_cmd ("QUIT"); }
    replycodea abort () { return send_cmd ("ABOR"); }
    replycodea user (const char* name) {return send_cmd ("USER", name);}
    replycodea passwd (const char* pw) {return send_cmd ("PASS", pw); }
    replycodea acct (const char* ac) {return send_cmd ("ACCT", ac);}
    replycodea cd   (const char* dir);
    replycodea useraddr (sockinetaddr sa);
    replycodea useraddr (const char* host, int portno);
    replycodea server_port (int portno);
    replycodea rep_type (reptype rt);
    replycodea file_stru (filestru fs);
    replycodea trans_mode  (transmode tm);

    // service commands
    replycodea getfile (const char* rpath, const char* lpath);
    replycodea list (const char* lpath=0, int justnames = 0);
    replycodea putfile (const char* lpath, const char* rpath);
    replycodea putfile (const char* lpath);
    replycodea append  (const char* lpath, const char* rpath);
    replycodea allocate (int numbytes);
    replycodea restart (int marker);
    replycodea rename (const char* rpath, const char* newrpath);
    replycodea rmfile (const char* rpath);
    replycodea rmdir  (const char* rpath);
    replycodea mkdir  (const char* rpath);
    replycodea pwd    () { return send_cmd ("PWD"); }
    replycodea system () { return send_cmd ("SYST"); }
    replycodea status () { return send_cmd ("STAT"); }

    virtual void        serve_clients (int portno = -1);
    virtual const char* rfc_name () const { return "ftp"; }
    virtual const char* rfc_doc  () const { return "rfc959"; }
  };

protected:
  ftp (): std::ios (0) {}

public:
  ftp (std::ostream* out);
  ~ftp () { delete std::ios::rdbuf (); std::ios::init (0); }


  ftpbuf* rdbuf ()       { return (ftpbuf*) protocol::rdbuf (); }
  ftpbuf* operator -> () { return rdbuf (); }
};

#endif // !FTP_H
