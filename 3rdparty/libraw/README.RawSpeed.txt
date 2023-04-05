================= Compile LibRaw with RawSpeed support ========================

0) RawSpeed version: 
   LibRaw supports 'master' version of RawSpeed library: https://github.com/darktable-org/rawspeed/tree/master

   Although this version is really outdated, newer versions does not looks stable enough for production
   use (for example, this critical issue not fixed for 10+ months: https://github.com/darktable-org/rawspeed/issues/100 )
   
1) Prerequisites

To build RawSpeed you need libxml2, iconv, and JPEG library installed on your 
system.

2) Build RawSpeed:

  -- consult http://rawstudio.org/blog/?p=800 for details

  -- Win32: you need POSIX Threads for Win32 installed on your system
     (http://sources.redhat.com/pthreads-win32/)

  -- use provided RawSpeed/rawspeed.samsung-decoder.patch  to fix
     old Samsung decoder bug

  -- you may use qmake .pro files supplied in LibRaw distribution
     (RawSpeed/rawspeed.qmake-pro-files.patch)
     Adjust path to libraries/includes according to your setup.

  -- Win32: you need to add __declspec(..) to external C++ classes.
     Use patch provided with LibRaw (RawSpeed/rawspeed.win32-dll.patch)

  -- Unix: you need to define rawspeed_get_number_of_processor_cores() call
     For most unix systems (Linux, MacOS X 10.4+, FreeBSD) patch provided
     with LibRaw (RawSpeed/rawspeed.cpucount-unix.patch) should work.

3) Build LibRaw with RawSpeed support:
 
   Win32: 
     --Uncomment CFLAGS_RAWSPEED and LDFLAGS_RAWSPEED lines in
       Makefile.msvc. Adjust paths to libraries/includes if needed.
     -- run nmake -f Makefile.msvc

   Unix/MacOS:
     -- Uncomment CFLAGS/LDADD lines in RawSpeed section in Makefile.dist
     -- Uncomment RAWSPEED_DATA line if you wish to rebuild
	internal copy of RawSpeed's cameras.xml
     -- run make -f Makefile.dist

   Compile options:
    -- You may specify -DNOSONY_RAWSPEED define if you do not want to use 
        RawSpeed's Sony formats decoder (because result of this decoder is 
        different from  LibRaw's built-in decoder)

4) Build/run your Apps with LibRaw+RawSpeed
   
   -- Build as usual, no changes required in your apps unless you
      access LibRaw::imgdata.rawdata.raw_image[] directly

   -- you may turn off RawSpeed support on runtime by setting 
      imgdata.params.use_rawspeed to 0.

   -- You'll need all shared libraries you linked to at runtime (libxml2,
      iconv, LibJPEG, and posix threads on Win32).
