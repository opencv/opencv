This is the source code of GDCM. It is available from sf.net website.
Official GIT repository is at:

  https://sourceforge.net/p/gdcm/gdcm/

For a general introduction/features/limitations/requirement please
refer to

  http://gdcm.sourceforge.net/

Just a quick note on the build process of GDCM. GDCM build process
make use of the cmake software(*). This allow us:
1. To get rid of the autoconf/autotools insanity
2. Transparently generate Unix Makefiles, NMake Makefiles,
VS8/9/10 Solution, XCode 2.1...
3. Automatic nightly testing, one of the most important thing
for a robust library/software devlpt process. GDCM devpt is develop
based on the XP definition, and to preserve backward compatibility
make sure that code is working from one release to another: each night
we configure, we build and we test GDCM. The result are then send to
the dashboard located at:

   http://public.kitware.com/dashboard.php?name=gdcm

A continuous dashboard make also sure that any commit did not introduce
any error on another plateform, a warning or broke a test...

Therefore you should be able to use GDCM from the bleeding edge without
knowing too much on what is going on. All you need to do is have a look
at the GDCM dashboard, and if your plateform is 'green' then you can
update your git copy and compile safely knowing that there are very few chances
that something won't work. Cheers !


(*) http://www.cmake.org for more information

For more help you can go online in the GDCM Wiki:
* http://gdcm.sourceforge.net/

In Particular:
* http://gdcm.sourceforge.net/wiki/index.php/GDCM_Release_2.0
* http://gdcm.sourceforge.net/wiki/index.php/FAQ

And a page describing each tool can be found at:
* http://gdcm.sourceforge.net/wiki/index.php/End_User_Applications

Eg:
* http://gdcm.sourceforge.net/wiki/index.php/Gdcminfo

Need VTK:
* http://gdcm.sourceforge.net/wiki/index.php/Gdcmviewer
