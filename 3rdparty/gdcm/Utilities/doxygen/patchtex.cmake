project(toto)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/refman.tex refman_file)
string(REPLACE "]{hyperref}"
"]{hyperref}\\\\hypersetup{pdftitle={GDCM Reference Guide},pdfkeywords={DICOM},baseurl={http:\\/\\/gdcm.sourceforge.net}}\\\\hyperbaseurl{http:\\/\\/gdcm.sourceforge.net}"
patched_refman_file
${refman_file}
)

file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/patched.tex ${patched_refman_file})
