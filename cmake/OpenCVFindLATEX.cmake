# - Find Latex
# This module finds if Latex is installed and determines where the
# executables are. This code sets the following variables:
#
#  LATEX_COMPILER:       path to the LaTeX compiler
#  PDFLATEX_COMPILER:    path to the PdfLaTeX compiler
#  BIBTEX_COMPILER:      path to the BibTeX compiler
#  MAKEINDEX_COMPILER:   path to the MakeIndex compiler
#  DVIPS_CONVERTER:      path to the DVIPS converter
#  PS2PDF_CONVERTER:     path to the PS2PDF converter
#  LATEX2HTML_CONVERTER: path to the LaTeX2Html converter
#

IF (WIN32)

  # Try to find the MikTex binary path (look for its package manager).

  FIND_PATH(MIKTEX_BINARY_PATH mpm.exe
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MiK\\MiKTeX\\CurrentVersion\\MiKTeX;Install Root]/miktex/bin"
    DOC
    "Path to the MikTex binary directory."
  )
  MARK_AS_ADVANCED(MIKTEX_BINARY_PATH)

  # Try to find the GhostScript binary path (look for gswin32).

  GET_FILENAME_COMPONENT(GHOSTSCRIPT_BINARY_PATH_FROM_REGISTERY_8_00
     "[HKEY_LOCAL_MACHINE\\SOFTWARE\\AFPL Ghostscript\\8.00;GS_DLL]" PATH
  )

  GET_FILENAME_COMPONENT(GHOSTSCRIPT_BINARY_PATH_FROM_REGISTERY_7_04
     "[HKEY_LOCAL_MACHINE\\SOFTWARE\\AFPL Ghostscript\\7.04;GS_DLL]" PATH
  )

  FIND_PATH(GHOSTSCRIPT_BINARY_PATH gswin32.exe
    ${GHOSTSCRIPT_BINARY_PATH_FROM_REGISTERY_8_00}
    ${GHOSTSCRIPT_BINARY_PATH_FROM_REGISTERY_7_04}
    DOC "Path to the GhostScript binary directory."
  )
  MARK_AS_ADVANCED(GHOSTSCRIPT_BINARY_PATH)

  FIND_PATH(GHOSTSCRIPT_LIBRARY_PATH ps2pdf13.bat
    "${GHOSTSCRIPT_BINARY_PATH}/../lib"
    DOC "Path to the GhostScript library directory."
  )
  MARK_AS_ADVANCED(GHOSTSCRIPT_LIBRARY_PATH)

ENDIF (WIN32)

FIND_HOST_PROGRAM(LATEX_COMPILER
  NAMES latex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin /usr/texbin
)

FIND_HOST_PROGRAM(PDFLATEX_COMPILER
  NAMES pdflatex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin /usr/texbin
)

FIND_HOST_PROGRAM(BIBTEX_COMPILER
  NAMES bibtex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin /usr/texbin
)

FIND_HOST_PROGRAM(MAKEINDEX_COMPILER
  NAMES makeindex
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin /usr/texbin
)

FIND_HOST_PROGRAM(DVIPS_CONVERTER
  NAMES dvips
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin /usr/texbin
)

FIND_HOST_PROGRAM(DVIPDF_CONVERTER
  NAMES dvipdfm dvipdft dvipdf
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin /usr/texbin
)

IF (WIN32)
  FIND_HOST_PROGRAM(PS2PDF_CONVERTER
    NAMES ps2pdf14.bat
    PATHS ${GHOSTSCRIPT_LIBRARY_PATH}
  )
ELSE (WIN32)
  FIND_HOST_PROGRAM(PS2PDF_CONVERTER
    NAMES ps2pdf14 ps2pdf
    PATHS /usr/bin /usr/texbin
  )
ENDIF (WIN32)

FIND_HOST_PROGRAM(LATEX2HTML_CONVERTER
  NAMES latex2html
  PATHS ${MIKTEX_BINARY_PATH}
        /usr/bin /usr/texbin
)


MARK_AS_ADVANCED(
  LATEX_COMPILER
  PDFLATEX_COMPILER
  BIBTEX_COMPILER
  MAKEINDEX_COMPILER
  DVIPS_CONVERTER
  DVIPDF_CONVERTER
  PS2PDF_CONVERTER
  LATEX2HTML_CONVERTER
)
