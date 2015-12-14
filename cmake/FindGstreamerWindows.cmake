FILE(TO_CMAKE_PATH "$ENV{GSTREAMER_DIR}" TRY1_DIR)
FILE(TO_CMAKE_PATH "${GSTREAMER_DIR}" TRY2_DIR)
FILE(GLOB GSTREAMER_DIR ${TRY1_DIR} ${TRY2_DIR})

FIND_PATH(GSTREAMER_gst_INCLUDE_DIR gst/gst.h
          PATHS ${GSTREAMER_DIR}/include/gstreamer-1.0 ${GSTREAMER_DIR}/include /usr/local/include/gstreamer-1.0 /usr/include/gstreamer-1.0
          ENV INCLUDE DOC "Directory containing gst/gst.h include file")

FIND_PATH(GSTREAMER_glib_INCLUDE_DIR glib.h
    PATHS ${GSTREAMER_DIR}/include/glib-2.0/
    ENV INCLUDE DOC "Directory containing glib.h include file")

FIND_PATH(GSTREAMER_glibconfig_INCLUDE_DIR glibconfig.h
    PATHS ${GSTREAMER_DIR}/lib/glib-2.0/include
    ENV INCLUDE DOC "Directory containing glibconfig.h include file")

FIND_PATH(GSTREAMER_gstconfig_INCLUDE_DIR gst/gstconfig.h
                                          PATHS ${GSTREAMER_DIR}/lib/gstreamer-1.0/include ${GSTREAMER_DIR}/include ${GSTREAMER_DIR}/lib/include /usr/local/include/gstreamer-1.0 /usr/include/gstreamer-1.0 /usr/local/lib/include/gstreamer-1.0 /usr/lib/include/gstreamer-1.0
                                          ENV INCLUDE DOC "Directory containing gst/gstconfig.h include file")

FIND_LIBRARY(GSTREAMER_gstaudio_LIBRARY NAMES gstaudio libgstaudio-1.0 gstaudio-1.0
                                        PATHS ${GSTREMAER_DIR}/lib ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                        ENV LIB
                                        DOC "gstaudio library to link with"
                                        NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstapp_LIBRARY NAMES gstapp libgstapp-1.0 gstapp-1.0
                                        PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                        ENV LIB
                                        DOC "gstapp library to link with"
                                        NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstbase_LIBRARY NAMES gstbase libgstbase-1.0 gstbase-1.0
                                       PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                       ENV LIB
                                       DOC "gstbase library to link with"
                                       NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GLIB_gstcdda_LIBRARY NAMES gstcdda libgstcdda-1.0 gstcdda-1.0
                                  PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                  ENV LIB
                                  DOC "gstcdda library to link with"
                                  NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstcontroller_LIBRARY NAMES gstcontroller libgstcontroller-1.0 gstcontroller-1.0
                                             PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                             ENV LIB
                                             DOC "gstcontroller library to link with"
                                             NO_SYSTEM_ENVIRONMENT_PATH)


FIND_LIBRARY(GSTREAMER_gstnet_LIBRARY NAMES gstnet libgstnet-1.0 gstnet-1.0
                                      PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                      ENV LIB
                                      DOC "gstnet library to link with"
                                      NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstpbutils_LIBRARY NAMES gstpbutils libgstpbutils-1.0 gstpbutils-1.0
                                          PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                          ENV LIB
                                          DOC "gstpbutils library to link with"
                                          NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstreamer_LIBRARY NAMES gstreamer libgstreamer-1.0 gstreamer-1.0
                                         PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                         ENV LIB
                                         DOC "gstreamer library to link with"
                                         NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstriff_LIBRARY NAMES gstriff libgstriff-1.0 gstriff-1.0
                                             PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                             ENV LIB
                                             DOC "gstriff library to link with"
                                             NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstrtp_LIBRARY NAMES gstrtp libgstrtp-1.0 gstrtp-1.0
                                    PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                    ENV LIB
                                    DOC "gstrtp library to link with"
                                    NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstrtsp_LIBRARY NAMES gstrtsp libgstrtsp-1.0 gstrtsp-1.0
                                       PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                       ENV LIB
                                       DOC "gstrtsp library to link with"
                                       NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstsdp_LIBRARY NAMES gstsdp libgstsdp-1.0 gstsdp-1.0
                                      PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                      ENV LIB
                                      DOC "gstsdp library to link with"
                                      NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gsttag_LIBRARY NAMES gsttag libgsttag-1.0 gsttag-1.0
                                 PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                 ENV LIB
                                 DOC "gsttag library to link with"
                                 NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GSTREAMER_gstvideo_LIBRARY NAMES gstvideo libgstvideo-1.0 gstvideo-1.0
                                        PATHS ${GSTREAMER_DIR}/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/bin ${GSTREAMER_DIR}/bin/bin C:/gstreamer/bin ${GSTREAMER_DIR}/lib ${GSTREAMER_DIR}/win32/lib /usr/local/lib /usr/lib
                                        ENV LIB
                                        DOC "gstvideo library to link with"
                                        NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GLIB_LIBRARY NAMES libglib-2.0 glib-2.0
                PATHS ${GSTREAMER_DIR}/lib
                ENV LIB
                DOC "Glib library"
                NO_SYSTEM_ENVIRONMENT_PATH)

FIND_LIBRARY(GOBJECT_LIBRARY NAMES libobject-2.0 gobject-2.0
                PATHS ${GSTREAMER_DIR}/lib
                ENV LIB
                DOC "Glib library"
                NO_SYSTEM_ENVIRONMENT_PATH)

IF (GSTREAMER_gst_INCLUDE_DIR AND GSTREAMER_gstconfig_INCLUDE_DIR AND
    GSTREAMER_gstaudio_LIBRARY AND GSTREAMER_gstbase_LIBRARY AND GSTREAMER_gstcontroller_LIBRARY AND GSTREAMER_gstnet_LIBRARY
    AND GSTREAMER_gstpbutils_LIBRARY AND GSTREAMER_gstreamer_LIBRARY AND
    GSTREAMER_gstriff_LIBRARY AND GSTREAMER_gstrtp_LIBRARY AND GSTREAMER_gstrtsp_LIBRARY AND GSTREAMER_gstsdp_LIBRARY AND
    GSTREAMER_gsttag_LIBRARY AND GSTREAMER_gstvideo_LIBRARY AND GLIB_LIBRARY AND GSTREAMER_gstapp_LIBRARY AND GOBJECT_LIBRARY)
  SET(GSTREAMER_INCLUDE_DIR ${GSTREAMER_gst_INCLUDE_DIR} ${GSTREAMER_gstconfig_INCLUDE_DIR} ${GSTREAMER_glib_INCLUDE_DIR} ${GSTREAMER_glibconfig_INCLUDE_DIR})

  list(REMOVE_DUPLICATES GSTREAMER_INCLUDE_DIR)
  SET(GSTREAMER_LIBRARIES ${GSTREAMER_gstaudio_LIBRARY} ${GSTREAMER_gstbase_LIBRARY}
                          ${GSTREAMER_gstcontroller_LIBRARY} ${GSTREAMER_gstdataprotocol_LIBRARY} ${GSTREAMER_gstinterfaces_LIBRARY}
                          ${GSTREAMER_gstnet_LIBRARY} ${GSTREAMER_gstpbutils_LIBRARY}
                          ${GSTREAMER_gstreamer_LIBRARY} ${GSTREAMER_gstriff_LIBRARY} ${GSTREAMER_gstrtp_LIBRARY}
                          ${GSTREAMER_gstrtsp_LIBRARY} ${GSTREAMER_gstsdp_LIBRARY} ${GSTREAMER_gsttag_LIBRARY} ${GSTREAMER_gstvideo_LIBRARY} ${GLIB_LIBRARY}
                          ${GSTREAMER_gstapp_LIBRARY} ${GOBJECT_LIBRARY})

  list(REMOVE_DUPLICATES GSTREAMER_LIBRARIES)
  SET(GSTREAMER_FOUND TRUE)
ENDIF (GSTREAMER_gst_INCLUDE_DIR AND GSTREAMER_gstconfig_INCLUDE_DIR AND
       GSTREAMER_gstaudio_LIBRARY AND GSTREAMER_gstbase_LIBRARY AND GSTREAMER_gstcontroller_LIBRARY
       AND GSTREAMER_gstnet_LIBRARY AND GSTREAMER_gstpbutils_LIBRARY AND GSTREAMER_gstreamer_LIBRARY AND
       GSTREAMER_gstriff_LIBRARY AND GSTREAMER_gstrtp_LIBRARY AND GSTREAMER_gstrtsp_LIBRARY AND GSTREAMER_gstsdp_LIBRARY AND
       GSTREAMER_gsttag_LIBRARY AND GSTREAMER_gstvideo_LIBRARY AND GLIB_LIBRARY AND GSTREAMER_gstapp_LIBRARY AND GOBJECT_LIBRARY)