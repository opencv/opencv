# Copyright 2008 Emanuele Rocca <ema@galliera.it>
# Copyright 2008 Marco De Benedetto <debe@galliera.it>
# Copyright (c) 2006-2011 Mathieu Malaterre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DICOM data access layer.

Write your own __getfile function to suit your needs.
"""

from lib import *
import images

def __getfile(studyUID, seriesUID, objectUID):
    """Given a studyUID, a seriesUID and an objectUID, this function must
    retrieve (somehow) the corresponding DICOM file and return the filename.
    Take this implementation as a rough example."""

    #tmpdir = "/var/spool/dicom"
    tmpdir = "/tmp/dicom"

    import os
    import errno
    import tempfile

    studydir = "%s/%s" % (tmpdir, studyUID)
    seriesdir = "%s/%s" % (studydir, seriesUID)
    objectfile = "%s/%s" % (seriesdir, objectUID)

    if not os.path.isdir(seriesdir):
        dump_file = tempfile.NamedTemporaryFile()
        dump_file.write("(0008,0052) CS [STUDY]\n(0020,000d) UI [%s]" % studyUID)
        dump_file.flush()

        dicom_filename = dump_file.name + ".dcm"
        trycmd("dump2dcm %s %s" % (dump_file.name, dicom_filename))

        # Si chiede all'aetitle DW_AM (PACS) di muovere su webapps
        # Per come e' definito il nodo webapps sul PACS (DW_AM) lo studio
        # viene inviato sulla porta 3000
        # Deve essere attivo il servizio (!!! avviato da www-data !!!) simple_storage:
        #   simple_storage -s -x /var/spool/dicom -n /etc/dicom/naming-convention -c webapps -f 3000
        # Le immagini DICOM vengono archiviate in /var/spool/dicom/studyUID/seriersUID/objectUID
        # come descritto in /etc/dicom/naming-convention
        # Il servizio simple_storage forka ed e' quindi possibile ricevere piu'
        # studi contemporaneamente

        movecmd = "movescu --study --aetitle webapps --move webapps --call DW_AM pacs.ceed 3102"
        trycmd("%s %s" % (movecmd, dicom_filename))

        dump_file.close()
        os.unlink(dicom_filename)
    return objectfile

def get(studyUID, seriesUID, objectUID, format='jpeg'):
    """Function called by the main program to get an image."""
    objectfile = __getfile(studyUID, seriesUID, objectUID)
    return images.Dicom(objectfile, format)

if __name__ == "__main__":
    print get(studyUID="1.3.76.13.10010.0.5.74.3996.1224256625.4053",
        seriesUID="1.3.12.2.1107.5.4.4.1053.30000008100608242373400002493",
        objectUID="1.3.12.2.1107.5.4.4.1053.30000008100608324685900001822")
