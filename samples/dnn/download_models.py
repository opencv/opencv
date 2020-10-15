'''
Helper module to download extra data from Internet
'''
from common import Model, GDriveLoader, URLLoader
import xml.etree.ElementTree as ET

# Models for samples that should be downloaded by default

def produceModel(model_name, filename, sha, url, save_dir, download_name=None, download_sha=None):
    if download_name is None:
        download_name = filename
    if download_sha is None:
        download_sha = sha
    if "drive.google.com" in url:
        token = ""
        token_part = url.rsplit('/', 1)[-1]
        if "&id=" not in token_part:
            token_part = url.rsplit('/', 1)[-2]
        for param in token_part.split("&"):
            if param.startswith("id="):                
                token = param[3:]
        if token:
            loader = GDriveLoader(download_name, download_sha, token)
        else:
            print("Warning: possibly wrong Google Drive link")
            loader = URLLoader(download_name, download_sha, url)
    else:
        loader = URLLoader(download_name, download_sha, url)
    return Model(
        name=model_name,
        filenames=[filename],
        shas=[sha],
        save_dir=save_dir,
        loader=loader
    )

def parseMetalinkFile(metalink_filepath, save_dir):
    NS = {'ml': 'urn:ietf:params:xml:ns:metalink'}
    models = []
    for file_elem in ET.parse(metalink_filepath).getroot().findall('ml:file', NS):
        url = file_elem.find('ml:url', NS).text
        fname = file_elem.attrib['name']
        name = file_elem.find('ml:identity', NS).text
        hash_sum = file_elem.find('ml:hash', NS).text
        models.append(produceModel(name, fname, hash_sum, url, save_dir))
    return models

if __name__ == '__main__':
    models = []
    models.extend(parseMetalinkFile('face_detector/weights.meta4', "."))
    for model in models:
        model.get()