'''
Helper module to download extra data from Internet
'''
import os
import sys
import yaml
import argparse
import xml.etree.ElementTree as ET
from common import Model, GDriveLoader, URLLoader

# Models for samples that should be downloaded by default

def produceModel(model_name, filename, sha, url, save_dir, download_name=None, download_sha=None, archive_member=None):
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
            loader = GDriveLoader(download_name, download_sha, token, archive_member)
        else:
            print("Warning: possibly wrong Google Drive link")
            loader = URLLoader(download_name, download_sha, url, archive_member)
    else:
        loader = URLLoader(download_name, download_sha, url, archive_member)
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

def parseYAMLFile(yaml_filepath, save_dir):
    models = []
    with open(yaml_filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
        for name, params in data_loaded.items():
            load_info = params.get("load_info", None)
            if load_info:
                fname = os.path.basename(params.get("model"))
                hash_sum = load_info.get("sha")
                url = load_info.get("url")
                download_sha = load_info.get("download_sha")
                download_name = load_info.get("download_name")
                archive_member = load_info.get("member")
                models.append(produceModel(name, fname, hash_sum, url, save_dir, 
                    download_name=download_name, download_sha=download_sha, archive_member=archive_member))

    return models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--save_dir', action="store", default=os.getcwd(),
                        help='Path to the directory to store downloaded files')
    parser.add_argument('model_name', type=str, default="", nargs='?', action="store",
                        help='name of the model to download')
    args = parser.parse_args()
    models = []
    save_dir = args.save_dir
    selected_model_name = args.model_name
    models.extend(parseMetalinkFile('face_detector/weights.meta4', save_dir))
    models.extend(parseYAMLFile('models.yml', save_dir))
    for m in models:
        print(m)
        if selected_model_name and not m.name.startswith(selected_model_name):
            continue
        print('Model: ' + selected_model_name)
        m.get()