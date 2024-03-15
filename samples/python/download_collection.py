# Script to download Scanned Objects by Google Research dataset and Stanford models
# Distributed by CC-BY 4.0 License

# Dataset page:
# https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research

import sys, json, requests
from pathlib import Path
import zipfile, tarfile, gzip

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    raise Exception("Python 3.5 or greater is required. Try running `python3 download_collection.py`")

verbose = False

collection_name = "Scanned Objects by Google Research"
owner_name = "GoogleResearch"

class ModelData:
    def __init__(self, name : str, description : str, filesize : int, thumb_url : str, categories ) -> None:
        self.name = name
        self.description = description
        self.filesize = filesize
        self.thumb_url = thumb_url
        self.categories = set(categories)

base_url ='https://fuel.gazebosim.org/'
fuel_version = '1.0'

def print_size(num):
    if num < 1024:
        return str(num) + " B"
    elif num < 1 << 20:
        return "%.3f KiB" % (num / 1024)
    elif num < 1 << 30:
        return "%.3f MiB" % (num / (1 << 20))
    else:
        return "%.3f GiB" % (num / (1 << 30))

def download_model(model_name, dir):
    if verbose:
        print()
        print("{}: {}".format(model.name, model.description))
        print("Categories: [", ", ".join(model.categories), "]")
        print("Size:", print_size(model.filesize))

    download_url = base_url + fuel_version + '/{}/models/'.format(owner_name) + model_name + '.zip'

    archive_path = Path(dir) / Path(model_name+'.zip')
    tmp_archive_path   = Path(dir) / Path(model_name+'.zip.tmp')
    mesh_path     = Path(dir) / Path(model_name+'.obj')
    tmp_mesh_path = Path(dir) / Path(model_name+'.obj.tmp')
    mtl_path     = Path(dir) / Path('model.mtl')
    tmp_mtl_path = Path(dir) / Path('model.mtl.tmp')
    texture_path     = Path(dir) / Path('texture.png')
    tmp_texture_path = Path(dir) / Path('texture.png.tmp')

    for tmp in [tmp_archive_path, tmp_mesh_path, tmp_mtl_path, tmp_texture_path]:
        tmp.unlink(missing_ok=True)

    if archive_path.exists():
        if verbose:
            print("Archive exists")
    else:
        print("URL:", download_url)
        attempt = 1
        while True:
            print("download attempt "+str(attempt)+"...", end="")
            try:
                download = requests.get(download_url, stream=True, timeout=5.0)
                break
            except requests.exceptions.Timeout:
                print("Timed out")
                attempt = attempt + 1
        with open(tmp_archive_path, 'wb') as fd:
            for chunk in download.iter_content(chunk_size=1024*1024):
                fd.write(chunk)
                print(".", end="")
        tmp_archive_path.rename(archive_path)
        print("..downloaded")

    with zipfile.ZipFile(archive_path) as z:
        if mesh_path.exists():
            if verbose:
                print("OBJ exists")
        else:
            with open(tmp_mesh_path, 'wb') as f:
                f.write(z.read("meshes/model.obj"))
            tmp_mesh_path.rename(mesh_path)
            print("OBJ unpacked")
        if texture_path.exists():
            if verbose:
                print("Texture exists")
        else:
            with open(tmp_texture_path, 'wb') as f:
                f.write(z.read("materials/textures/texture.png"))
            tmp_texture_path.rename(texture_path)
            print("Texture unpacked")

    if mtl_path.exists():
        if verbose:
            print("Material exists")
    else:
        mtlFile = """
# Copyright 2020 Google LLC.
# 
# This work is licensed under the Creative Commons Attribution 4.0
# International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by/4.0/ or send a letter
# to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
newmtl material_0
# shader_type beckmann
map_Kd texture.png

# Kd: Diffuse reflectivity.
Kd 1.000000 1.000000 1.000000
"""
        with open(tmp_mtl_path, 'xt') as f:
            f.writelines(mtlFile)
        tmp_mtl_path.rename(mtl_path)
        print("Material written")
    return mesh_path, texture_path

def get_thumb(model : ModelData, dir):
    if verbose:
        print(model.name)
    img_url = base_url + fuel_version + model.thumb_url
    img_path = Path(dir) / Path(model.name+'.jpg')
    tmp_path = Path(dir) / Path(model.name+'.jpg.tmp')
    tmp_path.unlink(missing_ok=True)
    if img_path.exists():
        if verbose:
            print("...exists")
    else:
        print("URL:", img_url)
        attempt = 1
        while True:
            print("download attempt "+str(attempt)+"...")
            try:
                download = requests.get(img_url, stream=True, timeout=5.0)
                break
            except requests.exceptions.Timeout:
                print("Timed out")
                attempt = attempt + 1
        with open(tmp_path, 'wb') as fd:
            for chunk in download.iter_content(chunk_size=1024*1024):
                fd.write(chunk)
                print(".", end="")
        tmp_path.rename(img_path)
        print("..downloaded")


def get_content(content_file):
    # Getting model names and URLs
    models_json = []

    # Iterate over the pages
    page = 0
    while True:
        page = page + 1
        next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)
        page_url = base_url + fuel_version + next_url

        print("Gettting page %d..." % page)
        r = requests.get(page_url)

        if not r or not r.text:
            break

        # Convert to JSON
        models_page = json.loads(r.text)

        if not models_page:
            break

        models_json.extend(models_page)

        print(len(models_json), " models")

    json_object = json.dumps(models_json, indent=4)
    with open(content_file, "w") as outfile:
        outfile.write(json_object)

    return models_json


# let's use different chunk sizes to get rid of timeouts
stanford_models = [
["http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz", 1, "bunny/reconstruction/bun_zipper.ply"],
["http://graphics.stanford.edu/pub/3Dscanrep/happy/happy_recon.tar.gz", 1024, "happy_recon/happy_vrip.ply"],
["http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz", 1024, "dragon_recon/dragon_vrip.ply"],
["http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz", 64, ""],
["http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz", 1024, "lucy.ply"],
["http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz", 1024, ""],
["http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_manuscript.ply.gz", 1024, ""],
["http://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz", 1024, ""],
]

def get_stanford_model(url : str, name : str, ext: str, dir : str, chunk_size : int, internal_path : str):
    archive_path     = Path(dir) / Path(name+'.'+ext)
    tmp_archive_path = Path(dir) / Path(name+'.'+ext+'.tmp')
    model_path = Path(dir) / Path(name+'.ply')
    tmp_model_path = Path(dir) / Path(name+'.ply.tmp')

    for tmp in [tmp_archive_path, tmp_model_path]:
        tmp.unlink(missing_ok=True)

    if archive_path.exists():
        if verbose:
            print("Archive exists")
    else:
        print("URL:", url)
        attempt = 1
        while True:
            print("download attempt "+str(attempt)+"...", end="")
            try:
                download = requests.get(url, stream=True, timeout=5.0)
                break
            except requests.exceptions.Timeout:
                print("Timed out")
                attempt = attempt + 1
        with open(tmp_archive_path, 'wb') as fd:
            for chunk in download.iter_content(chunk_size=chunk_size*1024):
                fd.write(chunk)
                print(".", end="")
        tmp_archive_path.rename(archive_path)
        print("..downloaded")

    if model_path.exists():
        if verbose:
            print("Model exists")
    else:
        # to reduce memory footprint for big models
        max_size = 1024*1024*16
        print("Extracting..", end="")
        with open(tmp_model_path, 'xb') as of:
            if ext=="tar.gz":
                tar_obj = tarfile.open(archive_path, 'r', encoding='utf-8', errors='surrogateescape')
                try:
                    reader = tar_obj.extractfile(internal_path)
                    while buf := reader.read(max_size):
                        of.write(buf)
                        print(".", end="")
                except Exception as err:
                    print(err)
                tar_obj.close()
            elif ext=="ply.gz":
                with gzip.open(archive_path) as gz:
                    while buf := gz.read(max_size):
                        of.write(buf)
                        print(".", end="")
        tmp_model_path.rename(model_path)
        print("done")
    return model_path, ""


# ==================================================

dirname = "dlmodels"

all_models = []

print("Getting Google Research models")

content_file = Path(dirname) / Path("content.json")
if content_file.exists():
    with open(content_file, "r") as openfile:
        models_json = json.load(openfile)
else:
    Path(dirname).mkdir(parents=True, exist_ok=True)
    models_json = get_content(content_file)

models = []
for model in models_json:
    model_name = model['name']
    desc  = model['description']
    fsize = model['filesize']
    thumb_url  = model['thumbnail_url']
    if 'categories' in model:
        categories = model['categories']
    else:
        categories = [ ]
    models.append(ModelData(model_name, desc, fsize, thumb_url, categories))

print("Getting thumbnail images")
for model in models:
    get_thumb(model, dirname)

print("Downloading models from the {}/{} collection.".format(owner_name, collection_name))

for model in models:
    model_dir = Path(dirname) / Path(model.name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path, texture_path = download_model(model.name, model_dir)
    all_models.append((model_path, texture_path))

print('Done.')

categories = set()
for model in models:
    for cat in model.categories:
        categories.add(cat)
print("Categories:", categories)
#{'Consumer Goods', 'Bag', 'Car Seat',
# 'Keyboard', 'Media Cases', 'Toys',
# 'Action Figures', 'Bottles and Cans and Cups',
# 'Shoe', 'Legos', 'Hat',
# 'Mouse', 'Headphones', 'Stuffed Toys',
# 'Board Games', 'Camera'}

print("\nGetting Stanford models")

for m in stanford_models:
    url, chunk_size, internal_path = m

    s = url.split("/")[-1].split(".")
    name = "stanford_"+s[0]
    ext = s[1]+"."+s[2]

    if verbose:
        print(name + ":")
    model_dir = Path(dirname) / Path(name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path, texture_path = get_stanford_model(url, name, ext, model_dir, chunk_size, internal_path)
    all_models.append((model_path, texture_path))


