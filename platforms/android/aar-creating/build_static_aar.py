import argparse
import json
from os import path
import os
import shutil
import string
import subprocess

OPENCV_VERSION = "4.8.0"

ABIS = ["arm64-v8a", "armeabi-v7a", "x86", "x86_64"] # if you want to change it, you also need to change Android project template

ANDROID_PROJECT_TEMPLATE_DIR = "OpenCVAndroidProject"
ANDROID_PROJECT_DIR = "build_static/AndroidProject"
COMPILED_AAR_PATH = path.join(ANDROID_PROJECT_DIR, "OpenCV/build/outputs/aar/opencv-release.aar")
AAR_UNZIPPED_DIR = "build_static/aar_unzipped"
FINAL_AAR_PATH = "outputs/opencv_static.aar"
FINAL_REPO_PATH = "outputs/maven_repo"

def fill_template(src_path, dst_path, args_dict):
    with open(src_path, "r") as f:
        template_text = f.read()
    template = string.Template(template_text)
    text = template.safe_substitute(args_dict)
    with open(dst_path, "w") as f:
        f.write(text)

def get_list_of_opencv_libs(sdk_dir):
    files = os.listdir(path.join(sdk_dir, "sdk/native/staticlibs/arm64-v8a"))
    libs = [f[3:-2] for f in files if f[:3] == "lib" and f[-2:] == ".a"]
    return libs

def get_list_of_3rdparty_libs(sdk_dir):
    libs = []
    for abi in ABIS:
        files = os.listdir(path.join(sdk_dir, "sdk/native/3rdparty/libs/" + abi))
        cur_libs = [f[3:-2] for f in files if f[:3] == "lib" and f[-2:] == ".a"]
        for lib in cur_libs:
            if lib not in libs:
                libs.append(lib)
    return libs

def add_printing_linked_libs(sdk_dir, opencv_libs):
    sdk_jni_dir = sdk_dir + "/sdk/native/jni"
    with open(path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/CMakeLists.txt"), "a") as f:
        f.write('\nset(OpenCV_DIR "' + sdk_jni_dir + '")\n')
        f.write('find_package(OpenCV REQUIRED)\n')
        for lib_name in opencv_libs:
            output_filename_prefix = "linkedlibs." + lib_name + "."
            f.write('get_target_property(OUT "' + lib_name + '" INTERFACE_LINK_LIBRARIES)\n')
            f.write('file(WRITE "' + output_filename_prefix + '${ANDROID_ABI}.txt" "${OUT}")\n')

def read_linked_libs(lib_name):
    deps_lists = []
    for abi in ABIS:
         with open(path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp", f"linkedlibs.{lib_name}.{abi}.txt")) as f:
            text = f.read()
            linked_libs = text.split(";")
            linked_libs = [x.replace("$<LINK_ONLY:", "").replace(">", "") for x in linked_libs]
            deps_lists.append(linked_libs)

    return merge_dependencies_lists(deps_lists)

def merge_dependencies_lists(deps_lists):
    result = []
    for d_list in deps_lists:
        for i in range(len(d_list)):
            if d_list[i] not in result:
                if i == 0:
                    result.append(d_list[i])
                else:
                    index = result.index(d_list[i-1])
                    result = result[:index + 1] + [d_list[i]] + result[index + 1:]

    return result

def convert_deps_list_to_prefab(linked_libs, opencv_libs, external_libs):
    prefab_linked_libs = []
    for lib in linked_libs:
        if (lib in opencv_libs) or (lib in external_libs):
            prefab_linked_libs.append(":" + lib)
        elif (lib[:3] == "lib" and lib[3:] in external_libs):
            prefab_linked_libs.append(":" + lib[3:])
        elif lib == "ocv.3rdparty.android_mediandk":
            prefab_linked_libs += ["-landroid", "-llog", "-lmediandk"]
        elif lib.startswith("ocv.3rdparty"):
            raise Exception("Unknown lib " + lib)
        else:
            prefab_linked_libs.append("-l" + lib)
    return prefab_linked_libs

def main(sdk_dir, opencv_version):
    print("Preparing Android project...")
    shutil.copytree(ANDROID_PROJECT_TEMPLATE_DIR, ANDROID_PROJECT_DIR)
        
    fill_template(path.join(ANDROID_PROJECT_DIR, "OpenCV/build.gradle.template"),
                  path.join(ANDROID_PROJECT_DIR, "OpenCV/build.gradle"),
                  {"LIB_NAME": "templib", "LIB_TYPE": "c++_static", "PACKAGE_NAME": "opencv-static", "OPENCV_VERSION": opencv_version})
    fill_template(path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/CMakeLists.txt.template"),
                  path.join(ANDROID_PROJECT_DIR, "OpenCV/src/main/cpp/CMakeLists.txt"),
                  {"LIB_NAME": "templib", "LIB_TYPE": "STATIC"})
        
    opencv_libs = get_list_of_opencv_libs(sdk_dir)
    external_libs = get_list_of_3rdparty_libs(sdk_dir)

    add_printing_linked_libs(sdk_dir, opencv_libs)

    print("Running gradle assembleRelease...")
    subprocess.run(["gradlew", "assembleRelease"],
                   shell=True,
                   cwd=ANDROID_PROJECT_DIR,
                   check=True)

    shutil.unpack_archive(COMPILED_AAR_PATH, AAR_UNZIPPED_DIR, "zip")

    print("Adding libs to AAR...")

    for lib in external_libs:
        for abi in ABIS:
            os.makedirs(path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/libs/android." + abi))
            if path.exists(path.join(sdk_dir, "sdk/native/3rdparty/libs/" + abi, "lib" + lib + ".a")):
                shutil.copy(path.join(sdk_dir, "sdk/native/3rdparty/libs/" + abi, "lib" + lib + ".a"),
                            path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/libs/android." + abi, "lib" + lib + ".a"))
            else:
                shutil.copy(path.join(AAR_UNZIPPED_DIR, "prefab/modules/templib/libs/android." + abi, "libtemplib.a"),
                            path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/libs/android." + abi, "lib" + lib + ".a"))
            shutil.copy(path.join(AAR_UNZIPPED_DIR, "prefab/modules/templib/libs/android." + abi + "/abi.json"),
                        path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/libs/android." + abi + "/abi.json"))
        shutil.copy(path.join(AAR_UNZIPPED_DIR, "prefab/modules/templib/module.json"),
                    path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/module.json"))

    for lib in opencv_libs:
        for abi in ABIS:
            os.makedirs(path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/libs/android." + abi))
            shutil.copy(path.join(sdk_dir, "sdk/native/staticlibs/" + abi, "lib" + lib + ".a"),
                        path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/libs/android." + abi, "lib" + lib + ".a"))
            shutil.copy(path.join(AAR_UNZIPPED_DIR, "prefab/modules/templib/libs/android." + abi + "/abi.json"),
                        path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/libs/android." + abi + "/abi.json"))
        os.makedirs(path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/include/opencv2"))
        shutil.copy(path.join(sdk_dir, "sdk/native/jni/include/opencv2/" + lib.replace("opencv_", "") + ".hpp"),
                    path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/include/opencv2/" + lib.replace("opencv_", "") + ".hpp"))
        shutil.copytree(path.join(sdk_dir, "sdk/native/jni/include/opencv2/" + lib.replace("opencv_", "")),
                        path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/include/opencv2/" + lib.replace("opencv_", "")))

        module_json_text = {
            "export_libraries": convert_deps_list_to_prefab(read_linked_libs(lib), opencv_libs, external_libs),
            "android": {},
        }
        with open(path.join(AAR_UNZIPPED_DIR, "prefab/modules/" + lib + "/module.json"), "w") as f:
            json.dump(module_json_text, f)

    for h_file in ("cvconfig.h", "opencv.hpp", "opencv_modules.hpp"):
        shutil.copy(path.join(sdk_dir, "sdk/native/jni/include/opencv2/" + h_file),
                    path.join(AAR_UNZIPPED_DIR, "prefab/modules/opencv_core/include/opencv2/" + h_file))


    shutil.rmtree(path.join(AAR_UNZIPPED_DIR, "prefab/modules/templib"))

    os.makedirs("outputs", exist_ok=True)
    shutil.make_archive(FINAL_AAR_PATH, "zip", AAR_UNZIPPED_DIR, ".")
    os.rename(FINAL_AAR_PATH + ".zip", FINAL_AAR_PATH)

    print("Creating local maven repo...")

    shutil.copy(FINAL_AAR_PATH, path.join(ANDROID_PROJECT_DIR, "OpenCV/opencv-release.aar"))

    subprocess.run(["gradlew", "publishReleasePublicationToMyrepoRepository"],
            shell=True,
            cwd=ANDROID_PROJECT_DIR,
            check=True)
    
    os.makedirs(path.join(FINAL_REPO_PATH, "org/opencv"), exist_ok=True)
    shutil.move(path.join(ANDROID_PROJECT_DIR, "OpenCV/build/repo/org/opencv/opencv-static"),
                path.join(FINAL_REPO_PATH, "org/opencv/opencv-static"))
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds AAR with static C++ libs from OpenCV SDK")
    parser.add_argument('opencv_sdk_path')
    parser.add_argument('opencv_version')
    args = parser.parse_args()

    main(args.opencv_sdk_path, args.opencv_version)
