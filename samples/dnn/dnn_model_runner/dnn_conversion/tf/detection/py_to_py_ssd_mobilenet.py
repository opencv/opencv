import os
import tarfile
import urllib

DETECTION_MODELS_URL = 'http://download.tensorflow.org/models/object_detection/'


def extract_tf_frozen_graph(model_name, extracted_model_path):
    # define model archive name
    tf_model_tar = model_name + '.tar.gz'
    # define link to retrieve model archive
    model_link = DETECTION_MODELS_URL + tf_model_tar

    tf_frozen_graph_name = 'frozen_inference_graph'

    try:
        urllib.request.urlretrieve(model_link, tf_model_tar)
    except Exception:
        print("TF {} was not retrieved: {}".format(model_name, model_link))
        return

    print("TF {} was retrieved.".format(model_name))

    tf_model_tar = tarfile.open(tf_model_tar)
    frozen_graph_path = ""

    for model_tar_elem in tf_model_tar.getmembers():
        if tf_frozen_graph_name in os.path.basename(model_tar_elem.name):
            tf_model_tar.extract(model_tar_elem, extracted_model_path)
            frozen_graph_path = os.path.join(extracted_model_path, model_tar_elem.name)
            break
    tf_model_tar.close()

    return frozen_graph_path


def main():
    tf_model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    graph_extraction_dir = "./"
    frozen_graph_path = extract_tf_frozen_graph(tf_model_name, graph_extraction_dir)
    print("Frozen graph path for {}: {}".format(tf_model_name, frozen_graph_path))


if __name__ == "__main__":
    main()
