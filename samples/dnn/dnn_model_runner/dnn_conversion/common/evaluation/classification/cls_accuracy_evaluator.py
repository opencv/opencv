import sys
import time

import numpy as np

from ...utils import get_final_summary_info


class ClsAccEvaluation:
    log = sys.stdout
    img_classes = {}
    batch_size = 0

    def __init__(self, log_path, img_classes_file, batch_size):
        self.log = open(log_path, 'w')
        self.img_classes = self.read_classes(img_classes_file)
        self.batch_size = batch_size

        # collect the accuracies for both models
        self.general_quality_metric = []
        self.general_inference_time = []

    @staticmethod
    def read_classes(img_classes_file):
        result = {}
        with open(img_classes_file) as file:
            for l in file.readlines():
                result[l.split()[0]] = int(l.split()[1])
        return result

    def get_correct_answers(self, img_list, net_output_blob):
        correct_answers = 0
        for i in range(len(img_list)):
            indexes = np.argsort(net_output_blob[i])[-5:]
            correct_index = self.img_classes[img_list[i]]
            if correct_index in indexes:
                correct_answers += 1
        return correct_answers

    def process(self, frameworks, data_fetcher):
        sorted_imgs_names = sorted(self.img_classes.keys())
        correct_answers = [0] * len(frameworks)
        samples_handled = 0
        blobs_l1_diff = [0] * len(frameworks)
        blobs_l1_diff_count = [0] * len(frameworks)
        blobs_l_inf_diff = [sys.float_info.min] * len(frameworks)
        inference_time = [0.0] * len(frameworks)

        for x in range(0, len(sorted_imgs_names), self.batch_size):
            sublist = sorted_imgs_names[x:x + self.batch_size]
            batch = data_fetcher.get_batch(sublist)

            samples_handled += len(sublist)
            fw_accuracy = []
            fw_time = []
            frameworks_out = []
            for i in range(len(frameworks)):
                start = time.time()
                out = frameworks[i].get_output(batch)
                end = time.time()
                correct_answers[i] += self.get_correct_answers(sublist, out)
                fw_accuracy.append(100 * correct_answers[i] / float(samples_handled))
                frameworks_out.append(out)
                inference_time[i] += end - start
                fw_time.append(inference_time[i] / samples_handled * 1000)
                print(samples_handled, 'Accuracy for', frameworks[i].get_name() + ':', fw_accuracy[i], file=self.log)
                print("Inference time, ms ", frameworks[i].get_name(), fw_time[i], file=self.log)

                self.general_quality_metric.append(fw_accuracy)
                self.general_inference_time.append(fw_time)

            for i in range(1, len(frameworks)):
                log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
                diff = np.abs(frameworks_out[0] - frameworks_out[i])
                l1_diff = np.sum(diff) / diff.size
                print(samples_handled, "L1 difference", log_str, l1_diff, file=self.log)
                blobs_l1_diff[i] += l1_diff
                blobs_l1_diff_count[i] += 1
                if np.max(diff) > blobs_l_inf_diff[i]:
                    blobs_l_inf_diff[i] = np.max(diff)
                print(samples_handled, "L_INF difference", log_str, blobs_l_inf_diff[i], file=self.log)

            self.log.flush()

        for i in range(1, len(blobs_l1_diff)):
            log_str = frameworks[0].get_name() + " vs " + frameworks[i].get_name() + ':'
            print('Final l1 diff', log_str, blobs_l1_diff[i] / blobs_l1_diff_count[i], file=self.log)

        print(
            get_final_summary_info(
                self.general_quality_metric,
                self.general_inference_time,
                "accuracy"
            ),
            file=self.log
        )
