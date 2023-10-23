####################################################################################
# This is the multi-processor version of
# https://github.com/vitoralbiero/face_analysis_pytorch/blob/master/feature_match.py
####################################################################################
import argparse
from datetime import datetime
from os import makedirs, path
from multiprocessing import Pool
import os

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Matcher:
    def __init__(self, probe_path, gallery_path, dataset_name):
        # lenght of ids to get from feature files
        self.id_length = -1
        self.core_num = os.cpu_count() // 2
        print(self.core_num)
        self.dataset_name = dataset_name
        # load features, subject ids, feature labels from probe file
        probe_file = np.sort(np.asarray(pd.read_csv(probe_path, header=None)).squeeze())
        self.probe_file = probe_file
        print("Collecting features...")
        self.probe, self.probe_ids, self.probe_labels = self.run_parallel(probe_file)

        if gallery_path is not None:
            print(f"Matching {probe_path} to {gallery_path}")
            gallery_file = np.sort(np.loadtxt(args.gallery, dtype=str))
            # if matching different files, load gallery features, ids and labels
            self.probe_equal_gallery = False
            self.gallery, self.gallery_ids, self.gallery_labels = self.run_parallel(
                gallery_file
            )
        else:
            print(f"Matching {probe_path} to {probe_path}")
            # if matching to the same file, just create a simbolic link to save memory
            self.probe_equal_gallery = True
            self.gallery = self.probe
            self.gallery_ids = self.probe_ids
            self.gallery_labels = self.probe_labels

        # initiate a matrix NxM with zeros representing impostor matches
        self.authentic_impostor = np.zeros(shape=(len(self.probe), len(self.gallery)))
        print(f"{self.authentic_impostor.shape} initialized...")
        for i in tqdm(range(len(self.probe))):
            # convert authentic matches to 1
            self.authentic_impostor[i, self.probe_ids[i] == self.gallery_ids] = 1

            # remove same feature files
            self.authentic_impostor[i, self.probe_labels[i] == self.gallery_labels] = -1

            if gallery_path is None:
                # remove duplicate matches if matching probe to probe
                self.authentic_impostor[i, 0: min(i + 1, len(self.gallery))] = -1
        self.current_file = None
        self.matches = None

    def get_features_label(self, feature_path):
        subject_id = path.split(feature_path)[1]
        feature_label = path.join(
            path.split(path.split(feature_path)[0])[1], subject_id[:-4]
        )

        if self.dataset_name == "CHIYA":
            subject_id = subject_id[:-5]

        elif self.dataset_name == "CHIYA_VAL":
            subject_id = feature_label[1:-4]

        elif self.dataset_name == "AGEDB":
            subject_id = subject_id.split("_")[1]

        elif (
                self.dataset_name == "PUBLIC_IVS"
                or self.dataset_name == "VGGFACE2"
                or self.dataset_name == "ASIANCELEB"
                or self.dataset_name == "BA-TEST"
                or self.dataset_name == "BFW"
        ):
            subject_id = path.split(feature_label)[0]

        elif self.id_length > 0:
            subject_id = subject_id[: self.id_length]
        else:
            subject_id = subject_id.split("_")[0]

        return subject_id, feature_label

    def run_parallel(self, file):
        self.current_file = file
        all_features = []
        all_labels = []
        all_subject_ids = []
        pool = Pool(self.core_num)
        indices = np.linspace(0, len(self.current_file) - 1, len(self.current_file)).astype(int)
        for result in tqdm(pool.map(self.get_features, indices)):
            if result is not None:
                all_features.append(result[0])
                all_subject_ids.append(result[2])
                all_labels.append(result[1])
        return (
            np.asarray(all_features),
            np.asarray(all_subject_ids),
            np.asarray(all_labels),
        )

    def get_features(self, indices):
        image_path = self.current_file[indices]
        features = np.load(image_path)
        subject_id, feature_label = self.get_features_label(image_path)
        return features, feature_label, subject_id

    def match_features(self):
        print("Start matching")
        self.matches = cosine_similarity(self.probe, self.gallery).astype(float)
        print("Done!")

    def create_label_indices(self, labels):
        indices = np.linspace(0, len(labels) - 1, len(labels)).astype(int)
        return np.transpose(np.vstack([indices, labels]))

    def get_indices_score(self, auth_or_imp):
        x, y = np.where(self.authentic_impostor == auth_or_imp)
        return np.transpose(
            np.vstack(
                [
                    x,
                    y,
                    np.round(self.matches[self.authentic_impostor == auth_or_imp], 6),
                ]
            )
        )

    def save_matches(self, output, group):
        print("Saving genuine..")
        np.save(path.join(output, f"{group}_genuine.npy"), self.get_indices_score(1))
        print("Saving impostor..")
        np.save(path.join(output, f"{group}_impostor.npy"), self.get_indices_score(0))
        print("Saving indexes..")
        np.savetxt(
            path.join(output, f"{group}_labels.txt"),
            self.create_label_indices(self.probe_labels),
            delimiter=" ",
            fmt="%s",
        )
        if not self.probe_equal_gallery:
            np.savetxt(
                path.join(output, f"{group}_gallery_labels.txt"),
                self.create_label_indices(self.gallery_labels),
                delimiter=" ",
                fmt="%s",
            )
        print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match Extracted Features")
    parser.add_argument("-probe", "-p", help="Probe feature list.")
    parser.add_argument("-gallery", "-g", help="Gallery feature list.")
    parser.add_argument("-output", "-o", help="Output folder.")
    parser.add_argument("-dataset", "-d", help="Dataset name.")
    parser.add_argument("-group", "-gr", help="Group name, e.g. AA")

    args = parser.parse_args()
    time1 = datetime.now()

    # remove if already exists, to get new copy
    # shutil.rmtree(args.output, ignore_errors=True)
    # makedirs(args.output)
    if not path.exists(args.output):
        makedirs(args.output)

    matcher = Matcher(args.probe, args.gallery, args.dataset.upper())
    matcher.match_features()
    matcher.save_matches(args.output, args.group)
    time2 = datetime.now()
    print(time2 - time1)
