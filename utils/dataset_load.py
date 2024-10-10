import itertools
import json

import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2
import tensorflow_datasets as tfds
from random import shuffle, choice, randint

tf.random.set_seed(1234)
np.random.seed(1234)


def load_dataset_ci_cartridges(
        image_size,
        class_names_file,
        class_caseids_file,
        dataset_dir,
        file_dataset_composition=None,
        sampling_mode=None,
        train_part=0.85,
        val_part=0.15,
        test_part=0
):
    image_dims = image_size + (3,)
    imagePaths_per_caseid, train_caseids, val_caseids, test_caseids, class_names = (
        get_data_for_datasets_generation(
            class_names_file,
            class_caseids_file,
            dataset_dir,
            file_dataset_composition,
            sampling_mode,
            train_part,
            val_part,
            test_part
        ))

    train_dataset = tf.data.Dataset.from_generator(
        generate_image_label_sample,
        args=([imagePaths_per_caseid[caseid]
               for caseid in imagePaths_per_caseid
               if caseid in train_caseids], image_dims, class_names),
        output_signature=(
            tf.TensorSpec(shape=image_dims, dtype=tf.int32),
            tf.TensorSpec(shape=(10,), dtype=tf.float32)
        )
    )

    validation_dataset = tf.data.Dataset.from_generator(
        generate_image_label_sample,
        args=([imagePaths_per_caseid[caseid]
               for caseid in imagePaths_per_caseid
               if caseid in val_caseids], image_dims, class_names),
        output_signature=(
            tf.TensorSpec(shape=image_dims, dtype=tf.int32),
            tf.TensorSpec(shape=(10,), dtype=tf.float32)
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        generate_image_label_sample,
        args=([imagePaths_per_caseid[caseid]
               for caseid in imagePaths_per_caseid
               if caseid in test_caseids], image_dims, class_names),
        output_signature=(
            tf.TensorSpec(shape=image_dims, dtype=tf.int32),
            tf.TensorSpec(shape=(10,), dtype=tf.float32)
        )
    )

    return train_dataset, test_dataset, validation_dataset, class_names


def load_siamese_dataset_ci_cartridges(
        image_size,
        class_names_file,
        class_caseids_file,
        dataset_dir,
        file_dataset_composition=None,
        sampling_mode=None,
        train_part=0.85,
        val_part=0.15,
        test_part=0
):
    image_dims = image_size + (3,)
    train_imagePaths_per_class, val_imagePaths_per_class, test_imagePaths_per_class, class_names = (
        get_data_for_datasets_generation_per_class(
            class_names_file,
            class_caseids_file,
            dataset_dir,
            file_dataset_composition,
            sampling_mode,
            train_part,
            val_part,
            test_part
        ))

    # train_path_pairs, val_path_pairs, test_path_pairs = get_90_10_siamese_image_label_pairs(train_imagePaths_per_class, val_imagePaths_per_class, test_imagePaths_per_class)
    # train_path_pairs, val_path_pairs, test_path_pairs = get_50_50_siamese_image_label_pairs_large(train_imagePaths_per_class, val_imagePaths_per_class, test_imagePaths_per_class)
    #train_path_pairs, val_path_pairs, test_path_pairs = get_50_50_siamese_image_label_pairs(train_imagePaths_per_class, val_imagePaths_per_class, test_imagePaths_per_class)


    # Shuffle the pairs to have a more randomized dataset
    # shuffle(train_path_pairs)
    # shuffle(val_path_pairs)
    # shuffle(test_path_pairs)

    # train_class_pairs = list(map(lambda el: (el[2], el[3]), train_path_pairs))
    # train_path_pairs = list(map(lambda el: (el[0], el[1]), train_path_pairs))
    #
    # val_class_pairs = list(map(lambda el: (el[2], el[3]), val_path_pairs))
    # val_path_pairs = list(map(lambda el: (el[0], el[1]), val_path_pairs))
    #
    # test_class_pairs = list(map(lambda el: (el[2], el[3]), test_path_pairs))
    # test_path_pairs = list(map(lambda el: (el[0], el[1]), test_path_pairs))

    # TODO: remove this, only fo test
    # train_path_pairs = train_path_pairs[:3000]
    # train_class_pairs = train_class_pairs[:3000]
    #
    # val_path_pairs = val_path_pairs[:1000]
    # val_class_pairs = val_class_pairs[:1000]
    #
    # test_path_pairs = test_path_pairs[:1000]
    # test_class_pairs = test_class_pairs[:1000]

    train_paths, val_paths, test_paths = get_train_test_paths_list(
            class_names_file,
            class_caseids_file,
            dataset_dir,
            sampling_mode,
            train_part,
            val_part,
            test_part
    )

    print("Total Number of training pairs: {}".format(2 * len(train_paths)))
    print("Total Number of validation pairs: {}".format(2 * len(val_paths)))
    print("Total Number of testing pairs: {}".format(2 *len(test_paths)))

    # Generate the dataset objects with the converted images
    train_dataset = tf.data.Dataset.from_generator(
        make_pairs_ci,
        args=(train_paths, class_names, image_dims),
        output_signature=(
            tf.TensorSpec(shape=(2,) + image_dims, dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).map(lambda x, y: ({'img_A_input': x[0], 'img_B_input': x[1]}, y))

    validation_dataset = tf.data.Dataset.from_generator(
        make_pairs_ci,
        args=(val_paths, class_names, image_dims),
        output_signature=(
            tf.TensorSpec(shape=(2,) + image_dims, dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).map(lambda x, y: ({'img_A_input': x[0], 'img_B_input': x[1]}, y))

    test_dataset = tf.data.Dataset.from_generator(
        make_pairs_ci,
        args=(test_paths, class_names, image_dims),
        output_signature=(
            tf.TensorSpec(shape=(2,) + image_dims, dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).map(lambda x, y: ({'img_A_input': x[0], 'img_B_input': x[1]}, y))

    return train_dataset, test_dataset, validation_dataset, ["0", "1"], class_names


def get_data_for_datasets_generation(
        class_names_file,
        class_caseids_file,
        dataset_dir,
        file_dataset_composition=None,
        sampling_mode=None,
        train_part=0.85,
        val_part=0.15,
        test_part=0
):
    df = pd.read_csv(class_names_file)
    class_names = df["Model"].values.tolist()

    f = open(class_caseids_file, 'r')
    caseids_per_class = json.load(f)
    f.close()

    imagePaths = list(list_files(dataset_dir, dirs_to_search=class_names, contains=["_BF"]))

    shuffle(imagePaths)

    # Associate each image path/name to their corresponding case id
    imagePaths_per_caseid = dict()
    for imagePath in imagePaths:
        if "_BF_" in imagePath and "_4_" in imagePath:
            continue

        clazz = imagePath.split('/')[1]
        caseid = list(filter(lambda x: x in imagePath, caseids_per_class[clazz]))
        if len(caseid) == 1:
            caseid = caseid[0]
        elif len(caseid) == 0:
            # print("Skipping " + imagePath + ", no valid case id found for this path")
            continue
        else:
            print("Warning, captured more than one case id")
            continue

        if caseid not in imagePaths_per_caseid:
            imagePaths_per_caseid[caseid] = [imagePath]
            # imagePaths_per_caseid[caseid] = [imagePath.replace("dataset", "dataset_copied").replace(".jp2", ".jpg")]
        else:
            imagePaths_per_caseid[caseid].append(imagePath)
            # imagePaths_per_caseid[caseid].append(imagePath.replace("dataset", "dataset_copied").replace(".jp2", ".jpg"))

    (train_caseids_per_class, val_caseids_per_class, test_caseids_per_class) = (
        split_caseids_per_class_per_group(
            caseids_per_class,
            sampling_mode=sampling_mode,
            train=train_part,
            val=val_part,
            test=test_part
        )
    )

    print(f"Total number of caseids for training: "
          f"{len([caseid for caseids in train_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for training: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in train_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    print(f"Total number of caseids for validation: "
          f"{len([caseid for caseids in val_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for validation: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in val_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    print(f"Total number of caseids for testing: "
          f"{len([caseid for caseids in test_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for testing: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in test_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    max_images_per_caseid = int(
        np.array([len(imagePaths_per_caseid[caseid]) for caseid in imagePaths_per_caseid.keys()]).max())
    for caseid in imagePaths_per_caseid:
        while len(imagePaths_per_caseid[caseid]) < max_images_per_caseid:
            imagePaths_per_caseid[caseid].append("")

    # TODO
    # if file_dataset_composition is not None:
    #     f = open(file_dataset_composition, "w")
    #
    # if file_dataset_composition is not None:
    #     f.write(className + ":\n")
    #     f.write("\tNumber of training cases: " + str(int(num_cases * train_part)) + '\n')
    #     f.write("\tNumber of validation cases: " + str(int(num_cases * (train_part + val_part)) - int(num_cases * train_part)) + '\n')
    #     f.write("\tNumber of test cases: " + str(int(num_cases * test_part)) + '\n')
    #
    # if file_dataset_composition is not None:
    #     f.close()

    train_caseids = [caseid for class_caseids in train_caseids_per_class.values() for caseid in class_caseids]
    val_caseids = [caseid for class_caseids in val_caseids_per_class.values() for caseid in class_caseids]
    test_caseids = [caseid for class_caseids in test_caseids_per_class.values() for caseid in class_caseids]

    return imagePaths_per_caseid, train_caseids, val_caseids, test_caseids, class_names

def get_train_test_paths_list(
        class_names_file,
        class_caseids_file,
        dataset_dir,
        sampling_mode=None,
        train_part=0.85,
        val_part=0.15,
        test_part=0
):
    df = pd.read_csv(class_names_file)
    class_names = df["Model"].values.tolist()

    f = open(class_caseids_file, 'r')
    caseids_per_class = json.load(f)
    f.close()

    imagePaths = list(list_files(dataset_dir, dirs_to_search=class_names, contains=["_BF"]))

    shuffle(imagePaths)

    # Associate each image path/name to their corresponding case id
    imagePaths_per_caseid = dict()
    for imagePath in imagePaths:
        if "_BF_" in imagePath and "_4_" in imagePath:
            continue

        clazz = imagePath.split('/')[1]
        caseid = list(filter(lambda x: x in imagePath, caseids_per_class[clazz]))
        if len(caseid) == 1:
            caseid = caseid[0]
        elif len(caseid) == 0:
            # print("Skipping " + imagePath + ", no valid case id found for this path")
            continue
        else:
            print("Warning, captured more than one case id")
            continue

        if caseid not in imagePaths_per_caseid:
            imagePaths_per_caseid[caseid] = [imagePath]
            # imagePaths_per_caseid[caseid] = [imagePath.replace("dataset", "dataset_copied").replace(".jp2", ".jpg")]
        else:
            imagePaths_per_caseid[caseid].append(imagePath)
            # imagePaths_per_caseid[caseid].append(imagePath.replace("dataset", "dataset_copied").replace(".jp2", ".jpg"))

    (train_caseids_per_class, val_caseids_per_class, test_caseids_per_class) = (
        split_caseids_per_class_per_group(
            caseids_per_class,
            sampling_mode=sampling_mode,
            train=train_part,
            val=val_part,
            test=test_part
        )
    )

    print(f"Total number of caseids for training: "
          f"{len([caseid for caseids in train_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for training: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in train_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    print(f"Total number of caseids for validation: "
          f"{len([caseid for caseids in val_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for validation: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in val_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    print(f"Total number of caseids for testing: "
          f"{len([caseid for caseids in test_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for testing: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in test_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    train_paths = []
    val_paths = []
    test_paths = []

    for clazz, caseids in train_caseids_per_class.items():
        for caseid in caseids:
            if caseid in imagePaths_per_caseid.keys():
                for path in imagePaths_per_caseid[caseid]:
                    train_paths.append(path)

    for clazz, caseids in val_caseids_per_class.items():
        for caseid in caseids:
            if caseid in imagePaths_per_caseid.keys():
                for path in imagePaths_per_caseid[caseid]:
                    val_paths.append(path)

    for clazz, caseids in test_caseids_per_class.items():
        for caseid in caseids:
            if caseid in imagePaths_per_caseid.keys():
                for path in imagePaths_per_caseid[caseid]:
                    test_paths.append(path)

    shuffle(train_paths)
    shuffle(val_paths)
    shuffle(test_paths)
    return train_paths, val_paths, test_paths


def get_data_for_datasets_generation_per_class(
        class_names_file,
        class_caseids_file,
        dataset_dir,
        file_dataset_composition=None,
        sampling_mode=None,
        train_part=0.85,
        val_part=0.15,
        test_part=0
):
    df = pd.read_csv(class_names_file)
    class_names = df["Model"].values.tolist()

    f = open(class_caseids_file, 'r')
    caseids_per_class = json.load(f)
    f.close()

    imagePaths = list(list_files(dataset_dir, dirs_to_search=class_names, contains=["_BF"]))

    shuffle(imagePaths)

    # Associate each image path/name to their corresponding case id
    imagePaths_per_caseid = dict()
    for imagePath in imagePaths:
        if "_BF_" in imagePath and "_4_" in imagePath:
            continue

        clazz = imagePath.split('/')[1]
        caseid = list(filter(lambda x: x in imagePath, caseids_per_class[clazz]))
        if len(caseid) == 1:
            caseid = caseid[0]
        elif len(caseid) == 0:
            # print("Skipping " + imagePath + ", no valid case id found for this path")
            continue
        else:
            print("Warning, captured more than one case id")
            continue

        if caseid not in imagePaths_per_caseid:
            imagePaths_per_caseid[caseid] = [imagePath]
            # imagePaths_per_caseid[caseid] = [imagePath.replace("dataset", "dataset_copied").replace(".jp2", ".jpg")]
        else:
            imagePaths_per_caseid[caseid].append(imagePath)
            # imagePaths_per_caseid[caseid].append(imagePath.replace("dataset", "dataset_copied").replace(".jp2", ".jpg"))

    (train_caseids_per_class, val_caseids_per_class, test_caseids_per_class) = (
        split_caseids_per_class_per_group(
            caseids_per_class,
            sampling_mode=sampling_mode,
            train=train_part,
            val=val_part,
            test=test_part
        )
    )

    print(f"Total number of caseids for training: "
          f"{len([caseid for caseids in train_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for training: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in train_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    print(f"Total number of caseids for validation: "
          f"{len([caseid for caseids in val_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for validation: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in val_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    print(f"Total number of caseids for testing: "
          f"{len([caseid for caseids in test_caseids_per_class.values() for caseid in caseids])}")

    print(f"Total number of images for testing: "
          f"{np.array([len(imagePaths) for (caseid, imagePaths) in imagePaths_per_caseid.items() if caseid in [caseId for caseIds in test_caseids_per_class.values() for caseId in caseIds]]).sum()}")

    # TODO
    # if file_dataset_composition is not None:
    #     f = open(file_dataset_composition, "w")
    #
    # if file_dataset_composition is not None:
    #     f.write(className + ":\n")
    #     f.write("\tNumber of training cases: " + str(int(num_cases * train_part)) + '\n')
    #     f.write("\tNumber of validation cases: " + str(int(num_cases * (train_part + val_part)) - int(num_cases * train_part)) + '\n')
    #     f.write("\tNumber of test cases: " + str(int(num_cases * test_part)) + '\n')
    #
    # if file_dataset_composition is not None:
    #     f.close()

    train_imagePaths_per_class = dict()
    for clazz in class_names:
        train_imagePaths_per_class[clazz] = []

    for clazz, caseids in train_caseids_per_class.items():
        for caseid in caseids:
            if caseid in imagePaths_per_caseid.keys():
                for path in imagePaths_per_caseid[caseid]:
                    train_imagePaths_per_class[clazz].append(path)

    val_imagePaths_per_class = dict()
    for clazz in class_names:
        val_imagePaths_per_class[clazz] = []

    for clazz, caseids in val_caseids_per_class.items():
        for caseid in caseids:
            if caseid in imagePaths_per_caseid.keys():
                for path in imagePaths_per_caseid[caseid]:
                    val_imagePaths_per_class[clazz].append(path)

    test_imagePaths_per_class = dict()
    for clazz in class_names:
        test_imagePaths_per_class[clazz] = []

    for clazz, caseids in test_caseids_per_class.items():
        for caseid in caseids:
            if caseid in imagePaths_per_caseid.keys():
                for path in imagePaths_per_caseid[caseid]:
                    test_imagePaths_per_class[clazz].append(path)


    return train_imagePaths_per_class, val_imagePaths_per_class, test_imagePaths_per_class, class_names


def get_90_10_siamese_image_label_pairs(train_imagePaths_per_class, val_imagePaths_per_class, test_imagePaths_per_class):
    train_path_pairs = []
    for orig_class, orig_paths in train_imagePaths_per_class.items():
        for path in orig_paths:
            for other_class, other_paths in train_imagePaths_per_class.items():
                random_other_path = np.random.choice(other_paths)
                train_path_pairs.append((path, random_other_path, orig_class, other_class))

    val_path_pairs = []
    for orig_class, orig_paths in val_imagePaths_per_class.items():
        for path in orig_paths:
            for other_class, other_paths in val_imagePaths_per_class.items():
                random_other_path = np.random.choice(other_paths)
                val_path_pairs.append((path, random_other_path, orig_class, other_class))

    test_path_pairs = []
    for orig_class, orig_paths in test_imagePaths_per_class.items():
        for path in orig_paths:
            for other_class, other_paths in test_imagePaths_per_class.items():
                random_other_path = np.random.choice(other_paths)
                test_path_pairs.append((path, random_other_path, orig_class, other_class))
    return train_path_pairs, val_path_pairs, test_path_pairs


def get_50_50_siamese_image_label_pairs_large(train_imagePaths_per_class, val_imagePaths_per_class, test_imagePaths_per_class):
    train_path_pairs = []
    for orig_class, orig_paths in train_imagePaths_per_class.items():
        for path in orig_paths:
            matching_pairs = []
            for other_path in train_imagePaths_per_class[orig_class]:
                matching_pairs.append((path, other_path, orig_class, orig_class))
            shuffle(matching_pairs)
            matching_pairs = matching_pairs[:500] # Max of 500 corresponding pairs per class

            dissonant_pairs = []
            for other_class, other_paths in train_imagePaths_per_class.items():
                if orig_class == other_class:
                    continue
                else:
                    for other_path in other_paths:
                        dissonant_pairs.append((path, other_path, orig_class, other_class))
            shuffle(dissonant_pairs)
            train_path_pairs.extend(dissonant_pairs[:len(matching_pairs)])

    val_path_pairs = []
    for orig_class, orig_paths in val_imagePaths_per_class.items():
        for path in orig_paths:
            matching_pairs = []
            for other_path in val_imagePaths_per_class[orig_class]:
                matching_pairs.append((path, other_path, orig_class, orig_class))

            dissonant_pairs = []
            for other_class, other_paths in val_imagePaths_per_class.items():
                if orig_class == other_class:
                    continue
                else:
                    for other_path in other_paths:
                        dissonant_pairs.append((path, other_path, orig_class, other_class))
            shuffle(dissonant_pairs)
            val_path_pairs.extend(dissonant_pairs[:len(matching_pairs)])

    test_path_pairs = []
    for orig_class, orig_paths in test_imagePaths_per_class.items():
        for path in orig_paths:
            matching_pairs = []
            for other_path in test_imagePaths_per_class[orig_class]:
                matching_pairs.append((path, other_path, orig_class, orig_class))

            dissonant_pairs = []
            for other_class, other_paths in test_imagePaths_per_class.items():
                if orig_class == other_class:
                    continue
                else:
                    for other_path in other_paths:
                        dissonant_pairs.append((path, other_path, orig_class, other_class))
            shuffle(dissonant_pairs)
            test_path_pairs.extend(dissonant_pairs[:len(matching_pairs)])
    return train_path_pairs, val_path_pairs, test_path_pairs


def get_50_50_siamese_image_label_pairs(train_imagePaths_per_class, val_imagePaths_per_class, test_imagePaths_per_class):
    train_path_pairs = []
    for orig_class, orig_paths in train_imagePaths_per_class.items():
        matching_pairs = []

        # Gather at most 900 matching pairs per class
        n = min(len(orig_paths), 30)
        for match_1 in np.random.choice(orig_paths, size=n, replace=False):
            for match_2 in np.random.choice(orig_paths, size=n, replace=False):
                matching_pairs.append((match_1, match_2, orig_class, orig_class))

        # Gather at most 900 dissonant pairs per class
        dissonant_pairs = []
        for path in orig_paths:
            for other_class, other_paths in train_imagePaths_per_class.items():
                if orig_class == other_class:
                    continue
                else:
                    for other_path in other_paths:
                        dissonant_pairs.append((path, other_path, orig_class, other_class))

        shuffle(dissonant_pairs)
        train_path_pairs.extend(matching_pairs)
        train_path_pairs.extend(dissonant_pairs[:len(matching_pairs)])

    val_path_pairs = []
    for orig_class, orig_paths in val_imagePaths_per_class.items():
        matching_pairs = []

        # Gather at most 900 matching pairs per class
        n = min(len(orig_paths), 30)
        for match_1 in np.random.choice(orig_paths, size=n, replace=False):
            for match_2 in np.random.choice(orig_paths, size=n, replace=False):
                matching_pairs.append((match_1, match_2, orig_class, orig_class))

        # Gather at most 900 dissonant pairs per class
        dissonant_pairs = []
        for path in orig_paths:
            for other_class, other_paths in val_imagePaths_per_class.items():
                if orig_class == other_class:
                    continue
                else:
                    for other_path in other_paths:
                        dissonant_pairs.append((path, other_path, orig_class, other_class))

        shuffle(dissonant_pairs)
        val_path_pairs.extend(matching_pairs)
        val_path_pairs.extend(dissonant_pairs[:len(matching_pairs)])

    test_path_pairs = []
    for orig_class, orig_paths in test_imagePaths_per_class.items():
        matching_pairs = []

        # Gather at most 900 matching pairs per class
        n = min(len(orig_paths), 30)
        for match_1 in np.random.choice(orig_paths, size=n, replace=False):
            for match_2 in np.random.choice(orig_paths, size=n, replace=False):
                matching_pairs.append((match_1, match_2, orig_class, orig_class))

        # Gather at most 900 dissonant pairs per class
        dissonant_pairs = []
        for path in orig_paths:
            for other_class, other_paths in test_imagePaths_per_class.items():
                if orig_class == other_class:
                    continue
                else:
                    for other_path in other_paths:
                        dissonant_pairs.append((path, other_path, orig_class, other_class))

        shuffle(dissonant_pairs)
        test_path_pairs.extend(matching_pairs)
        test_path_pairs.extend(dissonant_pairs[:len(matching_pairs)])
    return train_path_pairs, val_path_pairs, test_path_pairs


def load_dataset_dev(image_size, train_part=0.7, val_part=0.15, test_part=0.15, small_dataset=False, train_size=None, val_size=None):
    dataset_name = "mnist"
    (ds_train, ds_test), ds_info = tfds.load(
        dataset_name, split=["train", "test"], with_info=True, as_supervised=True
    )
    class_names = ds_info.features["label"].names

    # Reduce the mnist dataset, too large for testing purposes...
    ds_train = ds_train.take(tf.data.experimental.cardinality(ds_train) // 5)
    ds_test = ds_test.take(tf.data.experimental.cardinality(ds_test) // 5)


    ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, image_size), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, image_size), label))

    ds_train = ds_train.map(lambda image, label: (tf.image.grayscale_to_rgb(image), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.grayscale_to_rgb(image), label))

    if small_dataset:
        ds_train = ds_train.take(train_size)
        ds_test = ds_test.take(val_size)


    val_batches = tf.data.experimental.cardinality(ds_test)
    train_dataset = ds_train
    test_dataset = ds_test.take(val_batches // 5)
    validation_dataset = ds_test.skip(val_batches // 5)
    return train_dataset, test_dataset, validation_dataset, class_names


def load_siamese_dataset_dev(image_size, small_dataset=False, train_size=None, val_size=None):
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    x_size = 300
    y_size = 100

    if small_dataset:
        x_size = train_size
        y_size = val_size

    # Reduce the mnist dataset, too large for testing purposes...
    random_indices = np.random.choice(X_train.shape[0], x_size, replace=False)
    X_train_sample, y_train_sample = X_train[random_indices], y_train[random_indices]

    random_indices = np.random.choice(X_test.shape[0], y_size, replace=False)
    X_test_sample, y_test_sample = X_test[random_indices], y_test[random_indices]

    train_dataset = tf.data.Dataset.from_generator(
        generate_paired_samples_dev,
        args=(X_train_sample, y_train_sample, image_size),
        output_signature=(
            tf.TensorSpec(shape=(2,) + image_size + (3,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).map(lambda x, y: ({'img_A_input': x[0], 'img_B_input': x[1]}, y))

    val_dataset = tf.data.Dataset.from_generator(
        generate_paired_samples_dev,
        args=(X_test_sample, y_test_sample, image_size),
        output_signature=(
            tf.TensorSpec(shape=(2,) + image_size + (3,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).map(lambda x, y: ({'img_A_input': x[0], 'img_B_input': x[1]}, y))

    test_dataset = tf.data.Dataset.from_generator(
        generate_paired_samples_dev,
        args=(X_test_sample, y_test_sample, image_size),
        output_signature=(
            tf.TensorSpec(shape=(2,) + image_size + (3,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    ).map(lambda x, y: ({'img_A_input': x[0], 'img_B_input': x[1]}, y))

    return train_dataset, val_dataset, test_dataset, ["0", "1"]


def generate_image_label_sample(paths_case, image_dims, class_names):
    image_dims = tuple(image_dims)
    class_names = [x.decode() for x in class_names]
    for paths in paths_case:
        label = paths[0].decode().split('/')[1]
        for imagePath in paths:
            imagePath = imagePath.decode()
            if imagePath == "":
                continue
            img = cv2.imread(imagePath)
            if img.shape != image_dims:
                #print(f"Resizing image from {img.shape} to {image_dims}")
                img = cv2.resize(img, (image_dims[0], image_dims[1]))
            yield img, tf.one_hot(class_names.index(label), len(class_names))


def make_pairs_ci(x, class_names, image_dims):
    image_dims = tuple(image_dims)
    class_names = np.array([x.decode() for x in class_names])
    x = [y.decode() for y in x]
    y = list(map(lambda el: el.split('/')[1], x))
    z = []
    for el in y:
        z.append(np.where(class_names == el)[0][0])
    y = np.array(z)
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1] # path
        x1 = cv2.imread(x1) # image
        if x1.shape != image_dims:
            # print(f"Resizing image from {img.shape} to {image_dims}")
            x1 = cv2.resize(x1, (image_dims[0], image_dims[1]))
        label1 = y[idx1]

        idx2 = choice(digit_indices[label1])
        x2 = x[idx2] # path
        x2 = cv2.imread(x2)  # image
        if x2.shape != image_dims:
            x2 = cv2.resize(x2, (image_dims[0], image_dims[1]))

        yield (x1, x2), 0

        # add a non-matching example
        label2 = randint(0, num_classes - 1)
        while label2 == label1:
            label2 = randint(0, num_classes - 1)

        idx2 = choice(digit_indices[label2])
        x2 = x[idx2] # path
        x2 = cv2.imread(x2)  # image
        if x2.shape != image_dims:
            x2 = cv2.resize(x2, (image_dims[0], image_dims[1]))

        yield (x1, x2), 1


def generate_paired_samples_ci(path_pairs, image_dims):
    image_dims = tuple(image_dims)
    for pair in path_pairs:
        img_A_path = pair[0].decode()
        img_B_path = pair[1].decode()

        label_A = img_A_path.split('/')[1]
        label_B = img_B_path.split('/')[1]

        img_A = cv2.imread(img_A_path)
        if img_A.shape != image_dims:
            # print(f"Resizing image from {img.shape} to {image_dims}")
            img_A = cv2.resize(img_A, (image_dims[0], image_dims[1]))

        img_B = cv2.imread(img_B_path)
        if img_B.shape != image_dims:
            # print(f"Resizing image from {img.shape} to {image_dims}")
            img_B = cv2.resize(img_B, (image_dims[0], image_dims[1]))

        new_label = float(label_A != label_B)
        yield [img_A, img_B], new_label


def generate_paired_samples_dev(X, y, img_size):
    tuples = [(x1, y1) for x1, y1 in zip(X, y)]

    for t in itertools.product(tuples, tuples):
        img_A, label_A = t[0]
        img_B, label_B = t[1]

        img_A = tf.image.resize(tf.expand_dims(img_A, -1), img_size)
        img_A = tf.image.grayscale_to_rgb(img_A)

        img_B = tf.image.resize(tf.expand_dims(img_B, -1), img_size)
        img_B = tf.image.grayscale_to_rgb(img_B)

        new_label = float(label_A != label_B)
        yield [img_A, img_B], new_label


def list_files(basePath, dirs_to_search=None, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath, topdown=True):
        dirNames[:] = [d for d in dirNames if d in dirs_to_search]
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains array of strings is not none and the filename does not contain
            # the supplied strings, then ignore the file
            if contains is not None:
                contained = True
                for substr in contains:
                    if filename.find(substr) == -1:
                        contained = False
                        break
                if not contained:
                    continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def split_caseids_per_class_per_group(caseids_per_class, sampling_mode=None, train=0.85, val=0.15, test=0):
    for clazz in caseids_per_class.keys():
        np.random.shuffle(caseids_per_class[clazz])

    train_caseids_per_class = dict()
    val_caseids_per_class = dict()
    test_caseids_per_class = dict()
    min_class_cartridges = np.array([len(caseids_per_class[clazz]) for clazz in caseids_per_class.keys()]).min()
    for clazz in caseids_per_class.keys():
        if sampling_mode is not None and sampling_mode == "UNDERSAMPLE":
            train_caseids_per_class[clazz] = caseids_per_class[clazz][:int(min_class_cartridges * train)]
            val_caseids_per_class[clazz] = caseids_per_class[clazz][int(min_class_cartridges * train):int(min_class_cartridges * (train + val))]
            test_caseids_per_class[clazz] = caseids_per_class[clazz][int(min_class_cartridges * (train + val)): min_class_cartridges]
        else:
            num_cases = len(caseids_per_class[clazz])
            train_caseids_per_class[clazz] = caseids_per_class[clazz][:int(num_cases * train)]
            val_caseids_per_class[clazz] = caseids_per_class[clazz][int(num_cases * train):int(num_cases * (train + val))]
            test_caseids_per_class[clazz] = caseids_per_class[clazz][int(num_cases * (train + val)):]

    # For now we will only consider two subsets...
    return train_caseids_per_class, val_caseids_per_class, val_caseids_per_class
