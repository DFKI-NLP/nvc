##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Miscellaneous utilities for model training, evaluation and storing.

@author lisa.raithel@dfki.de
"""

import json
import os
import sys


def read_config(config_file):
    """Return config JSON as dict.

    Args:
        config_file:    str
                        The path to a JSON file.
    Returns:
        The configuration as a dict.
    """
    with open(config_file, "r") as read_handle:
        config = json.load(read_handle)

    return config


def save_config_to_file(config, model_path):
    """Save the configuration in the model folder.

    Args:
        config: dict
                The configuration of the current model.
        model_path: str
                    The path to the directory of the model.
    """
    model = model_path.split("/")[0]
    config_file_name = "{}/config_{}.json".format(model, model)

    with open(config_file_name, "w") as write_handle:
        json.dump(config, write_handle)


def create_dir(min_rep, min_num_instances, num_hidden, regularizers, timestr):
    """Create a directory with the name of all given parameters."""
    l2_regs_str = "-".join([str(x) for x in regularizers])

    model_path = "minrep{}_mininst{}_hidden{}_reg{}_{}".format(
        min_rep, min_num_instances, num_hidden, l2_regs_str, timestr)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("Created folder '{}'.".format(model_path))
        return model_path

    print("Folder '{}' does already exist.".format(model_path))
    sys.exit(1)


def write_test_words_to_file(indices, rep_matrix, model_path, id2word_dict,
                             concepts):
    """Write all test words and their associated concepts to a tsv file.

    Args:
        indices:    list
                    A list of test instance indices.
        rep_matrix: numpy ndarray
                    A matrix of multi-hot encoded concepts-
        model_path: str
                    The path to the model.
        id2word_dict:   dict
                        A dictionary mapping IDs to instances.
        concepts:   list
                    A list of all concepts (labels).
    """
    test_instances = []
    with open("{}/test_set_{}.csv".format(model_path, model_path),
              "w") as write_handle:

        for idx in indices:
            test_instance = id2word_dict[idx]
            test_instances.append(test_instance)
            # concept row
            reps = rep_matrix[idx]
            concepts_per_instance = [x for x, z in zip(concepts, reps) if z]

            write_handle.write("{}\t{}\n".format(
                test_instance, "\t".join(concepts_per_instance)))

    return test_instances


def sanity_checks(x_train, y_train, x_dev, y_dev, x_test, y_test, embed_matrix,
                  rep_matrix, concepts, idx_train, idx_dev, idx_test,
                  id2word_dict):
    """Check all data for overlaps etc."""
    print("x_train shape: {}, y_train shape: {}".format(
        x_train.shape, y_train.shape))
    print(("x_dev shape: {}, y_dev shape: {}\n"
           "x_test shape: {}, y_test shape: {}").format(
               x_dev.shape, y_dev.shape, x_test.shape, y_test.shape))

    assert x_dev.shape[0] >= 1 and x_test.shape[0] >= 1, (
        "Please change the training data size, there is not enough data "
        "for the dev and test sets.")
    assert len(x_train) + len(x_dev) + len(x_test) == len(embed_matrix)
    assert len(y_train) + len(y_dev) + len(y_test) == len(rep_matrix)
    assert len(y_train[0]) == len(concepts)

    train_instances = set()
    for idx_tr in idx_train:
        train_instances.add(id2word_dict[idx_tr])

    dev_instances = set()
    for idx_dev in idx_dev:
        dev_instances.add(id2word_dict[idx_dev])

    test_instances = set()
    for idx_te in idx_test:
        test_instances.add(id2word_dict[idx_te])

    # make sure the sets are disjoint
    assert not train_instances.intersection(
        dev_instances
    ), "Train and Dev data are overlapping, please check the data split."
    assert not train_instances.intersection(
        test_instances
    ), "Train and Test data are overlapping, please check the data split."
    assert not dev_instances.intersection(
        test_instances
    ), "Dev and Test data are overlapping, please check the data split."
