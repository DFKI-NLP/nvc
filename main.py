##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A test script for experiments with NVC.

@author lisa.raithel@dfki.de
"""
import argparse

from models.nvc_model import NeuralVectorConceptualizer
from models.embeddings import Embedding

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", type=str, help="The path to the tsv/json data.")

    parser.add_argument(
        "embedding_file", type=str, help="The path to the embedding data.")

    parser.add_argument("config", type=str, help="The configuration file.")

    ARGS, _ = parser.parse_known_args()

    embedding = Embedding(embedding_file=ARGS.embedding_file, voc_limit=100000)

    nvc = NeuralVectorConceptualizer(
        config_file=ARGS.config, embedding=embedding, threshold=0.5)

    # load the data (either as TSV or as JSON)
    # filter data according to criteria in config file
    # (min_rep & min_instances)
    nvc.load_data(path_to_data=ARGS.data, filtered=False)
    # load the already filtered data
    # nvc.load_data(path_to_data=ARGS.data, filtered=True)

    # you can also select a subset of concepts to be filtered
    # nvc.load_data(
    #     path_to_data=ARGS.data,
    #     filtered=False,
    #     selected_concepts=["city", "province"])

    # compare predictions and ground truth manually with 'inspect_concept'
    nvc.train()

    # specify the paths to the pre-trained model and the test data
    # model_path = (
    #     "minrep-19_mininst1_hidden2_reg1e-06-1e-06_2019_03_22-16_20/"
    #     "model_minrep-19_mininst1_hidden2_reg1e-06-1e-06_2019_03_22-16_20.h5")

    # data = ("minrep-19_mininst1_hidden2_reg1e-06-1e-06_2019_03_22-16_20/"
    #         "test_set_minrep-19_mininst1_hidden2_reg1e-06-1e-06_"
    #         "2019_03_22-16_20.csv")
    # nvc.load_pretrained_model(trained_model=model_path, x_val_file=data)
    nvc.show_activations(["stone", "sun"], max_highlights=3)
