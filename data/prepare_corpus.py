#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocessing the Microsoft Concept Graph (MCG) corpus.

A script for preparing the basic corpus such that is easy to modify
by just adding new preprocessing functions to a map function.

@author lisa.raithel@dfki.de
"""

import json
import numpy as np
import pandas as pd
import sys
import time

from collections import defaultdict


class DataLoader(object):
    """Class for loading and preparing data for a model."""

    def __init__(self, embedding, use_logs=True, only_unigrams=False):
        """Initialize the loader.

        Args:
            embedding:  class
                        A class providing embedding utilities.
            use_logs    bool
                        Use the logarithm of the MCG scores.
            only_unigrams   bool
                            Use only unigrams or phrases as well.
        """
        self.filtered_data = defaultdict(lambda: defaultdict(float))
        self.use_logs = use_logs
        self.only_unigrams = only_unigrams
        self.embedding = embedding
        self.raw_data_dict = defaultdict(lambda: defaultdict(float))
        self.timestr = time.strftime("%Y_%m_%d-%H_%M")

    def load_raw_data(self,
                      path_to_data="",
                      header=[],
                      chunk_size=1000**2,
                      save_to_json=True):
        """Prepare the raw data in dict format.

        Args:
            path_to_data:   str
                            The path to the original TSV data or a JSON
                            dump of it.
            header: list
                    The header of the TSV file.
            chunk_size: int
                        The chunk size for reading in the TSV.
            save_to_json:   bool
                            Save the resulting dict to a JSON file.
        Returns:
            A dict of all concepts associated with their instances:
            raw_data = {concept_1: {instance_1: rep_1, instance_2: rep_2},
                        concept_2: {instance_2: rep_2, instance_3, rep_3},...}
        """
        if path_to_data.endswith(("json", "JSON")):
            try:
                # if there is already a JSON file provided, use it instead
                # of reading in the original data (slow)
                with open(path_to_data, "r") as json_file:
                    print("Loading data file '{}'.\n".format(path_to_data))
                    self.raw_data_dict = json.load(json_file)

            except FileNotFoundError as e:
                print("File '{}' not found, please provide the correct "
                      "path.\n{}".format(path_to_data, e))
                sys.exit(1)
            # if there is only a TSV file provided, create a dict for all
            # concepts and instances and calculate the REP scores
            except json.decoder.JSONDecodeError as e:
                print("JSON file '{}' invalid: {}".format(path_to_data, e))
                sys.exit(1)

        else:
            try:
                if self.use_logs:
                    print("Using log of REP values.\n")
                # read in data chunk-wise
                for c, chunk in enumerate(
                        pd.read_csv(
                            path_to_data,
                            chunksize=chunk_size,
                            delimiter="\t",
                            names=header)):
                    df = pd.DataFrame(chunk)

                    for i, row in df.iterrows():

                        # check if a word or phrase is in the embedding
                        # vocabulary(checks also modified versions of the word,
                        # like upper/lower case, with underscores etc.)
                        modified_concept = self.embedding.in_embedding(
                            str(row["concept"]), self.only_unigrams)

                        if not modified_concept:
                            continue
                        else:
                            concept = modified_concept

                        proba_c_given_e = float(row["p(c|e)"])
                        proba_e_given_c = float(row["p(e|c)"])
                        instance = str(row["instance"])

                        # check if a version of the current instance is in the
                        # embedding vocab
                        # (this could also be a shorter phrase than the
                        # original instance)
                        modified_instance = self.embedding.in_embedding(
                            instance, self.only_unigrams)
                        # if non of the modified instances is in the embedding
                        # vocabulary, go to the next instance
                        if not modified_instance:
                            continue
                        else:
                            instance = modified_instance

                        if self.use_logs:
                            # use natural logarithm to calculate the REP values
                            self.raw_data_dict[concept][instance] = np.log(
                                proba_c_given_e) + np.log(proba_e_given_c)

                        else:
                            new_rep = proba_c_given_e * proba_e_given_c
                            self.raw_data_dict[concept][instance] = new_rep

                        if i % 10000 == 0:
                            print(
                                "{} lines done".format(i),
                                end="\r",  # noqa: E901
                                flush=True)  # noqa: E901

                # save the data to a JSON file to increase processing speed
                # for the next run
                if save_to_json:
                    if self.only_unigrams:
                        file_name = "raw_data_dict_only_unigrams.json"
                    else:
                        file_name = "raw_data_dict.json"

                    self.save_to_json(
                        data=self.raw_data_dict, file_name=file_name)

            except FileNotFoundError as e:
                print("File '{}' not found, please provide the correct "
                      "path.\n{}".format(path_to_data, e))
                sys.exit(1)

        print("#concepts in raw data: {}\n".format(
            len(self.raw_data_dict.keys())))

        return self.raw_data_dict

    def load_filtered_data_matrix(self,
                                  data=None,
                                  min_num_instances=2,
                                  min_rep=-6.0,
                                  save_to_json=True,
                                  selected_concepts=[]):
        """Load the filtered data.

        Args:
            data:   dict or str
                    The data as dict or JSON file -- could be either the raw
                    data or the already filtered data.
            min_num_instances:  int
                                The minimum number of instances a concept
                                need to have.
            min_rep:    float
                        The minimum REP value an instance needs to have.
            save_to_json:   bool
                            Save the filtered data to a JSON file.
            selected_concepts:  list
                                A list of selected concepts that should be
                                considered (and no other concepts).
        """
        filtered_data = defaultdict(lambda: defaultdict(float))
        # try if the given file is already filtered (because then it's in JSON
        # format)
        try:
            with open(data, "r") as json_file:
                filtered_data = json.load(json_file)
            print("Loading existing filtered data file '{}'.".format(data))

        # otherwise, check if the filtered data already exists
        except json.decoder.JSONDecodeError as e:
            print("JSON DecodingError: {}.\nPlease provide a valid JSON file.".
                  format(e))
            sys.exit()

        except TypeError:
            print("Filtering data ...")
            if selected_concepts:
                filtered_data = self.get_selected_concepts_and_instances(
                    data, selected_concepts, min_rep, min_num_instances,
                    save_to_json)
            else:
                filtered_data = self.get_filtered_data(
                    data, min_rep, min_num_instances, save_to_json)

        except FileNotFoundError as e:
            print("No existing filtered data found: {}.\n".format(e))
            sys.exit(1)

        # - create dataframe to get instances for rows and concepts
        #   for columns
        # - the dataframe-from-dict format ensures that every instance is
        #   occurring only *once* in the data matrix, ensuring again that
        #   the train/dev/test splits are disjoint
        df_filtered = pd.DataFrame.from_dict(filtered_data)
        df_filtered = df_filtered.fillna(0)
        # make sure the columns are always filtered lexicographically to
        # ensure the order of labels is always the same for the same data
        df_filtered.reindex(sorted(df_filtered.columns), axis=1)
        print("shape filtered data (instances x concepts): {}".format(
            df_filtered.shape))

        # needs to be done only once per filtered data
        self.get_all_words_not_in_mcg(df_filtered, min_num_instances, min_rep)

        return df_filtered

    def get_selected_concepts_and_instances(self, raw_data_dict,
                                            selected_concepts, min_rep,
                                            min_inst, save_to_json):
        """Get only a predefined set of concepts and their instances.

        Args:
            raw_data_dict:  dict
                            The raw data in dict format.
            selected_concepts:  list
                                A list of concepts.
            min_rep:    float
                        The minimum REP value a instance needs to have.
            min_inst:   int
                        The minimum number of instances a concepts needs
                        to have.
            save_to_json:   bool
                            Save the filtered data to a JSON file.
        Returns:
            The filtered data.
        """
        concept_counter = 0
        filtered_data = defaultdict(lambda: defaultdict(float))

        for concept in selected_concepts:
            concept_counter += 1
            print(
                "{} concepts done".format(concept_counter),
                end="\r",
                flush=True)

            instance_rep_dict = raw_data_dict[concept]
            inst_values = instance_rep_dict.values()
            num_inst_geq_v = sum([x >= min_rep for x in inst_values])

            new_instances_dict = {}
            # check if there are enough instances with a rep > v
            if num_inst_geq_v >= min_inst:
                for instance, rep in instance_rep_dict.items():
                    # take only instances with a REP > v
                    if rep >= min_rep:
                        new_instances_dict[instance] = rep

                filtered_data[concept] = new_instances_dict
            else:
                continue

        if save_to_json:
            file_name = "selected_concepts_i{}_v{}_{}_{}.json".format(
                min_inst, min_rep, selected_concepts[0], selected_concepts[1])

            self.save_to_json(data=filtered_data, file_name=file_name)

        return filtered_data

    def get_filtered_data(self, raw_data_dict, min_rep, min_inst,
                          save_to_json):
        """Filter the data.

        Args:
            raw_data_dict:  dict
                            The raw data in dict format.
            min_rep:    float
                        The minimum REP value a instance needs to have.
            min_inst:   int
                        The minimum number of instances a concepts needs
                        to have.
            save_to_json:   bool
                            Save the filtered data to a JSON file.
        Returns:
            The filtered data.
        """
        concept_counter = 0
        filtered_data = defaultdict(lambda: defaultdict(float))

        for concept, instance_rep_dict in raw_data_dict.items():
            concept_counter += 1
            if concept_counter % 10000 == 0:
                print(
                    "{} concepts done".format(concept_counter),
                    end="\r",
                    flush=True)

            # concepts of filtered data with more than i instances
            # having a REP >= v:
            inst_values = instance_rep_dict.values()
            num_inst_geq_v = sum([x >= min_rep for x in inst_values])

            new_instances_dict = {}
            # check if there are enough instances with a rep > v
            if num_inst_geq_v >= min_inst:
                for instance, rep in instance_rep_dict.items():
                    # take only instances with a REP > v
                    if rep >= min_rep:
                        new_instances_dict[instance] = rep

                filtered_data[concept] = new_instances_dict
            else:
                continue

        assert len(filtered_data.keys()) != 0, ("\nNo concepts collected when "
                                                "filtering data.\nPlease "
                                                "change the filter parameters."
                                                "\n")

        if save_to_json:
            file_name = "filtered_data_i{}_v{}.json".format(min_inst, min_rep)
            self.save_to_json(data=filtered_data, file_name=file_name)

        return filtered_data

    def get_all_words_not_in_mcg(self, filtered_data_matrix, min_inst,
                                 min_rep):
        """Get all words that are not in the concept graph.

        Retrieve all words that are in the vocabulary of word2vec,
        but not in the MCG, and save them to a text file, one word per line.
        These can be used to find instances for plotting etc., which were not
        used in training, validating or testing the model.

        Args:
            filtered_data_matrix:   numpy ndarray
                                    The already filtered data.
            min_inst:   int
                        The minimum number of instances a concepts needs
                        to have.
            min_rep:    float
                        The minimum REP value a instance needs to have.
        """
        # get all instances from the data matrix
        instances = list(filtered_data_matrix.index)
        # 'remove' the instances used in training/validation/test
        remaining_words = set(self.embedding.vocab) - set(instances)

        file_name = "words_not_in_mcg_i{}_v{}.txt".format(
            min_inst, min_rep, self.timestr)

        print("Writing {} instances not in embedding vocabulary to file '{}'.".
              format(
                  len(remaining_words),
                  file_name,
              ))

        with open(file_name, "w") as write_handle:
            for word in remaining_words:
                write_handle.write(word)
                write_handle.write("\n")
        print("Done.\n")

    def load_from_json(self, json_file):
        """Load data from json."""
        with open(json_file, "r") as handle:
            data = json.load(handle)
        return data

    def save_to_json(self, data, file_name):
        """Save data to json file."""
        with open(file_name, "w") as handle:
            json.dump(data, handle)

        print("Data stored in {}.\n".format(file_name))


if __name__ == '__main__':
    json_file = sys.argv[1]
    # "graph_as_dict.json"
    with open(json_file, "r") as handle:
        raw_data = json.load(handle)
    # i: minimum number of instances per concept
    i = 3
    # v: lowest rep value for instances
    v = -6.0
    # take only instances that are unigrams
    only_unigrams = True
    dl = DataLoader(embedding=None, use_logs=True, only_unigrams=False)
    header = [
        "concept", "instance", "count", "p(c|e)", "p(e|c)", "rep", "empty"
    ]

    filtered_data = dl.load_data_matrix(
        raw_data_dict=raw_data,
        min_inst=i,
        min_rep=v,
        only_unigrams=only_unigrams)
    print("\n#concepts of filtered data with more than {} instances "
          "having a log REP >= {}: {}".format(i, v, len(filtered_data.keys())))
