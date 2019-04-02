##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A class for easier access to embeddings.

- allows to calculate distances and cosine similarities of words and
  vectors.
- allows to check for vocabulary words / modifies words to fit to
  embedding vocabulary. E.g. the word 'New York' is modified to
  'New_York'.

@author lisa.raithel@dfki.de
"""

import gzip
import numpy as np
import pandas as pd
import sys

from gensim.models import KeyedVectors


class Embedding(object):
    """Class for making several embeddings accessible."""

    def __init__(self, embedding_file="", voc_limit=None):
        """Initialize the embedding module.

        Args:
            embedding_file: str
                            Path to the word vectors.
            voc_limit:  int
                        A restriction on the embedding vocabulary.
        """
        if embedding_file:
            self.embedding_file = embedding_file
        else:
            print("No file for embedding data given.")
            sys.exit(1)

        self.model = None

        self.voc_limit = voc_limit
        self.word_vectors = self.load_embedding_model()
        self.vocab = self.word_vectors.vocab

    def load_embedding_model(self):
        """Prepare the embedding model from the given file."""
        try:
            with gzip.open(self.embedding_file, "r") as file_handle:
                self.model = KeyedVectors.load_word2vec_format(
                    file_handle, binary=True, limit=self.voc_limit)

        except FileNotFoundError as e:
            raise e

        except OSError:
            self.model = KeyedVectors.load_word2vec_format(
                self.embedding_file, binary=True, limit=self.voc_limit)

        print("\nEmbedding loaded.")

        self.embedding_dim = self.model.vector_size
        return self.model.wv

    def get_word_vectors(self):
        """Getter method to have access to the word vectors."""
        return self.word_vectors

    def get_minimal_distant_concept(self, word_vector, all_concepts):
        """Get the nearest concept.

        Args:
            word_vector:    numpy ndarray
                            The word vector.
            all_concepts:   list
                            A list of all concepts.
        Returns:
            The minimal distance and the associated concept.
        """
        new_min_sim = self.word_vectors.distances(word_vector, all_concepts)
        idx = np.argmin(new_min_sim)

        return new_min_sim[idx], all_concepts[idx]

    def get_sim_btw_vector_and_concepts(self, word_vector, word_vec_name,
                                        all_concepts):
        """Get the similarity for each vector to all given concepts.

        Args:
            word_vector:    numpy ndarray
                            The word vector.
            word_vec_name:  str
                            The "word" or dummy.
            all_concepts:   list
                            A list of all concepts.
        Returns:
            A pandas dataframe comprising the instance and its distance
            to all concepts.
        """
        # get the distances
        distances = self.word_vectors.distances(word_vector, all_concepts)
        # calculate the similarity score
        similarities = 1 - distances
        # create the dataframe with the concepts as columns
        df = pd.DataFrame(
            dict(zip(all_concepts, similarities)),
            columns=list(all_concepts),
            index=[word_vec_name])

        return df

    def get_similarity_to_concepts(self, instance, concepts):
        """Get the nearest neighbors of an instance.

        Args:
            instance:   str
                        A word.
            concepts:   list
                        All available concepts
        Returns:
            An array of similarity scores for all concepts.
        """
        sim_scores = []

        for concept in concepts:
            sim_scores.append(self.word_vectors.similarity(instance, concept))
        sim_scores = np.asarray(sim_scores)

        return sim_scores

    def get_embedding_for_word(self, word):
        """Get the embedding of one word.

        Args:
            word:   str
                    A word.

        Returns:
            The word vector for the given instance.
        """
        if word in self.word_vectors:
            return self.word_vectors[word]
        return None

    def in_embedding(self, word, only_unigrams):
        """Check if word or phrase or its modified version is in embedding.

        Args:
            word:   str
                    A word.
            only_unigrams:  bool
                            Consider only unigrams in the data.
        Returns:
            An instance, if the instance or a modified version are in the
            embedding vocabulary.
            False, if not.
        """
        word_list = word.split(" ")

        if only_unigrams and len(word_list) > 1:
            return False

        # try two different variant of modification
        # e.g. "new york" --> "New_York"
        variant_1 = word.replace(" ", "_")
        # e.g. "new york" --> "New_York"
        variant_2 = "_".join(
            [x[0].upper() + x[1:] for x in word_list if x != ""])

        if variant_1 in self.word_vectors:
            return variant_1

        elif variant_2 in self.word_vectors:
            return variant_2

        elif len(word_list) > 1:
            longest_phrase = ""
            # create all possible phrases from the given word list in the
            # given order
            # the longest phrases in w2v will be returned
            for i in range(1, len(word_list)):
                new_phrase = "_".join(word_list[0:i + 1])
                if new_phrase in self.word_vectors:
                    longest_phrase = new_phrase

            if longest_phrase != "":
                return longest_phrase

        return False
