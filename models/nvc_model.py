#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A simple classifier for neural word vector conceptualization (NVC).

@author lisa.raithel@dfki.de
"""

import argparse
import numpy as np
import pandas as pd
import time
import sys

from keras.layers import Dense, Flatten, Input
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             precision_recall_fscore_support)

# own modules
from data.prepare_corpus import DataLoader
from utils.callbacks_and_metrics import MetricsCallback
from utils import utils
from utils import visualizations as viz


class NeuralVectorConceptualizer(object):
    """A model for neural vector conceptualization of word vectors."""

    def __init__(self, config_file, embedding, threshold=0.5):
        """Initialize the NVC model class.

        Args:
            config_file:    str
                            The path to a configuration in JSON format.
            embedding:      class
                            A class providing embedding utilities.
            threshold:      float
                            A threshold for the classification performance.
        """
        self.embedding = embedding
        self.word_vectors = embedding.get_word_vectors()
        self.embedding_dim = embedding.embedding_dim

        # get the configuration
        self.config = utils.read_config(config_file)
        self.timestr = time.strftime("%Y_%m_%d-%H_%M")
        self.threshold = threshold
        self.pretrained = False

        self.model = None
        self.model_path = ""
        self.model_name = ""

        self.word2id = {}
        self.id2embedding = {}
        self.id2word = {}

        self.results_per_class_table = None
        self.average_results_table = None

    def create_model_name_and_dir(self):
        """Create a directory for new trained model and associated files."""
        self.model_path = utils.create_dir(
            self.config["min_rep"], self.config["min_num_inst"],
            self.config["num_hidden_layers"], self.config["regularizers"],
            self.timestr)

        self.model_name = "{}/model_{}.h5".format(self.model_path,
                                                  self.model_path)

    def load_data(self, path_to_data, filtered, selected_concepts=[]):
        """Load raw data via the DataLoader.

        Args:
            path_to_data:   string
                                The path to either a json or a tab-separated
                                file of the raw data with a header as given
                                in the configuration file.
        """
        # prepare the data loader
        self.loader = DataLoader(
            embedding=self.embedding,
            use_logs=True,
            only_unigrams=self.config["only_unigrams"],
        )
        if filtered:
            self.filter_data(data=path_to_data)
        else:
            # load the raw data
            self.raw_data = self.loader.load_raw_data(
                path_to_data=path_to_data,
                header=self.config["mcg_file_header"],
                save_to_json=True)
            # and filter it according to the given criteria
            self.filter_data(
                data=self.raw_data, selected_concepts=selected_concepts)

    def filter_data(self, data={}, selected_concepts=[]):
        """Filter the dataset according to certain criteria.

        The minimum number of instances and the minimal REP values these
        instances have to have are extracted from the configuration file.
        Moreover, you can only train on a subset of concepts.
        Per default, all concepts that fulfill the configuration criteria are
        used for creating the dataset.

        Args:
            selected_concepts:  list, optional
                                Train only on a subset of concepts and their
                                respective instances.
        """
        # the minimum number of instances a concept needs to have to be
        # considered
        self.min_num_instances = self.config["min_num_inst"]
        # the minimal REP value the instances need to have to be 'accepted'
        self.min_rep = self.config["min_rep"]
        # get filtered data as pandas data frame:
        # rows: instances
        # columns: concepts
        self.filtered_data = self.loader.load_filtered_data_matrix(
            data=data,
            min_num_instances=self.min_num_instances,
            min_rep=self.min_rep,
            save_to_json=True,
            selected_concepts=selected_concepts)

        # collect all instances and labels in lists and create lookup tables
        self.instances, self.inst2id, self.id2inst = self.prepare_instances()
        self.labels, self.label2id, self.id2label = self.prepare_labels()

    def prepare_instances(self):
        """Create instance list, instance2id and id2instance dictionaries.

        Returns:
            instances:  list
                        A list of all instances.
            instance2id:    dict
                            A dictionary from instances to IDs.
            id2instance:    dict
                            A dictionary from IDs to instances.
        """
        instances = list(self.filtered_data.index)
        instance2id = {}
        id2instance = {}

        for i, instance in enumerate(instances):
            instance2id[instance] = i
            id2instance[i] = instance

        return instances, instance2id, id2instance

    def prepare_labels(self):
        """Create concept list, concept2id and vice versa.

        Returns:
            labels: list
                    A list of all concepts (labels).
            label2id:   dict
                        A dictionary from concepts to IDs.
            id2label:   dict
                        A dictionary from IDs to concepts.
        """
        labels = list(self.filtered_data)
        label2id = {}
        id2label = {}

        for i, label in enumerate(labels):
            label2id[label] = i
            id2label[i] = label

        return labels, label2id, id2label

    def split_data(self,
                   embedding_matrix,
                   rep_matrix,
                   train_size,
                   shuffle=True):
        """Split in training, development and test data.

        Args:
            embedding_matrix:   numpy ndarray
                                A (num_instances x embedding_size) matrix
                                containing all instances as word vectors.
            rep_matrix: numpy ndarray
                        A (num_instances x num_concepts) matrix containing
                        all multi-hot encoded label vectors for all instances.
            train_size: float
                        The percentage of training examples.
            shuffle:    bool
                        Shuffle the dataset or not.

        Returns:
            x_train:    numpy ndarray
                        A matrix of training instances.
            y_train:    numpy ndarray
                        A matrix of training labels.
            x_dev:  numpy ndarray
                    A matrix of validation instances.
            y_dev:  numpy ndarray
                    A matrix of validation labels.
            idx_dev:    list
                        A list of indices for the dev data.
            x_test: numpy ndarray
                    A matrix of test instance.
            y_test: numpy ndarray
                    A matrix of test labels.
            test_inst_list: list
                            A list of test words (not word vectors).
        """
        # Get the split-up training instances/labels and the remaining data,
        # as well as their indices.
        # The remaining data will be split into validation and test set.
        (x_train, x_remaining, y_train, y_remaining, idx_train,
         idx_remaining) = train_test_split(
             embedding_matrix,
             rep_matrix,
             np.arange(0, len(embedding_matrix)),
             train_size=train_size,
             shuffle=shuffle)

        # Split the remaining data into validation and test set (50/50).
        # Also, get the respective indices.
        x_dev, x_test, y_dev, y_test, idx_dev, idx_test = train_test_split(
            x_remaining,
            y_remaining,
            idx_remaining,
            train_size=0.5,
            shuffle=shuffle)

        # write the test instances and their respective concepts to a file
        test_inst_list = utils.write_test_words_to_file(
            indices=idx_test,
            rep_matrix=rep_matrix,
            model_path=self.model_path,
            id2word_dict=self.id2word,
            concepts=self.labels)

        # check again for data overlaps / size of datasets etc.
        utils.sanity_checks(
            x_train=x_train,
            y_train=y_train,
            x_dev=x_dev,
            y_dev=y_dev,
            x_test=x_test,
            y_test=y_test,
            embed_matrix=embedding_matrix,
            rep_matrix=rep_matrix,
            concepts=self.labels,
            idx_train=idx_train,
            idx_dev=idx_dev,
            idx_test=idx_test,
            id2word_dict=self.id2word)

        print("Split data.\n\n")
        return (x_train, y_train, x_dev, y_dev, idx_dev, x_test, y_test,
                test_inst_list)

    def create_embedding_and_label_matrix(self, embedding_dim):
        """Create the dataset for further processing.

        Args:
            embedding_dim:  int
                            Embedding dimension of the embedding model used.
        Returns:
            embedding_matrix:   numpy ndarray
                                A matrix of size
                                (num_instances x embedding_dim).
            concept_reps_matrix:    numpy ndarray
                                    A matrix of size
                                    (num_instances x num_labels).
        """
        # initialize embedding and label matrices
        embedding_matrix = np.zeros((len(self.instances), embedding_dim))
        concept_reps_matrix = np.zeros((len(self.instances), len(self.labels)))
        num_positive_labels = 0

        for i, word in enumerate(self.instances):

            # get the word vector for every instance
            embedding_vec = self.embedding.get_embedding_for_word(word)

            if embedding_vec is not None:
                # add the word vector to the embedding matrix and save
                # its ID for later lookups
                embedding_matrix[i] = embedding_vec
                self.id2embedding[i] = embedding_vec
                self.word2id[word] = i
                self.id2word[i] = word

            # get the rep values for the current instance
            reps = np.array(self.filtered_data.loc[word])

            # convert the rep vector in a multi-hot encoded vector
            reps[reps != 0.0] = 1
            # count the number of activated concepts per instance
            num_positive_labels += sum(reps)

            concept_reps_matrix[i] = reps

        print("Avrg #positive concepts per instance: {}".format(
            num_positive_labels / len(self.instances)))

        return embedding_matrix, concept_reps_matrix

    def get_model(self, num_output_units):
        """Build and compile the model.

        Args:
            num_output_units:   int
                                Number of output units for the classifier.
                                Corresponds to the number of classes.
        Returns:
            The compiled model.
        """
        # retrieve the list of l2 regularization factors and
        # number of hidden layers from the config file
        l2_regs = self.config["regularizers"]
        num_hidden_layers = self.config["num_hidden_layers"]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # instantiate the keras input tensor with batches of
        # 300-dimensional vectors
        input_data = Input(shape=(self.embedding_dim, 1), dtype="float32")
        # flatten the input data (redundant, because we first expanded the
        # dimension of the input data and then flatten it again, this is
        # a TODO)
        x = Flatten()(input_data)

        if num_hidden_layers != 0:

            # add fully connected layers without regularizers
            if len(l2_regs) == 0:
                for i in range(num_hidden_layers):
                    x = Dense(units=num_output_units, activation="relu")(x)
            # add fully connected layers with l2 regularizers in each layer
            else:
                for i, reg in zip(range(num_hidden_layers), l2_regs):
                    x = Dense(
                        units=num_output_units,
                        activation="relu",
                        kernel_regularizer=regularizers.l2(reg))(x)
        # add an output layer with a sigmoid activation for each output
        # neuron
        output = Dense(units=num_output_units, activation="sigmoid")(x)

        model = Model(inputs=[input_data], outputs=[output])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # compile the model
        model.compile(
            # binary cross entropy to account for independent concepts
            loss="binary_crossentropy",
            optimizer="adam",
            # calculate mean accuracy rate
            # across all predictions
            metrics=["categorical_accuracy"])

        return model

    def inspect_predictions_for_class(self, inspect_concept, x, x_idx, y):
        """Inspect the predictions manually.

        Prints tab-separated strings of instance-predicted-true to the screen.

        Args:
            inspect_concept:  str
                            The class to be inspected.
            x:  numpy ndarray
                The input data for the model.
            x_idx:  list
                    The indices of the input data.
            y:  numpy ndarray
                The label matrix for the input data.
        """
        try:
            # get the ID for the inspected class
            idx = self.labels.index(inspect_concept)
        except ValueError as e:
            print(
                ("\nThe word {}.\nPlease choose one of the available labels.\n"
                 "Exit.").format(e))
            sys.exit(1)

        # predict the labels for the given data
        y_pred = self.model.predict(x)
        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0

        # collect all predictions and the respective ground truths
        results = {}
        for i, word_idx in enumerate(x_idx):
            word = self.id2word[word_idx]
            results[word] = {"predicted": y_pred[i][idx], "true": y[i][idx]}

        print("\nConcept to be inspected: {}\n".format(inspect_concept))
        df = pd.DataFrame.from_dict(results, orient="index")
        print(df)
        print("\n")

    def check_vectors(self, vec=[], word=""):
        """Check if the vectors are valid.

        Check if the given word vectors/instances are part of the dataset
        and if they are part of the word2vec vocabulary.

        Args:
            vec_1:  numpy ndarray
                    A word vector.
            word_1: str
                    The actual word represented by the word vector.
        Returns:
            True if all word is not in train/dev/test data
            but in word2vec vocabulary, otherwise False.
        """
        # check if the word/vector was used before
        if word in self.word2id:
            print("Word '{}' in train/dev/test data, please choose "
                  "another word.".format(word))
            return False

        # check if the vector is part of the word2vec vocabulary
        if vec is None:
            print("Word '{}' cannot be found in word2vec, please choose "
                  "another word.".format(word))
            return False

        return True

    def sample_vectors(self, word_1, word_2, n_steps=3, model_name=""):
        """Sample the vectors between two vectors with n steps.

        Args:
            word_1: str
                    The first word.
            word_2: str
                    The last word.
            n_steps:    int
                        The number of vectors to be sampled between word_1
                        and word_2.
            model_name: str
                        The path to the model directory.
        """
        # get the word vectors for the given instances
        vec_1 = self.embedding.get_embedding_for_word(word_1)
        vec_2 = self.embedding.get_embedding_for_word(word_2)

        # check if instance is in training or test data
        for word, vec in [(word_1, vec_1), (word_2, vec_2)]:
            if word in self.word2id:
                print(
                    "The word '{}' is part of the train/dev/test data.".format(
                        word))

        # create dataframe for distances of target word to all concepts
        cosine_dataframe = self.embedding.get_sim_btw_vector_and_concepts(
            vec_1, word_1, self.labels)

        # create the embedding matrix as input for the model
        embedding_matrix = np.zeros((n_steps + 2, 300))
        # create an embedding matrix to feed into the model
        embedding_matrix[0] = vec_1
        # collect all known words + placeholders for the 'step-vectors'
        ticks_text = [word_1]

        for i in range(1, n_steps + 1):
            # calculate the next vector
            new_vec = (i / (n_steps + 1)) * (vec_2 - vec_1) + vec_1
            # add it to the embedding matrix
            embedding_matrix[i] = new_vec
            vec_name = "{}".format(i)
            # add a dummy name for the ticks in the plot
            ticks_text.append(vec_name)

            # get the similarities for the next vector
            next_df = self.embedding.get_sim_btw_vector_and_concepts(
                new_vec, vec_name, self.labels)
            # concatenate the new dataframe with the old
            cosine_dataframe = pd.concat([cosine_dataframe, next_df])

        # add the last vector to the embedding matrix and the last word to the
        # ticks list
        embedding_matrix[n_steps + 1] = vec_2
        ticks_text.append(word_2)

        next_df = self.embedding.get_sim_btw_vector_and_concepts(
            vec_2, word_2, self.labels)
        cosine_dataframe = pd.concat([cosine_dataframe, next_df])

        # TODO: this is redundant
        x = np.expand_dims(embedding_matrix, axis=2)
        predictions = self.model.predict(x)
        # create a dataframe from all predictions
        predictions_dataframe = pd.DataFrame(
            predictions, columns=self.labels, index=ticks_text)

        # visualize the samples for NVC and cosine similarity
        viz.show_samples(
            activations=predictions_dataframe,
            similarities=cosine_dataframe,
            concepts=self.labels,
            ticks_text=ticks_text,
            first_word=word_1,
            last_word=word_2,
            steps_between=n_steps,
            model_path=self.model_path,
            pretrained=self.pretrained)

    def show_activations(self, instances, max_highlights=3):
        """Plot the activations of an instance.

        Args:
            instances:  list
                        A list of instances whose activations should be
                        plotted.
            max_highlights: int
                            The number of labeled instances.
        """
        embedding_matrix = np.zeros((len(instances), 300))

        instances_in_w2v = []
        # check if instance is in training or test data
        for i, word in enumerate(instances):
            vec = self.embedding.get_embedding_for_word(word)

            if self.check_vectors(word=word, vec=vec):
                embedding_matrix[i] = vec
                instances_in_w2v.append(word)

        # TODO: this is redundant
        x = np.expand_dims(embedding_matrix, axis=2)

        # predict the concept activations
        predictions = self.model.predict(x)

        # for comparison purposes, also plot the most cosine-similar concepts
        similarities = []

        for instance in instances_in_w2v:
            sim = self.embedding.get_similarity_to_concepts(
                instance, self.labels)
            similarities.append(sim)

        viz.display_concept_activations(
            instances=instances_in_w2v,
            classes=self.labels,
            predictions=predictions,
            similarities=similarities,
            max_highlights=max_highlights,
            model_path=self.model_path,
            pretrained=self.pretrained)

    def train(self, inspect_concept=False):
        """Train the NVC model.

        Args:
            inspect_concept:    bool or str
                                Inspect a given concept with all instances.
        """
        self.create_model_name_and_dir()
        # create two matrices: one for the instances and one for the concepts
        embed_matrix, label_matrix = self.create_embedding_and_label_matrix(
            embedding_dim=self.embedding_dim)

        print("shape embedding matrix: {}".format(embed_matrix.shape))
        print("shape label matrix: {}\n".format(label_matrix.shape))

        # create a callback for metrics like F1 score
        metrics_callback = MetricsCallback(
            labels=self.labels, threshold=self.config["classif_threshold"])
        callbacks = [metrics_callback]

        # configure early stopping if it is set in the config file
        if self.config["early_stopping"]:
            early_stop = EarlyStopping(
                monitor='val_loss',
                min_delta=0.0002,
                patience=3,
                verbose=1,
                mode='auto',
                baseline=None,
                restore_best_weights=False)

            callbacks.append(early_stop)

        if self.config["cross_val"]:
            # Train the model via cross validation and save the best one
            self.train_with_cross_validation(
                embed_matrix, label_matrix, callbacks, save_best_model=True)

        else:
            # otherwise, split the data for conventional training
            (x_train, y_train, x_dev, y_dev, idx_dev, x_val, y_val,
             test_instances_list) = self.split_data(
                 embed_matrix,
                 label_matrix,
                 train_size=self.config["train_set_size"])

            # TODO: this is redundant
            x_train = np.expand_dims(x_train, axis=2)
            x_dev = np.expand_dims(x_dev, axis=2)

            # if the class weights option is set in the config file,
            # calculate the weights from the given training labels
            class_weights = None
            if self.config["class_weights"]:
                class_weights = class_weight.compute_class_weight(
                    "balanced", np.unique(y_train), y_train.flatten())

            # build and compile the model
            print("Training new model: '{}'.\n".format(self.model_name))
            model = self.get_model(num_output_units=label_matrix.shape[1])
            model.summary()

            # train the model with the configuration as given in the config
            # file
            model.fit(
                x_train,
                y_train,
                validation_data=(x_dev, y_dev),
                epochs=self.config["epochs"],
                callbacks=callbacks,
                batch_size=self.config["batch_size"],
                class_weight=class_weights,
                verbose=True)

            self.model = model

            # if a specific concept is given for manual inspection,
            # print the predicted and true labels to the screen
            if inspect_concept:
                self.inspect_predictions_for_class(inspect_concept, x_dev,
                                                   idx_dev, y_dev)

            # save the model and the configuration
            self.model.save(self.model_name)
            utils.save_config_to_file(self.config, self.model_path)
            # evaluate the model and save the evaluation data to a file
            self.predict_and_evaluate(
                x_val=x_val, y_val=y_val, test_instances=test_instances_list)

    def load_pretrained_model(self, trained_model="", x_val_file=""):
        """Load a pre-trained model.

        Args:
            trained_model:  str
                            The path to the h5 model.
            x_val_file:     str
                            The path to the validation data.

        """
        if not trained_model:
            print("Please specify a model.\nExit.")
            sys.exit(1)

        self.pretrained = True

        # Create the original embedding and label matrix (with *all* data)
        # These matrices always have the same ordering when given the same
        # filtered_data json file.
        embed_matrix, label_matrix = self.create_embedding_and_label_matrix(
            embedding_dim=self.embedding_dim)

        print("Loading pretrained model: '{}'\n".format(trained_model))
        self.model = load_model(trained_model)
        # overwrite model path
        self.model_path = trained_model.split("/")[0]

        test_instances_list = []
        # create a label and embedding matrix from the given test data
        # file
        with open(x_val_file, "r") as read_handle:
            lines = read_handle.readlines()
            x_val = np.zeros((len(lines), self.embedding_dim))
            y_val = np.zeros((len(lines), len(self.labels)))

            for i, line in enumerate(lines):
                instance = line.split()[0]
                test_instances_list.append(instance)
                idx_of_instance = self.word2id[instance]

                # create the validation matrices
                x_val[i] = embed_matrix[idx_of_instance]
                y_val[i] = label_matrix[idx_of_instance]

        # predict the labels for the given test data and evaluate the
        # model's performance
        self.predict_and_evaluate(
            x_val=x_val, y_val=y_val, test_instances=test_instances_list)

    def train_with_cross_validation(self,
                                    embedding_matrix,
                                    label_matrix,
                                    callbacks,
                                    save_best_model=True):
        """Run the training with cross validation.

        Args:
            embedding_matrix:   numpy ndarray
                                The data matrix.
            label_matrix:   numpy ndarray
                            The label matrix.
            callbacks:      list
                            Callbacks like metrics and early stopping.
            save_best_model:    bool
                                If True, save the best model as h5 file.
        """
        best_f1 = -1.0
        kf = KFold(
            n_splits=self.config["cross_val"], shuffle=True, random_state=42)

        for i, (train_idx, dev_idx) in enumerate(
                kf.split(embedding_matrix, label_matrix)):

            print("\nTraining on fold {} ...".format(i + 1))

            x_train_cv = embedding_matrix[train_idx]
            y_train_cv = label_matrix[train_idx]

            # calculate class weights if specified in config file
            class_weights = None
            if self.config["class_weights"]:
                class_weights = class_weight.compute_class_weight(
                    "balanced", np.unique(y_train_cv), y_train_cv.flatten())

            # create the dev data
            x_dev_cv = embedding_matrix[dev_idx]
            y_dev_cv = label_matrix[dev_idx]

            # TODO: this is redundant
            x_train_cv = np.expand_dims(x_train_cv, axis=2)
            x_dev_cv = np.expand_dims(x_dev_cv, axis=2)

            # build and compile the model
            model = self.get_model(num_output_units=label_matrix.shape[1])

            if i == 0:
                model.summary()

            # train the model
            model.fit(
                x_train_cv,
                y_train_cv,
                validation_data=(x_dev_cv, y_dev_cv),
                epochs=self.config["epochs"],
                batch_size=self.config["batch_size"],
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=True)

            # save best model wrt weighted F1 score
            current_f1 = callbacks[0].report['weighted avg']["f1-score"]
            if current_f1 > best_f1:
                best_f1 = current_f1
                self.model = model
                data = x_dev_cv
                labels = y_dev_cv

        self.predict_and_evaluate(
            data, labels, test_instances=[], save_predictions=False)

        print(self.get_results_per_class())
        print("\n")
        print(self.get_average_results())

        if save_best_model:
            self.model.save(self.model_name)

    def predict_and_evaluate(self,
                             x_val,
                             y_val,
                             test_instances,
                             save_predictions=True):
        """Predict the labels for a given dataset."""
        # TODO: this is redundant
        if len(x_val.shape) != 3:
            x_val = np.expand_dims(x_val, axis=2)

        # let the model predict
        predictions = self.model.predict(x_val)
        # assign label via a threshold
        predictions[predictions >= self.threshold] = 1
        predictions[predictions < self.threshold] = 0

        # calculate precision, recall and F1 score
        self.calculate_scores(y_val, predictions)

        # save both the predictions and the used test data (the words) to
        # text files.
        if save_predictions:

            with open(
                    "{}/predictions_{}.csv".format(self.model_path,
                                                   self.model_path),
                    "w") as write_handle:
                for i, instance in enumerate(test_instances):
                    write_handle.write("{}\t{}\n".format(
                        instance, "\t".join([str(x) for x in predictions[i]])))

            with open(
                    "{}/ground_truth_{}.csv".format(self.model_path,
                                                    self.model_path),
                    "w") as write_handle:
                for i, instance in enumerate(test_instances):
                    write_handle.write("{}\t{}\n".format(
                        instance, "\t".join([str(x) for x in y_val[i]])))

    def calculate_scores(self, y_val, predictions):
        """Calculate precision, recall and F1 score.

        Args:
            y_val:  numpy ndarray
                    The validation data.
            predictions:    numpy ndarray
                            The predicted activations.
        """
        # -------------------- Scores per concept ----------------------
        # calculate scores for all classes
        precs, recs, f1_scores, supports = precision_recall_fscore_support(
            y_val, predictions, labels=range(len(self.labels)))

        # create a dataframe for all classes
        col_names = ["concept", "precision", "recall", "F1", "support"]
        self.results_per_class_table = pd.DataFrame(
            list(zip(self.labels, precs, recs, f1_scores, supports)),
            columns=col_names)
        # -------------------- Averaged scores -------------------------
        # calculate averaged scores
        f1_weighted = f1_score(y_val, predictions, average="weighted")
        f1_macro = f1_score(y_val, predictions, average="macro")
        f1_micro = f1_score(y_val, predictions, average="micro")

        precision_weighted = precision_score(
            y_val, predictions, average="weighted")
        precision_macro = precision_score(y_val, predictions, average="macro")
        precision_micro = precision_score(y_val, predictions, average="micro")

        recall_weighted = recall_score(y_val, predictions, average="weighted")
        recall_macro = recall_score(y_val, predictions, average="macro")
        recall_micro = recall_score(y_val, predictions, average="micro")

        # create a dataframe for the averaged scores
        col_names_avrg = ["score", "weighted", "macro", "micro"]
        self.average_results_table = pd.DataFrame(
            list(
                zip(["Precision", "Recall", "F1"],
                    [precision_weighted, recall_weighted, f1_weighted],
                    [precision_macro, recall_macro, f1_macro],
                    [precision_micro, recall_micro, f1_micro])),
            columns=col_names_avrg)

    def get_results_per_class(self):
        """Return a table with scores per class."""
        return self.results_per_class_table

    def get_average_results(self):
        """Return a table with average scores."""
        return self.average_results_table


if __name__ == '__main__':

    from models.embeddings import Embedding

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "raw_data", type=str, help=("The path to the tsv data."))

    parser.add_argument(
        "embedding_file", type=str, help=("The path to the embedding data."))

    ARGS, _ = parser.parse_known_args()

    embedding = Embedding(embedding_file=ARGS.embedding_file, voc_limit=100000)

    model = NeuralVectorConceptualizer(
        config_file=ARGS.config, embedding=embedding)
    model.train(inspect_concept=False)
