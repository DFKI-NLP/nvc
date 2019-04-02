##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Useful callbacks.

@author lisa.raithel@dfki.de
"""

from keras.callbacks import Callback
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, classification_report,
                             precision_recall_fscore_support)

import numpy as np
from utils import visualizations
import warnings
warnings.filterwarnings('ignore')


class MetricsCallback(Callback):
    """Custom callbacks for calculating evaluation scores."""

    def __init__(self, labels=[], threshold=0.5):
        """Initialize the callback.

        Args:
            labels: list
                    All available classes
            threshold:  float
                        The classification threshold.
        """
        self.threshold = threshold
        self.labels = labels

    def on_train_begin(self, logs={}):
        """Initialize variables at beginning of training."""
        self.losses = []
        self.aucs = []
        self.report = {}
        self.f1s_weighted = []
        self.f1s_macro = []
        self.f1s_micro = []

        self.recalls_weighted = []
        self.recalls_macro = []
        self.recalls_micro = []

        self.precisions_weighted = []
        self.precisions_macro = []
        self.precisions_micro = []

    def _show_random_results(self):
        """Compare scores to scores of random predictions."""
        print("Classification report for random results:\n")
        y_true = self.validation_data[1]

        random_predictions = np.random.uniform(0., 1.0, size=y_true.shape)
        random_predictions[random_predictions >= self.threshold] = 1
        random_predictions[random_predictions < self.threshold] = 0

        print(classification_report(
            y_true, random_predictions, target_names=self.labels, digits=3))

        _, _, _, support = precision_recall_fscore_support(
            y_true,
            random_predictions,
            average=None,
            labels=range(len(self.labels)))

        visualizations.plot_confusion_matrix(
            y_true, random_predictions, labels=self.labels, support=support)

    def _show_classification_report(self, print_report=True):
        """..."""
        y_true = self.validation_data[1]

        y_pred = self.model.predict(self.validation_data[0])

        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0

        self.report = classification_report(
            y_true,
            y_pred,
            target_names=self.labels,
            digits=3,
            output_dict=True)

        if print_report:
            print("Classification report: \n")
            print(classification_report(
                y_true,
                y_pred,
                target_names=self.labels,
                digits=3,
                output_dict=False))

        _, _, _, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.labels)))

    def on_train_end(self, logs={}):
        """dummy."""
        self._show_classification_report(print_report=False)

        # self._show_random_results()

        return

    def on_epoch_begin(self, epoch, logs={}):
        """dummy."""
        return

    def _calculate_auc(self, y_true, y_pred):
        """Compute AUC score."""
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
            self.aucs.append(roc_auc)
            return roc_auc

        except ValueError:
            return 0.0

    def _calculate_f1(self, y_true, y_pred):
        """Compute F1 score."""
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_micro = f1_score(y_true, y_pred, average="micro")

        self.f1s_weighted.append(f1_weighted)
        self.f1s_macro.append(f1_macro)
        self.f1s_micro.append(f1_micro)

        return f1_weighted

    def _calculate_precision(self, y_true, y_pred):
        """Compute precision score."""
        precision_weighted = precision_score(
            y_true, y_pred, average="weighted")
        precision_macro = precision_score(y_true, y_pred, average="macro")
        precision_micro = precision_score(y_true, y_pred, average="micro")

        self.precisions_weighted.append(precision_weighted)
        self.precisions_macro.append(precision_macro)
        self.precisions_micro.append(precision_micro)

        return precision_weighted

    def _calculate_recall(self, y_true, y_pred):
        """Compute recall."""
        recall_weighted = recall_score(y_true, y_pred, average="weighted")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        recall_micro = recall_score(y_true, y_pred, average="micro")

        self.recalls_weighted.append(recall_weighted)
        self.recalls_macro.append(recall_macro)
        self.recalls_micro.append(recall_micro)

        return recall_weighted

    def on_epoch_end(self, batch, logs={}):
        """Calculate score at the end of each epoch.

        Can be accessed during training.
        """
        self.losses.append(logs.get('loss'))

        y_true = self.validation_data[1]

        y_pred = self.model.predict(self.validation_data[0])
        y_pred[y_pred >= self.threshold] = 1
        y_pred[y_pred < self.threshold] = 0

        # self._calculate_auc(y_true, y_pred)
        f1_weighted = self._calculate_f1(y_true, y_pred)
        prec_weighted = self._calculate_precision(y_true, y_pred)
        rec_weighted = self._calculate_recall(y_true, y_pred)

        print(
            "- weighted F1: {} - weighted precision: {} - weighted recall: {}"
            " ~~ filtering warnings ~~".format(f1_weighted, prec_weighted,
                                               rec_weighted))

        return

    def on_batch_begin(self, batch, logs={}):
        """dummy."""
        return

    def on_batch_end(self, batch, logs={}):
        """dummy."""
        return
