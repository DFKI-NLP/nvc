#!/usr/bin/python
# -*- coding: utf8 -*-
'''
Compute BLC according to
"An Inference Approach to Basic Level of Categorization",
Wang et al, 2015,
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/scoring.pdf

@date: 30.03.17
@author: leonhard.hennig@dfki.de (modified for Py3 and use in jupyter notebook
                                  by lisa.raithel@dfki.de, March 2019)
'''
# from http://stackoverflow.com/questions/2276200/changing-default-encoding-of-python
import time
import codecs
from scipy.sparse import csr_matrix
from scipy import log, exp
import argparse

# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')


def calc_p_e_given_c(concept_by_entity_counts):
    """
    Compute a smoothed P'(e|c) for the given matrix of (concept, entity/instance) counts.
    P'(e|c)  = (N(c,e) + eps) / (Sum over e_i N(c,e_i) + eps  * |columns|)

    N_instances, the normalizing value for smoothing, is the number of unique(?!) instances, not the sum of all
    counts! therefore, we use concept_by_entity_counts.shape[0]

    See https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/scoring.pdf, Sec 3.4, Eq. 2

    :param concept_by_entity_counts:
    :return:
    """
    # global log_smoothed_p_e_given_c, concept_idx, entity_idx
    # not vectorized, rather inefficient
    # N_instances, the normalizing value for smoothing, is the number of unique(?!) instances, not the sum of all
    # counts! therefore, we use concept_by_entity_counts.shape[0]
    log_smoothed_p_e_given_c = {}
    for concept_idx in range(concept_by_entity_counts.shape[0]):
        concept_row = concept_by_entity_counts.getrow(concept_idx)
        concept_row_sum = float(concept_row.sum())
        if concept_row_sum > 0:
            for entity_idx in concept_row.nonzero()[1]:
                log_smoothed_p_e_given_c[(
                    concept_idx, entity_idx
                )] = log(concept_row[0, entity_idx] + eps) - log(
                    concept_row_sum + eps * concept_by_entity_counts.shape[1])
        if concept_idx % 100 == 0:
            print("{} - Computed {}/{} p(e|c)".format(
                time.strftime("%Y/%m/%d: %H:%M:%S"), concept_idx,
                concept_by_entity_counts.shape[0]))  # for X rows (concepts)
    return log_smoothed_p_e_given_c


def calc_p_c_given_e(concept_by_entity_counts):
    """
    Compute p(c|e), unsmoothed.

    :param concept_by_entity_counts:
    :return:
    """
    log_p_c_given_e = {}
    concept_by_entity_counts_by_col = concept_by_entity_counts.tocsc()
    for entity_idx in range(concept_by_entity_counts_by_col.shape[1]):
        entity_col = concept_by_entity_counts_by_col.getcol(entity_idx)
        entity_col_sum = entity_col.sum()
        if entity_col_sum > 0:
            for concept_idx in entity_col.nonzero()[0]:
                log_p_c_given_e[(concept_idx, entity_idx)] = log(
                    entity_col[concept_idx, 0]) - log(entity_col_sum)
        if entity_idx % 100 == 0:
            print("{} - Computed {}/{} p(c|e)".format(
                time.strftime("%Y/%m/%d: %H:%M:%S"), entity_idx,
                concept_by_entity_counts_by_col.shape[1]))
    return log_p_c_given_e


def write_results(outfile, m, log_smoothed_p_e_given_c, log_p_c_given_e,
                  concept_dict_rev, entity_dict_rev):

    with codecs.open(outfile, 'wb', 'utf8') as f:
        for (i, (concept_idx,
                 entity_idx)) in enumerate(log_smoothed_p_e_given_c):
            if (concept_idx, entity_idx) in log_p_c_given_e:
                rep_e_c = log_smoothed_p_e_given_c[(
                    concept_idx, entity_idx)] + log_p_c_given_e[(concept_idx,
                                                                 entity_idx)]
                # write rep
                f.write(
                    "%s\t%s\t%d\t%.6f\t%.6f\t%.12f\n" %
                    (concept_dict_rev[concept_idx],
                     entity_dict_rev[entity_idx], m[concept_idx, entity_idx],
                     exp(log_p_c_given_e[(concept_idx, entity_idx)]),
                     exp(log_smoothed_p_e_given_c[(concept_idx, entity_idx)]),
                     exp(rep_e_c)))
            if i % 1000000 == 0:
                print("{} - Wrote {}/{} lines".format(
                    time.strftime("%Y/%m/%d: %H:%M:%S"), i,
                    len(log_smoothed_p_e_given_c)))


def init_data(infile, min_count, eps):

    concept_dict = {}
    entity_dict = {}
    concept_dict_rev = {}
    entity_dict_rev = {}
    # from example on https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
    concept_row_indices = []
    entity_col_indices = []
    data = []
    total_count = 0
    with codecs.open(infile, 'rb', 'utf8') as f:
        for (i, l) in enumerate(f):
            (concept, entity, count) = l.strip().split('\t')
            # skip low-count entries
            if int(count) <= min_count:
                continue

            concept_index = concept_dict.setdefault(concept, len(concept_dict))
            concept_dict_rev[concept_dict[concept]] = concept
            entity_index = entity_dict.setdefault(entity, len(entity_dict))
            entity_dict_rev[entity_dict[entity]] = entity
            concept_row_indices.append(concept_index)
            entity_col_indices.append(entity_index)
            data.append(int(count))
            # total_count += int(count)
            if i % 1000000 == 0:
                print("{} - Processed {} lines".format(
                    time.strftime("%Y/%m/%d: %H:%M:%S"), i))
    print("{} - Creating matrix ...".format(
        time.strftime("%Y/%m/%d: %H:%M:%S")))

    concept_by_entity_counts = csr_matrix(
        (data, (concept_row_indices, entity_col_indices)), dtype=int)

    return (concept_by_entity_counts, concept_dict_rev, entity_dict_rev)


if __name__ == '__main__':

    min_count = 0
    eps = 0.0001
    # infile = '/home/leonhard/Dokumente/forschung/data/ms-concept-graph/data-concept-instance-relations.txt'
    # infile = '/home/leonhard/Dokumente/forschung/data/ms-concept-graph/test-microsoft.txt'
    # outfile = '/home/leonhard/Dokumente/forschung/data/ms-concept-graph/data-concept-instance-relations-with-blc-2.tsv'

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "infile", type=str, help="The path to the original data.")

    parser.add_argument(
        "outfile", type=str, help="The path to the output file.")

    ARGS, _ = parser.parse_known_args()

    infile = ARGS.infile
    outfile = ARGS.outfile
    print("Input file: {}".format(infile))
    print("Outputfile: {}".format(outfile))

    (concept_by_entity_counts, concept_dict_rev, entity_dict_rev) = init_data(
        infile, min_count, eps)

    # compute BLC as per https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/scoring.pdf
    # P'(e|c) = N(c,e) + eps / (Sum over e_i N(c,e_i) + eps  * total_instances)
    # BLC = P(e|c) * P'(c|e)
    print("{} - Computing BLC...".format(time.strftime("%Y/%m/%d: %H:%M:%S")))

    log_smoothed_p_e_given_c = calc_p_e_given_c(
        concept_by_entity_counts)  # 12501527
    log_p_c_given_e = calc_p_c_given_e(concept_by_entity_counts)

    write_results(outfile, concept_by_entity_counts, log_smoothed_p_e_given_c,
                  log_p_c_given_e, concept_dict_rev, entity_dict_rev)

    print('{} - Done.'.format(time.strftime("%Y/%m/%d: %H:%M:%S")))
