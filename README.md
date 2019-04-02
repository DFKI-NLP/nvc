# Neural Vector Conceptualization (NVC)

A new method for interpreting arbitrary word vector samples.

Accompanying code for the paper:
```
@inproceedings{Schwarzenberg_nvc_2019,
  title = {Neural Vector Conceptualization for Word Vector Space Interpretation},
  booktitle = {Proceedings of the NAACL-HLT 2019 Workshop on Evaluating Vector Space Representations for NLP (RepEval)},
  author = {Schwarzenberg, Robert and Raithel, Lisa and Harbecke, David},
  address = {Minneapolis, Minnesota, USA},
  year = {2019}
  }
```
![nvc](https://github.com/DFKI-NLP/nvc/blob/master/data/drawing_listening_corrected.png)
## Installation

Create and activate an environment with Python 3.6.

```
conda create --name NVC python=3.6
source activate NVC
```
Make sure, git-lfs is installed.

Note: when cloning the repository, 800 MB of data will be downloaded from the GitHub LFS server automatically (if git-lfs is installed).

After cloning the repository, install requirements.

```
pip install -r requirements.txt
```
Unzip the data:

```
unzip data/input_data.zip
```



## Data

### Word Vectors:

The current underlying word vectors were learned with the word2vec model (Mikolov et al., 2013). Please download the pre-trained word vectors to the `data` directory: https://code.google.com/archive/p/word2vec/ (`GoogleNews-vectors-negative300.bin.gz.`)

### Microsoft Concept Graph

- For using the Microsoft Concept Graph data from scratch, please download the data here: https://concept.research.microsoft.com/Home/Download. This data dump does only comprise concepts, instances and associated counts, no probabilities or REP values. These need to be calculated before training. Therefore, see the script `utils/ms_concept_graph_scoring.py` in the notebook which calculates all needed probabilities and writes the data to a TSV file.

- The resulting file can be found in `data/data-concept-instance-relations-with-rep.tsv`.
To get the REP values, follow the procedure as described in the notebook. The result will be a JSON file `data/raw_data_dict.json`.

- **We recommend to use the preprocessed data `data/raw_data_dict.json` which includes all concepts and instances with their corresponding REP values in a JSON file.**

## Run and Replicate Experiments with the Notebook

The jupyter notebook `demo_nvc.ipynb` demonstrates how to use our (pre-)trained neural vector conceptualization (NVC) model to display the reported activation profiles.

Start the notebook:

```
jupyter notebook demo_nvc.ipynb 
```

or 

```
EXPORT CUDA_VISIBLE_DEVICES=DEVICE_NUM jupyter notebook demo_nvc.ipynb
```
if you want to run it on a GPU.


You can run two versions of the notebook:
  1. Use our pre-trained NVC model
  2. Train a new model

### Use the pre-trained model

If you run all cells of the notebook in the given order and without changing anything, our pre-trained NVC model is applied to a given filtered dataset and the results are reported at the end of the notebook.


### Train a new model

1. Comment cell #4 and #5.
2. Uncomment cell #6:
    1. specify the data you want to use:
        1. either the same filtered data as above
        2. or a differently filtered data
    2. specify the file containing the word vectors
    3. specify the configuration file
3. Load the necessary modules: the embedding and model (cell #7)
4. In cell #9, load the data: `nvc.load_data()` is callable in three versions:
    1. `nvc.load_data(path_to_data=path_to_filtered_data, filtered=True)` use already filtered data.
    2. `nvc.load_data(path_to_data=path_to_raw_data, filtered=False)` use raw data and filter it according to the parameters set in the configuration.
    3. `nvc.load_data(path_to_data=path_to_raw_data, filtered=False, selected_concepts=["city", "province"])` use raw data and filter it according to the parameters set in the configuration *and* according to a list of selected concepts.
5. Run `nvc.train()` (as in cell #14) to train a new model. The data split etc. is set in the configuration file.
6. The remaining cells display the activation profiles and report the results achieved by the model

