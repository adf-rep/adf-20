# ADF
The source code of ADF and experiments for: *Adaptive Deep Forest for Online Learning from Drifting Data Streams* (ICDM 2020 submission).

### Requirements

- MOA 2019.05.0 (moa.cms.waikato.ac.nz/downloads)
- Java 11 + jcommon-1.0.23, jfreechart-1.0.19, javafx-base-11
- Python 3.6 + packages from *requirements.txt*


### Algorithm
- The source code of ADF can be found in: ***src/csl/DeepForest.java***.

- Some usage examples are given in: ***src/eval/cases/df/DF_FinalAdfExperiment.java***.

### Data
- Links for all shallow streams (these have to be converted to ARFF files):
    - ACTIVITY: cis.fordham.edu/wisdm/dataset.php
    - CONNECT4: archive.ics.uci.edu/ml/datasets/connect-4
    - COVER: openml.org/d/150
    - EEG: archive.ics.uci.edu/ml/datasets/EEG+Eye+State
    - ELEC: moa.cms.waikato.ac.nz/datasets
    - GAS: archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset
    - POKER: moa.cms.waikato.ac.nz/datasets

- Links for all contextual batches that have to be downloaded:
    - CMATER-B: github.com/prabhuomkar/CMATERdb/tree/master/datasets/bangla-numerals
    - MALARIA: lhncbc.nlm.nih.gov/publication/pub9932
    - DOGSvCATS: microsoft.com/en-us/download/details.aspx?id=54765
    - INTEL: kaggle.com/puneet6060/intel-image-classification
    - IMGNETTE: github.com/fastai/imagenette
    - BBC: kaggle.com/c/learn-ai-bbc
    - AGNEWS: kaggle.com/amananandrai/ag-news-classification-dataset
    - SOGOU: academictorrents.com/details/b2b847b5e1946b0479baa838a0b0547178e5ebe8
    - SEMG: archive.ics.uci.edu/ml/datasets/sEMG+for+Basic+Hand+movements

- Some of the image streams have to be trimmed using ***scripts/gendata/trim.sh*** (remember to set a path before using it): DOGSvCATS -> 32x32, IMGNETTE -> 64x64, INTEL -> 32x32, MALARIA -> 32x32.

- To prepare base contextual ARFF streams: set paths in ***scripts/gendata/main.py*** and run the script.

- To generate drifting contextual streams: set paths in ***scripts/synth/main.py*** and run it.

- Set paths in ***src/eval/experiment/ExperimentStream.java***.

### Test
All unit tests can be found in *tests*.

### Evaluation

- In order to conduct experiments set paths in ***src/eval/Evaluator.java*** and run it. You can pick between different experiments in *runAdaptiveDeepForestExperiments()*.

- Details of experiments are defined in ***src/eval/experiment/Experiment.java*** instances. 

- Comments describe how parameters should be set for different types of data. 

- To run selected rows on specific data, simply uncomment selected streams in ***src/eval/experiment/ExperimentStream.java***