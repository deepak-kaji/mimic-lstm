# mimic-lstm

This is a complete preprocessing, model training, and figure generation repo for "An attention based deep learning model of clinical events in the intensive care unit"

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

To begin, clone the mimic-lstm repository. Within the repository, create a folder called, "mimic_database" where the MIMIC-III will live. First request access from https://mimic.physionet.org/gettingstarted/access/, complete the required training course, then request the access to MIMIC-III. Download all MIMIC-III CSVs in the "mimic_database".

process_mimic.py contains scripts for transforming the MIMIC-III tables into a
pandas datatable with the required feature columns. Specifically, the MimicParser class will provide all required methods for this operation. Moreover, these methods are listd at the bottom of the script. In order for this script to function, a number of dependencies are required and listed below. The mimic_database folder will require a mapped elements folder insdie of it (./mimic-lstm/mimic_database/mapped_elements/) for proper functionality. Not all MimicParser methods are required for operation.

rnn_mimic.py contains classes for transforming the pandas dataframe into a 3rd order tensor (batch_size, time_steps, features). It will require pad_sequences as an included dependency. the pickle_objects method will create training sets for vancomycin, sepsis, and MI. The train models will generate the models required for all figures. 

Models and figures are generated in the ipynb notebook. Simply adjusting the target to 'MI', 'Sepsis', or 'Vancomycin' will generate the figures panels and images required for each part of the figure.

### Prerequisites
The pad_sequences.py file is reuqired for rnn_mimic.py functionality. Additionally, the attention function adapted from Phillipe Remy's Github has been cloned as attention_function.py. All networks were constructed in Tensorflow using the Keras API. The 

### Authors
Deepak A. Kaji was the core contributor for this repository. 

### License
This project is licensed under the MIT License - see the LICENSE.md file for details

### Acknowledgments
We thank Phillipe Remy for his implementation of variable level attention in Keras. 

