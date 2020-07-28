# opinion-lab-group-1.3
KCL and LDS Effects on Aspect Extraction

This folder contains the codebase used to run the experiments and measure results of the experiments outlined in our report. At the root level this repository consists of 5 folders:

- `data/raw/`: Contains the raw dataset from the SemEval 2016 Aspect Based Sentiment Analysis which was used throughout our experiments.
- `notebooks`: A handful of notebooks which were mostly used to interactively explore the dataset and plot the various results as we trained models
- `records` : This folder contains the logs saved as each model was trained. In general the files here are timestamped as a unique identifier and then metadata about model and performance are logged as line separated json objects. Major experiments were further organized into subfolders to keep results manageable.
- `src` : all source code related to running experiments and generating plots
- `trash` : A set of logs which we need to get rid of. To avoid the risk of deleting any useful logs the data was moved into this folder. However, the information logged here may be incorrect and is generally deemed unreliable.