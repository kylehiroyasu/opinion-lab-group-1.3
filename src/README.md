This folder contains the code to run experiments related to aspect based entity extraction using kcl and mcl. The most important files include the following:

- `Main.py`: Contains core code to train models via CLI, this script writes logs to the `records` folder and saves the trained models under `models` for further benchmarking. See `run.sh` for examples of usage.
- `Test.py`: Responsible for loading the relevant models and generating the aggregated F1-scores which were reported. These aggregated metrics were calculated by directly modifying the file instead of creating a CLI since we only used this script a handful of times.
- `TestJoint.py`: Script used to evaluate the joint attribute & entity performance. Based on the log folder, we automatically fetch the correct model.
- `Trainer.py`: Contains the code related to training the ABAE model with the KCL/MCL training adaption.
- `Model.py`: Contains the code which encapsulates the detailed architecture of the ABAE model.

Additional code includes the following scripts:

- `preprocess.py`: Functionality relevant for loading the original SemEval dataset and loading it into an appropriate format for a pandas DataFrame.
- `Dataset.py`: Wraps dataframe in pytorch style dataset to train models in general
- `Learners.py` and `Loss.py`: Code which originated from the original MCL/KCL repo to calculate the new loss function.
- `plots.py`: Code used to load and parse streams of saved log data to generate plots from the various model trainings. See notebooks for example of usage.
- `statistic.py`: Scripts used to generate box-whisker plots.
- `ModelTest.py`: Code which was originally used to check if model implementation was working.
