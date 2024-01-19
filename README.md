# Music Classification

## Environment Setup

Anaconda was used to manage the library dependencies in this project. To setup your own environment with anaconda, run `conda install -n <env_name> requirements.txt`.

## Code Structure

The majority of code required is located in the `/src` folder. Pre-trained models are saved in `/models`, with training logs in `/logs` and saved resulting data and produced figures in `/results`.

## Testing

To generate the datasets used in this project, run the `get_mazurkas()` and `get_rondodb100()` functions in `/src/dataset_retrieval.py`. For the Covers80 dataset, follow the link in `dataset_files/covers80.txt` to download the dataset .zip from their website, and format that data by running the `get_covers80()` and `get_covers80_listfiles()` functions in `/src/dataset_retrieval.py`.

The TFKeras definitions of the models used in this project are in `/src/keras_models.py`, with wrapper classes for training, pruning, and quantization found in `/src/models.py`.

An example snippet of code to train a model is found in `/src/main.py`.

Additionally, an work-in-progress example android app to predict live sound using TFLite models is found in `/examples`.

## Results

All code used to generate resulting graphs is found in `/src/results.py`.

## Report

All figures and LaTeX code used to write the report is found in `/report`. Figures used in the report were produced with PowerPoint.
