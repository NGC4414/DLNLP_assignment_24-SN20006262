This folder conatins the code for the 2023-2024 DLNLP assignment. 

The project folder is organised as follows:

-`DLNLP_assignment`
  - `Folder A` : contains the code to execute task A and the pretrained models folder `models`.
  - `Folder B` : contains the code to execute task B and the pretrained models folder `models`.
  - `Datasets`: Contains the two datasets used to run Task A (books) and Task B(lyrics). The dataset for Task A is retrievable from https://huggingface.co/datasets/azlan8289/Book_Genre. For ease of use, the books folder contains the initial dataset, the reduced dataset (to avoid sampling it again when running the code), and the Albert and Roberta folders, which contain the .pt files for models training. The lyrics dataset can be retrieved from https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification/data. The folder contains the original datasets test.cv and train.cv, as well as the cleaned and reduced versions clean_lyrics.cv and reduced_lyrics.cv, to minimise pre-processing power. The lyrics folder also contains the Roberta and Albert folders with the .pt files for model training.
  - `main.py`: file that handles the execution of the project.
  - `README.md`: this file.
  - `environment.yml` : file that contains all the libraries used and their versions. This is a .yml file because the project was run using a conda virtual environment.


# Folder A: Book-genre classification task

## Folder files
- `data_preprocessing_A.py`: file used to run the menu for task A
- `menuA.py`: file that contains all the functions that execute task A
- `modelA.py`: file used in order to accept relative paths
- `models`: pretrained model `roberta_f.bin`, `roberta_sm.bin`, `albert.bin`.

## Menu

    ##################################
    ----------------EDA---------------
    1. EDA - Exploratory Data Analysis
    -------------Baseline-------------
    2. Naive Bayes classifier
    --------------RoBERTa-------------
    4. RoBERTa (pre-trained) - full dataset
    5. RoBERTa (pre-trained) - sampled dataset
    6. RoBERTa (train on your machine) - full dataset
    7. RoBERTa (train on your machine) - sampled dataset
    --------------ALBERT-------------
    8. ALBERT (pre-trained) - sampled dataset
    9. ALBERT (train on your machine) - sampled dataset
    10. Exit program
    ##################################
    Enter the option number:


# Folder B: Music genre-classification task 

## Folder files
- `data_preprocessing_B.py`: file used to run the menu for task B.
- `utils.py`: contains functions used in the `data_preprocessing_B.py` module.
- `menuB.py`: file that contains all the functions that execute task B.
- `modelB.py`: file used in order to accept relative paths.
- `models`: pretrained model `albert_lyrics.bin`, `roberta_lyrics.bin`, `tf_albert.bin`, `tf_roberta.bin`.

## Menu

    ##################################
    ----------------EDA---------------
    1. EDA - Exploratory Data Analysis
    -------------Baseline-------------
    2. Naive Bayes classifier
    --------------RoBERTa-------------
    3. RoBERTa (pre-trained)
    4. RoBERTa (train on your machine)
    5. RoBERTa - transfer learning on music genre dataset
    6. RoBERTa - transfer learning on music genre dataset (train on your machine)
    --------------ALBERT-------------
    7. ALBERT (pre-trained)
    8. ALBERT (train on your machine)
    9. ALBERT - transfer learning on music genre dataset
    10. ALBERT - transfer learning on music genre dataset (train on your machine)
    11. Exit program
    ##################################
    Enter the option number:



# How to Run
1. Ensure all the required packages are installed.
2. To run task A write the `python main.py A` command in the terminal. To run task B write the command `python main.py B` in the terminal. The arguments `A` for `python main.py A` and `B` for `python main.py B` must be uppercase to make the program work.
4. Once the preferred task has been selected, follow the on-screen prompts to train new models or use pre-trained models for both tasks.
