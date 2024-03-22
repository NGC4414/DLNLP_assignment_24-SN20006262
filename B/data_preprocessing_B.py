import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import warnings
from B import utils

# Suppress specific FutureWarnings from Seaborn
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# nltk.download('punkt')
# nltk.download('stopwords')

# Attempt to load the resources. If they're not found, download them.
try:
    # This checks if the 'punkt' tokenizer models are already downloaded
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # If not, download them
    nltk.download('punkt')

try:
    # This checks if the 'stopwords' dataset is already downloaded
    nltk.data.find('corpora/stopwords')
except LookupError:
    # If not, download it
    nltk.download('stopwords')

Stopwords = set(stopwords.words('english'))


def proportional_sampling(train_path = '', test_path='', genre_col='Genre', min_samples_per_genre=200, reduced_lyrics_path=''):
    # Load and concatenate datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    data = pd.concat([train_data, test_data], axis=0)
    data = data.drop_duplicates().reset_index(drop=True)

    # Calculate the base size for sampling based on the smallest genre size
    base_size = data[genre_col].value_counts().min()
    
    # Calculate sampling ratios
    genre_counts = data[genre_col].value_counts()
    sampling_ratios = {genre: min(max(min_samples_per_genre, (count / base_size) * min_samples_per_genre), count) 
                       for genre, count in genre_counts.items()}

    # Sample the dataset according to calculated ratios
    sampled_data = pd.DataFrame()
    for genre, samples_needed in sampling_ratios.items():
        genre_data = data[data[genre_col] == genre]
        sampled_genre_data = genre_data.sample(n=int(samples_needed), random_state=1)
        sampled_data = pd.concat([sampled_data, sampled_genre_data], ignore_index=True)

    sampled_data.to_csv(reduced_lyrics_path, index=False)
    
    return sampled_data, reduced_lyrics_path

# Proportionally sample the dataset before cleaning
#sampled_data = proportional_sampling(train_path = '', test_path='', genre_col='Genre', min_samples_per_genre=200, reduced_lyrics_path= '')


def prepare_and_clean_dataset(reduced_lyrics_path,  cleaned_music_path='./Datasets/lyrics/clean_data.csv'):
    # retrieving the reduced data file
    reduced_data = pd.read_csv(reduced_lyrics_path)

    # Proceed with the cleaning steps
    reduced_data = reduced_data[reduced_data['Language'] == 'en']
    reduced_data = reduced_data[(reduced_data['Genre'] == 'Hip-Hop') | (reduced_data['Genre'] == 'Pop') | (reduced_data['Genre'] == 'Country') | (reduced_data['Genre'] == 'Rock') | (reduced_data['Genre'] == 'Electronic')]
    
    reduced_data['Lyrics'] = reduced_data['Lyrics'].astype(str)
    reduced_data['Lyrics'] = utils.clean_data(reduced_data['Lyrics'])
    reduced_data = reduced_data.drop(reduced_data[reduced_data['Lyrics'].str.len() == 0].index)
    
    # save the cleaned and sampled dataset to a new file
    reduced_data.to_csv(cleaned_music_path, index=False)
    
    return reduced_data, cleaned_music_path


def preprocess_lyrics_data(cleaned_music_path , tokenizer_name="roberta-base", max_length=512):
    # Load data
    data = pd.read_csv(cleaned_music_path)
    num_labels = len(data['Genre'].unique())
    
    # Convert genre to ids and get unique genres
    genre2id = {genre: i for i, genre in enumerate(data['Genre'].unique())}
    data["genre_id"] = data['Genre'].apply(lambda a: genre2id[a])
    unique_genres = data['Genre'].unique()  # Save unique genres for target names
    
    # Tokenize lyrics
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_length=max_length)
    input_ids, attention_masks = [], []
    for lyrics in data['Lyrics'].values:
        encoded_dict = tokenizer.encode_plus(
            lyrics, add_special_tokens=True, max_length=max_length, truncation=True,
            pad_to_max_length=True, padding='max_length', return_attention_mask=True, return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    # Convert lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(data["genre_id"].values)
    
    # Create TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset, unique_genres, num_labels


def split_dataset(dataset, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42, train_dataset_path= '', val_dataset_path = '', test_dataset_path = '' ):
    
    torch.manual_seed(seed)  # For reproducibility
    
    assert train_frac + test_frac + val_frac == 1, "Fractions must sum to 1"
    
    # Calculate split sizes
    train_size = int(train_frac * len(dataset))
    test_size = int(test_frac * len(dataset))
    val_size = len(dataset) - train_size - test_size
    
    # Split the dataset
    train_dataset, remaining_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    test_dataset, val_dataset = random_split(remaining_dataset, [test_size, val_size])
    
    # Save the splits to files
    torch.save(train_dataset, train_dataset_path)
    torch.save(val_dataset, val_dataset_path)
    torch.save(test_dataset, test_dataset_path)
    
    print(f'{len(train_dataset):>5,} training samples')
    print(f'{len(val_dataset):>5,} validation samples')
    print(f'{len(test_dataset):>5,} test samples')
    
    return train_dataset, val_dataset, test_dataset



def load_datasets_from_files(train_dataset_path, val_dataset_path, test_dataset_path):
    train_dataset = torch.load(train_dataset_path)
    val_dataset = torch.load(val_dataset_path)
    test_dataset = torch.load(test_dataset_path)
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size, num_workers=4, pin_memory=True)
    valid_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size, num_workers=4, pin_memory=True)
    
    return train_dataloader, valid_dataloader, test_dataloader



