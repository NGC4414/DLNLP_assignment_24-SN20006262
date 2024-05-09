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



def eda_lyrics(processed_lyrics_path='',  plt_title1='Genre Distribution', plt_title2='Length of Lyrics', plt_title3='Average Length of Lyrics per Genre'):
    # Suppress specific FutureWarnings from Seaborn
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
    # Load  dataset
    data = pd.read_csv(processed_lyrics_path)

    # Replace NaN values in 'Lyrics' with an empty string
    #data['Lyrics'] = data['Lyrics'].fillna('')

    # Display basic dataset information
    print('Number of samples: ', data['Lyrics'].count())
    print(data['Genre'].value_counts())

    # Set up the matplotlib figure (1 row, 3 columns)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Adjust layout
    plt.tight_layout(pad=4.0)

    # Genre distribution
    sns.countplot(x='Genre', data=data, ax=axs[0], palette="Set2", hue= 'Genre')
    axs[0].set_title(plt_title1)
    axs[0].set_xlabel('Genre')
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x', rotation=35)

    # Histogram for the length of lyrics
    lyrics_lengths = data['Lyrics'].apply(lambda x: len(x.split()))
    sns.histplot(lyrics_lengths, bins=20, ax=axs[1], kde=False, color='skyblue')
    axs[1].set_title(plt_title2)
    axs[1].set_xlabel('Length of Lyrics')
    axs[1].set_ylabel('Count')

    # Calculate average lyrics length per genre
    average_lengths_per_genre = data.groupby('Genre')['Lyrics'].apply(lambda x: x.str.split().apply(len).mean())

    # Average lyrics length per genre
    sns.barplot(x=average_lengths_per_genre.index, y=average_lengths_per_genre.values, ax=axs[2], palette='Set2')
    axs[2].set_title(plt_title3)
    axs[2].set_xlabel('Genre')
    axs[2].set_ylabel('Average Length of Lyrics')
    axs[2].tick_params(axis='x', rotation=35)

    # Display the plots
    plt.show()



def proportional_sampling2(train_path = '', test_path='', genre_col='Genre', min_samples_per_genre=200, reduced_lyrics_path=''):
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



def proportional_sampling5(train_path='', test_path='', genre_col='Genre', 
                                               min_samples_per_genre=200, 
                                               reduced_lyrics_path=''):
    # Load and concatenate datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    data = pd.concat([train_data, test_data], axis=0)
    data = data.drop_duplicates().reset_index(drop=True)

    # Determine the number of samples for each genre
    genre_counts = data[genre_col].value_counts()

    # Calculate total number of samples after ensuring minimum samples for underrepresented genres
    additional_samples_needed = sum(max(min_samples_per_genre - count, 0) for count in genre_counts if count < min_samples_per_genre)
    total_samples_post_min_requirement = sum(genre_counts) + additional_samples_needed
    
    # Initialize an empty dataframe for the reduced dataset
    sampled_data = pd.DataFrame()
    
    for genre, count in genre_counts.items():
        genre_data = data[data[genre_col] == genre]
        
        if count < min_samples_per_genre:
            # For genres below the minimum threshold, use all available samples
            sampled_data = pd.concat([sampled_data, genre_data], ignore_index=True)
        else:
            # For genres above the minimum threshold, calculate new count preserving original imbalance
            proportional_count = int((count / sum(genre_counts)) * total_samples_post_min_requirement)
            target_count = max(min_samples_per_genre, proportional_count)
            
            reduced_samples = genre_data.sample(n=target_count, random_state=42)
            sampled_data = pd.concat([sampled_data, reduced_samples], ignore_index=True)

    # Save the sampled dataset
    sampled_data.to_csv(reduced_lyrics_path, index=False)
    
    return reduced_lyrics_path, sampled_data


def proportional_sampling4(train_path='', test_path='', genre_col='Genre', 
                          min_samples_per_genre=200, target_samples=500, 
                          reduced_lyrics_path=''):
    
    # Load and concatenate datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    data = pd.concat([train_data, test_data], axis=0)
    data = data.drop_duplicates().reset_index(drop=True)

    # Determine the number of samples for each genre
    genre_counts = data[genre_col].value_counts()

    # Find the maximum number of samples for genres exceeding the minimum threshold
    max_samples = genre_counts[genre_counts >= min_samples_per_genre].max()

    # Initialize an empty dataframe for the reduced dataset
    sampled_data = pd.DataFrame()

    # Adjust the reduction factor to consider the min_samples_per_genre
    if max_samples > target_samples:
        reduction_factor = target_samples / max_samples
    else:
        reduction_factor = 1

    for genre, count in genre_counts.items():
        genre_data = data[data[genre_col] == genre]
        
        if count >= min_samples_per_genre:
            target_count = max(min_samples_per_genre, int(count * reduction_factor))
        else:
            # For genres below the min threshold, attempt to use all available samples
            target_count = count

        reduced_samples = genre_data.sample(n=min(target_count, len(genre_data)), random_state=42)
        sampled_data = pd.concat([sampled_data, reduced_samples], ignore_index=True)

    # Save the sampled dataset
    sampled_data.to_csv(reduced_lyrics_path, index=False)
    
    return reduced_lyrics_path, sampled_data


# def proportional_sampling(cleaned_music_path='', genre_col='Genre', 
#                           min_samples_per_genre=300, target_samples=1000,
#                           reduced_lyrics_path=''):

#     # Load and concatenate datasets
#     data = pd.read_csv(cleaned_music_path)

#     # Calculate sampling ratios while ensuring the max_samples_per_genre constraint is respected
#     genre_counts = data[genre_col].value_counts()
#     #max_samples = min(genre_counts.max(), target_samples)          # Adjust max_samples to not exceed max_samples_per_genre
#     max_samples = genre_counts[genre_counts > min_samples_per_genre].max()
#     sampled_data = pd.DataFrame()

#     for genre, count in genre_counts.items():
#         genre_data = data[data[genre_col] == genre]
        
#         if count > min_samples_per_genre:
#             reduction_factor = min(target_samples / max_samples, 1)
#             target_count = int(count*reduction_factor)
#             reduced_samples = genre_data.sample(n=target_count, random_state=42)
#             #target_count = max(min_samples_per_genre, min(target_count, target_samples))  # Ensure target_count is within specified bounds
#         else:
#             #target_count = count  # Keep all samples if below min_samples_per_genre
#             reduced_samples = data[data[genre_col] == genre]
            
#         sampled_data = pd.concat([sampled_data, reduced_samples])

#         # sampled_genre_data = genre_data.sample(n=target_count, random_state=1)
#         # sampled_data = pd.concat([sampled_data, sampled_genre_data], ignore_index=True)

#     # Save the sampled dataset
#     sampled_data.to_csv(reduced_lyrics_path, index=False)
#     return sampled_data, reduced_lyrics_path

def proportional_sampling(cleaned_music_path='', genre_col='Genre', 
                          min_samples_per_genre=300, target_samples=1000,
                          reduced_lyrics_path=''):
    # Load the dataset
    data = pd.read_csv(cleaned_music_path)
    
    # Initialize the dataframe for sampled data
    sampled_data = pd.DataFrame()
    
    # Get counts of each genre
    genre_counts = data[genre_col].value_counts()
    
    # Find the max number of samples for genres above the minimum threshold
    max_samples = genre_counts[genre_counts >= min_samples_per_genre].max()

    # Calculate reduction factor
    reduction_factor = target_samples / max_samples if max_samples > target_samples else 1

    for genre, count in genre_counts.items():
        genre_data = data[data[genre_col] == genre]

        if count > min_samples_per_genre:
            # Calculate target count after reduction but ensure at least min_samples_per_genre
            target_count = max(int(count * reduction_factor), min_samples_per_genre)
        else:
            # Keep all samples for genres below the minimum threshold
            target_count = count

        # Sample data for this genre
        reduced_samples = genre_data.sample(n=target_count, random_state=42)
        sampled_data = pd.concat([sampled_data, reduced_samples], ignore_index=True)

    # Save the sampled dataset
    sampled_data.to_csv(reduced_lyrics_path, index=False)
    
    return sampled_data, reduced_lyrics_path


# def prepare_and_clean_dataset(train_path='./Datasets/lyrics/train.csv', test_path='./Datasets/lyrics/test.csv', cleaned_music_path='./Datasets/lyrics/clean_data.csv'):

#     # Load and concatenate datasets
#     train_data = pd.read_csv(train_path)
#     test_data = pd.read_csv(test_path)
#     cleaned_data = pd.concat([train_data, test_data], axis=0)
#     cleaned_data = cleaned_data.drop_duplicates().reset_index(drop=True)

#     # Proceed with the cleaning steps
#     cleaned_data = cleaned_data[cleaned_data['Language'] == 'en']
#     cleaned_data = cleaned_data[(cleaned_data['Genre'] == 'Hip-Hop') | (cleaned_data['Genre'] == 'Pop') | (cleaned_data['Genre'] == 'Country') | (cleaned_data['Genre'] == 'Rock') | (cleaned_data['Genre'] == 'Electronic')]
    
#     cleaned_data['Lyrics'] = cleaned_data['Lyrics'].astype(str)
#     cleaned_data['Lyrics'] = utils.clean_data(cleaned_data['Lyrics'])
#     cleaned_data = cleaned_data.drop(cleaned_data[cleaned_data['Lyrics'].str.len() == 0].index)
    
#     # save the cleaned and sampled dataset to a new file
#     cleaned_data.to_csv(cleaned_music_path, index=False)

#     return cleaned_data, cleaned_music_path


def prepare_and_clean_dataset(train_path='./Datasets/lyrics/train.csv', test_path='./Datasets/lyrics/test.csv',  cleaned_music_path='./Datasets/lyrics/clean_data.csv'):

    # Load and concatenate datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    cleaned_data = pd.concat([train_data, test_data], axis=0)
    cleaned_data = cleaned_data.drop_duplicates().reset_index(drop=True)

    # Proceed with the cleaning steps
    cleaned_data = cleaned_data[cleaned_data['Language'] == 'en']
    cleaned_data = cleaned_data[(cleaned_data['Genre'] == 'Hip-Hop') | (cleaned_data['Genre'] == 'Pop') | (cleaned_data['Genre'] == 'Country') | (cleaned_data['Genre'] == 'Rock') | (cleaned_data['Genre'] == 'Electronic')]
    
    cleaned_data['Lyrics'] = cleaned_data['Lyrics'].astype(str)
    cleaned_data['Lyrics'] = utils.clean_data(cleaned_data['Lyrics'])
    cleaned_data = cleaned_data.drop(cleaned_data[cleaned_data['Lyrics'].str.len() == 0].index)
    
    # save the cleaned and sampled dataset to a new file
    cleaned_data.to_csv(cleaned_music_path, index=False)
    
    return cleaned_data, cleaned_music_path



def prepare_and_clean_dataset2(reduced_lyrics_path,  cleaned_music_path='./Datasets/lyrics/clean_data.csv'):
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



def preprocess_lyrics_data(path='' , tokenizer_name="roberta-base", max_length=512):
    # Load data
    data = pd.read_csv(path)
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
    
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size, num_workers=0, pin_memory=True)
    valid_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size, num_workers=0, pin_memory=True)
    
    return train_dataloader, valid_dataloader, test_dataloader



