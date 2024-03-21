import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import warnings

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

# Now that we've ensured the resources are downloaded, we can safely import and use them
Stopwords = set(stopwords.words('english'))


def split_multinomialNB(data_frame, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42):
    """
    Splits a DataFrame into training, validation, and test sets.
    """
    assert train_frac + test_frac + val_frac == 1, "Fractions must sum to 1"
    
    # First split: separate out the test set
    train_val_df, test_df = train_test_split(data_frame, test_size=test_frac, random_state=seed)
    
    # Adjust train_frac to account for the initial split
    adjusted_train_frac = train_frac / (train_frac + val_frac)
    
    # Second split: split the remaining data into training and validation sets
    train_df, val_df = train_test_split(train_val_df, test_size=(1 - adjusted_train_frac), random_state=seed)
    
    return train_df, val_df, test_df



def preprocess_text_NB(text):
    """
    Preprocesses input text by lowercasing, removing punctuation, 
    tokenizing, removing stopwords, and stemming.
    
    Parameters:
    - text (str): The text to preprocess.
    
    Returns:
    - str: The preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Optionally, perform stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Rejoin tokens into a single string
    preprocessed_text = ' '.join(stemmed_tokens)
    
    return preprocessed_text



def clean(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Tokenize text
    text_tokens = word_tokenize(text)
    
    # Remove stopwords and words shorter than 4 characters
    filtered_tokens = [word for word in text_tokens if word not in Stopwords and len(word) > 3]
    
    # Rejoin tokens into a string
    text = ' '.join(filtered_tokens)
    
    # Remove single characters (this also removes spaces around the single character, so it's combined with the next rule)
    # Remove HTML tags (though this might be better placed before tokenization if HTML tags are not split correctly)
    text = re.sub('<.*?>+', ' ', text)
    
    # Remove extra spaces - combined the two rules into one for efficiency
    text = re.sub(r'\s+[a-zA-Z]\s+|\n|\s+', ' ', text)
    
    return text


# def eda_books(books_file_path, plt_title1 = '', plt_title2 = '', plt_title3 = ''):

#     # Load data
#     data = pd.read_csv(books_file_path, index_col="index")
#     print('Number of samples: ', data['genre'].count())
#     print(data['genre'].value_counts())

#     # Set up the matplotlib figure (1 row, 3 columns)
#     fig, axs = plt.subplots(1, 3, figsize=(20, 5))

#     # Adjust layout
#     plt.tight_layout(pad=4.0)

#     # Plotting genre distribution
#     axs[0].bar(data['genre'].value_counts().index, data['genre'].value_counts().values)
#     axs[0].set_title(plt_title1)
#     axs[0].set_xlabel('Genre')
#     axs[0].set_ylabel('Count')
#     axs[0].tick_params(axis='x', rotation=30)

#     # Calculate and display histogram for the length of summaries
#     summary_lengths = data['summary'].str.split().apply(len)
#     axs[1].hist(summary_lengths, bins=20)
#     axs[1].set_title(plt_title2)
#     axs[1].set_xlabel('Length of Summary')
#     axs[1].set_ylabel('Count')

#     # Calculate average summary length per genre
#     data['summary_length'] = data['summary'].apply(lambda x: len(x.split()))
#     average_lengths_per_genre = data.groupby('genre')['summary_length'].mean()
    
#     axs[2].bar(average_lengths_per_genre.index, average_lengths_per_genre.values)
#     axs[2].set_title(plt_title3)
#     axs[2].set_xlabel('Genre')
#     axs[2].set_ylabel('Average Length of Summary')
#     axs[2].tick_params(axis='x', rotation=30)

#     plt.show()


def eda_books(books_file_path, plt_title1='', plt_title2='', plt_title3=''):
    # Load data
    data = pd.read_csv(books_file_path, index_col="index")
    print('Number of samples: ', data['genre'].count())
    print(data['genre'].value_counts())

    # Set up the matplotlib figure (1 row, 3 columns)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # Adjust layout
    plt.tight_layout(pad=4.0)

    # For genre distribution, adjust to remove deprecation warning
    sns.countplot(x='genre', data=data, ax=axs[0], palette="Set2", legend=False)
    axs[0].set_title(plt_title1)
    axs[0].set_xlabel('Genre')
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x', rotation=30)

    # Calculate and display histogram for the length of summaries
    summary_lengths = data['summary'].str.split().apply(len)
    sns.histplot(summary_lengths, bins=20, ax=axs[1], kde=False, color='skyblue')
    axs[1].set_title(plt_title2)
    axs[1].set_xlabel('Length of Summary')
    axs[1].set_ylabel('Count')

    # Calculate average summary length per genre
    data['summary_length'] = data['summary'].apply(lambda x: len(x.split()))
    average_lengths_per_genre = data.groupby('genre')['summary_length'].mean()

    # For average summary length per genre, adjust to remove deprecation warning
    sns.barplot(x=average_lengths_per_genre.index, y=average_lengths_per_genre.values, ax=axs[2], palette='Set2', legend=False)
    axs[2].set_title(plt_title3)
    axs[2].set_xlabel('Genre')
    axs[2].set_ylabel('Average Length of Summary')
    axs[2].tick_params(axis='x', rotation=30)

    # Display the plots
    plt.show()



def proportional_sampling_min(books_file_path, reduced_books_file_path, target_samples=400, minimum_samples=200):

    # Load the dataset
    data = pd.read_csv(books_file_path, index_col="index")
    
    # Determine the number of samples for each genre
    samples_per_genre = data['genre'].value_counts()
    
    # Find the maximum number of samples in any genre above the minimum threshold
    max_samples = samples_per_genre[samples_per_genre > minimum_samples].max()
    
    # Initialize an empty dataframe to store the reduced dataset
    reduced_data = pd.DataFrame()
    
    # Iterate through each genre and reduce the number of samples proportionally, if above minimum_samples
    for genre, count in samples_per_genre.items():
        if count > minimum_samples:
            # Calculate the reduction factor to maintain proportionality above the minimum sample threshold
            reduction_factor = min(target_samples / max_samples, 1)
            target_count = int(count * reduction_factor)
            
            reduced_samples = data[data['genre'] == genre].sample(n=target_count, random_state=42)
        else:
            # If the genre has less or equal to the minimum threshold, keep all samples
            reduced_samples = data[data['genre'] == genre]
        
        # Append the reduced samples for this genre to the reduced_data dataframe
        reduced_data = pd.concat([reduced_data, reduced_samples])
    
    # Save the reduced dataset to a new file or overwrite the old one
    reduced_data.to_csv(reduced_books_file_path)

    return reduced_books_file_path, reduced_data


def preprocess_data(reduced_books_file_path, tokenizer_name="roberta-base", max_length=512):
    # Load data
    data = pd.read_csv(reduced_books_file_path, index_col="index")
    num_labels = len(data.genre.unique())
    
    # Convert genre to ids and get unique genres
    genre2id = {genre: i for i, genre in enumerate(data.genre.unique())}
    data["genre_id"] = data.genre.apply(lambda a: genre2id[a])
    unique_genres = data.genre.unique()  # Save unique genres for target names

    
    # Combine title and summary for tokenization
    data["title_and_summary"] = data.apply(lambda a: a["title"] + "." + a["summary"], axis=1)
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_length=max_length)
    input_ids, attention_masks = [], []
    for sent in data["title_and_summary"].values:
        encoded_dict = tokenizer.encode_plus(
            sent, add_special_tokens=True, max_length=max_length, truncation=True,
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



def preprocess_and_save(books_file_path, preprocessed_books_path):

    # Load data
    data = pd.read_csv(books_file_path, index_col="index")
    
    # Preprocess title and summary by cleaning
    data["title"] = data["title"].apply(clean)
    data["summary"] = data["summary"].apply(clean)

    # Combine cleaned title and summary for later use
    data["title_and_summary"] = data["title"] + ". " + data["summary"]
    
    #preprocessed_books_path = "./Datasets/preprocessed_books.csv"
    data.to_csv(preprocessed_books_path)
    
    return preprocessed_books_path



def tokenize_dataset(preprocessed_books_path, tokenizer_name="roberta-base", max_length=512):
    # Load preprocessed data
    data = pd.read_csv(preprocessed_books_path, index_col="index")
    num_labels = len(data.genre.unique())
    
    # Convert genre to ids and get unique genres
    genre2id = {genre: i for i, genre in enumerate(data.genre.unique())}
    data["genre_id"] = data.genre.apply(lambda a: genre2id[a])
    unique_genres = data.genre.unique()  # Save unique genres for target names
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_length=max_length)
    input_ids, attention_masks = [], []
    for sent in data["title_and_summary"].values:
        encoded_dict = tokenizer.encode_plus(
            sent, add_special_tokens=True, max_length=max_length, truncation=True,
            padding='max_length', return_attention_mask=True, return_tensors='pt',
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
    
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    valid_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
    
    return train_dataloader, valid_dataloader, test_dataloader






# def proportional_sampling(books_file_path, reduced_books_file_path, target_samples=400):

#     # Load the dataset
#     data = pd.read_csv(books_file_path, index_col="index")
    
#     # Determine the number of samples for each genre
#     samples_per_genre = data['genre'].value_counts()
    
#     # Find the maximum number of samples in any genre
#     max_samples = samples_per_genre.max()
    
#     # Calculate the reduction factor to maintain proportionality
#     reduction_factor = target_samples / max_samples
    
#     # Initialize an empty dataframe to store the reduced dataset
#     reduced_data = pd.DataFrame()
    
#     # Iterate through each genre and reduce the number of samples proportionally
#     for genre, count in samples_per_genre.items():
#         # Calculate the new target count for this genre, maintaining proportionality
#         target_count = int(count * reduction_factor)
        
#         # Ensure at least 1 sample is selected for very small genres
#         target_count = max(1, target_count)
        
#         reduced_samples = data[data['genre'] == genre].sample(n=target_count, random_state=42)
        
#         # Append the reduced samples for this genre to the reduced_data dataframe
#         reduced_data = pd.concat([reduced_data, reduced_samples])
    
#     # Save the reduced dataset to a new file or overwrite the old one
#     reduced_data.to_csv(reduced_books_file_path)

#     return reduced_books_file_path, reduced_data