import pandas as pd
from A.data_preprocessing_A import preprocess_text_NB, split_multinomialNB
import numpy as np
import random
import torch
from transformers import RobertaConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AdamW, BertConfig
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import classification_report
from ignite.metrics import Precision, Recall

import time
import datetime
from tqdm.auto import tqdm

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def predict_with_multinomialNB(books_file_path):
#     """
#     Reads a CSV file into a DataFrame, preprocesses the text data,
#     combines titles and summaries, trains a MultinomialNB model,
#     uses a validation set, and prints out a classification report.
    
#     Parameters:
#     - books_file_path (str): The filepath to the input CSV dataset.
#     """
#     # Load the dataset from the CSV file
#     data = pd.read_csv(books_file_path)
    
#     # Check for necessary columns
#     if 'title' not in data.columns or 'summary' not in data.columns or 'genre' not in data.columns:
#         raise ValueError("DataFrame must contain 'title', 'summary', and 'genre' columns.")
    
#     # Convert genre to ids and get unique genres
#     genre2id = {genre: i for i, genre in enumerate(data['genre'].unique())}
#     data["genre_id"] = data['genre'].apply(lambda a: genre2id[a])
#     unique_genres = list(genre2id.keys())  # Save unique genres for target names
    
#     # Combine title and summary for tokenization
#     data["title_and_summary"] = data.apply(lambda a: a["title"] + ". " + a["summary"], axis=1)
    
#     # Preprocess the combined text
#     data['preprocessed_text'] = data['title_and_summary'].apply(preprocess_text_NB)
    
#     # Split the data into training+validation and test sets
#     train_df, val_df, test_df = split_multinomialNB(data, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42)
    
#     # Extract features and labels
#     X_train, y_train = train_df['preprocessed_text'], train_df['genre_id']
#     X_val, y_val = val_df['preprocessed_text'], val_df['genre_id']
#     X_test, y_test = test_df['preprocessed_text'], test_df['genre_id']

#     # Create and train the pipeline
#     pipeline = Pipeline([
#         ('vectorizer', TfidfVectorizer(min_df=5, max_df=0.85, ngram_range=(1, 2))),
#         ('classifier', MultinomialNB(alpha=0.01)),
#     ])
    
#     pipeline.fit(X_train, y_train)
    
#     # Evaluate the model on the validation set
#     val_predictions = pipeline.predict(X_val)
#     print("Validation Set Classification Report:")
#     print(classification_report(y_val, val_predictions, target_names=unique_genres))
    
#     # Make predictions and evaluate the model on the test set
#     test_predictions = pipeline.predict(X_test)
#     print("Test Set Classification Report:")
#     print(classification_report(y_test, test_predictions, target_names=unique_genres))
    
#     # Display confusion matrix for the test set
#     cm = confusion_matrix(y_test, test_predictions)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_genres)
#     print(f"Test Set Confusion Matrix:")
#     disp.plot(cmap=plt.cm.Blues)
#     plt.xticks(rotation=30)  # Rotate x-axis labels
#     plt.yticks(rotation=30)  # Rotate y-axis labels
#     plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
#     plt.show()


def predict_with_multinomialNB(books_file_path):
    # Load the dataset from the CSV file
    data = pd.read_csv(books_file_path)
    
    # Check for necessary columns
    if 'title' not in data.columns or 'summary' not in data.columns or 'genre' not in data.columns:
        raise ValueError("DataFrame must contain 'title', 'summary', and 'genre' columns.")
    
    # Convert genre to ids and get unique genres
    genre2id = {genre: i for i, genre in enumerate(data['genre'].unique())}
    data["genre_id"] = data['genre'].apply(lambda a: genre2id[a])
    unique_genres = list(genre2id.keys())  # Save unique genres for target names
    
    # Combine title and summary for tokenization
    data["title_and_summary"] = data.apply(lambda a: a["title"] + ". " + a["summary"], axis=1)
    
    # Preprocess the combined text
    data['preprocessed_text'] = data['title_and_summary'].apply(preprocess_text_NB)
    
    # Split the data into training+validation and test sets
    train_df, val_df, test_df = split_multinomialNB(data, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42)
    
    # Extract features and labels
    X_train, y_train = train_df['preprocessed_text'], train_df['genre_id']
    X_val, y_val = val_df['preprocessed_text'], val_df['genre_id']
    X_test, y_test = test_df['preprocessed_text'], test_df['genre_id']
    

    # Define your pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB()),
    ])

    # Define the parameter grid
    param_grid = {
        'vectorizer__min_df': [3, 5, 7],
        'vectorizer__max_df': [0.75, 0.85, 0.95],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.001, 0.01, 0.1, 1],
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # Split data remains the same
    X_train_val, X_test, y_train_val, y_test = train_test_split(data['preprocessed_text'], data['genre_id'], test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    # Fit the GridSearchCV to find the best model
    grid_search.fit(X_train, y_train)

    # After fitting, GridSearchCV automatically uses the best parameter combination found
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)

    # Evaluate on the validation set
    val_predictions = grid_search.predict(X_val)
    print("Validation Set Classification Report:")
    print(classification_report(y_val, val_predictions, target_names=unique_genres))

    # Evaluate on the test set
    test_predictions = grid_search.predict(X_test)
    print("Test Set Classification Report:")
    print(classification_report(y_test, test_predictions, target_names=unique_genres))

    # Display confusion matrix for the test set
    cm = confusion_matrix(y_test, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_genres)
    print("Test Set Confusion Matrix:")
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)
    plt.tight_layout()
    plt.show()


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))    # Round to the nearest second
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train_model2(train_dataloader, validation_dataloader, num_labels, model_path, epochs=3, learning_rate=2e-5, epsilon=1e-8):
    
    configuration = RobertaConfig() # Initializing a RoBERTa configuration
    configuration.num_labels = num_labels

    # tokenizer = AutoTokenizer.from_pretrained("roberta-base", max_length = 512)

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = num_labels, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        #output_attentions = False, # Whether the model returns attentions weights.
        #output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    batch_loss = 0

    for epoch_i in range(0, epochs):

        # Training

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):


            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()        

            res = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 labels=b_labels)
            loss = res['loss']
            logits = res['logits']

            total_train_loss += loss.item()
            batch_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            # Clean up GPU memory
            if step % 100 == 0:
                torch.cuda.empty_cache()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))


        # Validation

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)


            with torch.no_grad():        
                res = model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
            loss = res['loss']
            logits = res['logits']

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
      # Save the model weights
    #model_save_path = 'A/models/roberta.bin'
    torch.save(model.state_dict(), model_path)
    return training_stats


# def train_model(train_dataloader, validation_dataloader, num_labels, model_name, model_path, epochs=3, learning_rate=2e-5, epsilon=1e-8):
#     # Load model configuration, model, and tokenizer dynamically based on model_name
#     config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
#     #tokenizer = AutoTokenizer.from_pretrained(model_name)

#     model.to(device) # Move model to specified device

#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon) # Initialize optimizer
#     total_steps = len(train_dataloader) * epochs

#     # Create the learning rate scheduler
#     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

#     seed_val = 42 # Set the seed value all over the place to make this reproducible.

#     random.seed(seed_val)
#     np.random.seed(seed_val)
#     torch.manual_seed(seed_val)
#     torch.cuda.manual_seed_all(seed_val)

#     training_stats = []  # Initialize training statistics

#     total_t0 = time.time() # Measure total training time

#     batch_loss = 0

#     for epoch_i in range(epochs): # Training loop
#         print("")
#         print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
#         print('Training...')

#         # Measure how long the training epoch takes.
#         t0 = time.time()

#         # Reset the total loss for this epoch.
#         total_train_loss = 0
#         model.train()

#         # Training step
#         # For each batch of training data...
#         for step, batch in enumerate(train_dataloader):
#             b_input_ids = batch[0].to(device)
#             b_input_mask = batch[1].to(device)
#             b_labels = batch[2].to(device)
#             model.zero_grad()        

#             res = model(b_input_ids, 
#                                  token_type_ids=None, 
#                                  attention_mask=b_input_mask, 
#                                  labels=b_labels)
#             loss = res['loss']
#             logits = res['logits']

#             total_train_loss += loss.item()
#             batch_loss += loss.item()

#             loss.backward()

#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             optimizer.step()

#             # Update the learning rate.
#             scheduler.step()

#             # Clean up GPU memory
#             if step % 100 == 0:
#                 torch.cuda.empty_cache()

#         # Calculate the average loss over all of the batches.
#         avg_train_loss = total_train_loss / len(train_dataloader)            

#         # Measure how long this epoch took.
#         training_time = format_time(time.time() - t0)

#         print("")
#         print("  Average training loss: {0:.2f}".format(avg_train_loss))
#         print("  Training epcoh took: {:}".format(training_time))

#         print("")  # Validation
#         print("Running Validation...")

#         t0 = time.time()

#         model.eval() # Put the model in evaluation mode--the dropout layers behave differently
#                      # during evaluation.

#         # Tracking variables 
#         total_eval_accuracy = 0
#         total_eval_loss = 0
#         nb_eval_steps = 0

#         # Evaluate data for one epoch
#         for batch in validation_dataloader:
#             b_input_ids = batch[0].to(device)
#             b_input_mask = batch[1].to(device)
#             b_labels = batch[2].to(device)

#             with torch.no_grad():        
#                 res = model(b_input_ids, 
#                                        token_type_ids=None, 
#                                        attention_mask=b_input_mask,
#                                        labels=b_labels)
#             loss = res['loss']
#             logits = res['logits']

#             # Accumulate the validation loss.
#             total_eval_loss += loss.item()

#             logits = logits.detach().cpu().numpy()
#             label_ids = b_labels.to('cpu').numpy()

#             total_eval_accuracy += flat_accuracy(logits, label_ids)

#         # Report the final accuracy for this validation run.
#         avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
#         print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

#         # Calculate the average loss over all of the batches.
#         avg_val_loss = total_eval_loss / len(validation_dataloader)

#         # Measure how long the validation run took.
#         validation_time = format_time(time.time() - t0)

#         print("  Validation Loss: {0:.2f}".format(avg_val_loss))
#         print("  Validation took: {:}".format(validation_time))

#         # Record all statistics from this epoch.
#         training_stats.append(
#             {'epoch': epoch_i + 1,
#                 'Training Loss': avg_train_loss,
#                 'Valid. Loss': avg_val_loss,
#                 'Valid. Accur.': avg_val_accuracy,
#                 'Training Time': training_time,
#                 'Validation Time': validation_time
#             })

#     print("")
#     print("Training complete!")
#     print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
#     # Save the model weights
#     #model_save_path = 'A/models/roberta.bin'

#     torch.save(model.state_dict(), model_path)
#     return training_stats

def train_model(train_dataloader, validation_dataloader, num_labels, model_name, model_path, epochs=3, learning_rate=2e-5, epsilon=1e-8):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)  # Move model to specified device

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        total_train_accuracy = 0  # Initialize tracking for accuracy

        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_train_accuracy += flat_accuracy(logits, label_ids)

            if step % 100 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print(f"\n  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training accuracy: {avg_train_accuracy:.2f}")
        print(f"  Training epoch took: {training_time}")

        # Validation
        print("\nRunning Validation...")
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0

        t0 = time.time()
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs.loss
            logits = outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print(f"  Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")

        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Accuracy': avg_train_accuracy,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")
    torch.save(model.state_dict(), model_path)
    return training_stats



# def plot_training_curves(training_stats):
#     # Extract training and validation loss, and accuracies
#     train_loss = [entry['Training Loss'] for entry in training_stats]
#     val_loss = [entry['Valid. Loss'] for entry in training_stats]
#     train_accuracy = [entry['Training Accuracy'] for entry in training_stats]  # Ensure this key matches your training stats
#     val_accuracy = [entry['Valid. Accur.'] for entry in training_stats]
#     epochs = range(1, len(training_stats) + 1)

#     # Plot training and validation loss
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
#     plt.plot(epochs, val_loss, 'r-o', label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.xticks(epochs)
#     plt.show()

#     # Plot training and validation accuracy
#     plt.figure(figsize=(12, 6))
#     plt.plot(epochs, train_accuracy, 'b-o', label='Training Accuracy')
#     plt.plot(epochs, val_accuracy, 'g-o', label='Validation Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.xticks(epochs)
#     plt.show()

def plot_training_curves(training_stats):
    # Extract training and validation loss, and accuracies
    train_loss = [entry['Training Loss'] for entry in training_stats]
    val_loss = [entry['Valid. Loss'] for entry in training_stats]
    train_accuracy = [entry['Training Accuracy'] for entry in training_stats]
    val_accuracy = [entry['Valid. Accur.'] for entry in training_stats]
    epochs = range(1, len(training_stats) + 1)

    # Decrease the figure size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))  # Smaller figure size

    # Plot training and validation loss on the first subplot
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-o', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_xticks(epochs)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Plot training and validation accuracy on the second subplot
    ax2.plot(epochs, train_accuracy, 'b-o', label='Training Accuracy')
    ax2.plot(epochs, val_accuracy, 'g-o', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_xticks(epochs)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    plt.show()  # Display the plots



def load_model_and_predict(dataloader, device, num_labels, target_names, phase="", model_path="", model_name=""):

    # Load the model architecture and weights
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    predictions, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend(np.argmax(logits, axis=1))
            labels.extend(label_ids)
    
    print(f"{phase} Classification Report:")
    print(classification_report(labels, predictions, target_names=target_names))
    
    # Display confusion matrix
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    print(f"{phase} Confusion Matrix:")
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=30)  # Rotate x-axis labels
    plt.yticks(rotation=30)  # Rotate y-axis labels
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()
    