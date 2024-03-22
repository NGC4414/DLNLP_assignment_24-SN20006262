import sys
import os
import torch

from B.data_preprocessing_B import create_data_loaders, prepare_and_clean_dataset, preprocess_lyrics_data, proportional_sampling, split_dataset
from A.model import plot_training_curves, predict_with_multinomialNB, train_model, load_model_and_predict

def print_menu_B():
    print("##################################")
    print("----------------EDA---------------")
    print('1. EDA - Exploratory Data Analysis')  
    print("-------------Baseline-------------")
    print("2. Naive Bayes classifier")
    print("--------------RoBERTa-------------")
    print("3. RoBERTa (pre-trained) sampled")
    print("4. RoBERTa (train on your machine) sampled")
    print("--------------ALBERT-------------")
    print("5. ALBERT (pre-trained)")
    print("6. ALBERT (train on your machine)")
    print("7. Exit program")
    print("##################################")

train_path= './Datasets/lyrics/train.csv' 
test_path = './Datasets/lyrics/test.csv'

def run_task_B():     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() # function to release all the memory that can be freed
    while True:
        print_menu_B()
        try:
            option = int(input("Enter the option number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if option == 1:
            print("----------Exploratory data analysis----------")
            

        elif option == 2:
            print("Executing Naive Bayes classifier")
            pass 


        elif option == 3:
            if os.path.exists('A/models/roberta.bin'):
                
                pass
            else:
                print("Pre-trained model not found. Please train a model first.")
            pass
           

        elif option == 4:
            print('Training roberta on song lyrics...')
            reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
            cleaned_music_path = './Datasets/lyrics/clean_lyrics.csv'
            train_dataset_path=  './Datasets/lyrics/Roberta/train_dataset.pt'
            val_dataset_path= './Datasets/lyrics/Roberta/val_dataset.pt'
            test_dataset_path = './Datasets/lyrics/Roberta/test_dataset.pt'

            
            sampled_data, reduced_lyrics_path = proportional_sampling(train_path = train_path, test_path=test_path, genre_col='Genre', min_samples_per_genre=200, reduced_lyrics_path= reduced_lyrics_path)
            print('...Sampled song lyrics data')

            reduced_data, cleaned_music_path = prepare_and_clean_dataset(reduced_lyrics_path,  cleaned_music_path=cleaned_music_path)
            print('...Cleaned dataset')

            dataset, unique_genres, num_labels = preprocess_lyrics_data(cleaned_music_path, tokenizer_name="roberta-base", max_length=512)
            print('...Tokenizing dataset')

            train_dataset, val_dataset, test_dataset= split_dataset(dataset, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42, 
            train_dataset_path= train_dataset_path, val_dataset_path = val_dataset_path, test_dataset_path = test_dataset_path)
            print('...Splitting dataset')

            train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16)
            print('...Creating dataloaders')

            print("Training RoBERTa model.")
            training_stats = train_model(train_dataloader, valid_dataloader, num_labels, model_name="roberta-base", model_path= 'B/models/roberta.bin', epochs=3, learning_rate=2e-5, epsilon=1e-8)
            
            plot_training_curves(training_stats)
            print("Loading pre-trained model and making predictions on validation set:")
            load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/roberta.bin', model_name="roberta-base")
            
            print("Making predictions on test set:")
            load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/roberta.bin', model_name="roberta-base")        



        elif option == 5:
            pass


        elif option == 6:
            if os.path.exists('A/models/albert.bin'):
                
                pass
            else:
                print("Pre-trained model not found. Please train a model first.")
            pass 


        elif option == 7:
            sys.exit()

        else:
            print("Invalid option. Please enter 1, 2, 3, 4, 5, or 7")
