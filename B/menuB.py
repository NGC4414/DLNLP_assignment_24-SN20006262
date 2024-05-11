import sys
import os
import torch
import warnings

from B.data_preprocessing_B import create_data_loaders, eda_lyrics, load_datasets_from_files, prepare_and_clean_dataset, preprocess_lyrics_data, proportional_sampling, split_dataset
from B.modelB import plot_training_curves, predict_with_multinomialNB_lyrics, train_model, load_model_and_predict, train_model_tf

def print_menu_B():
    print("##################################")
    print("----------------EDA---------------")
    print('1. EDA - Exploratory Data Analysis')  
    print("-------------Baseline-------------")
    print("2. Naive Bayes classifier")
    print("--------------RoBERTa-------------")
    print("3. RoBERTa (pre-trained) ")
    print("4. RoBERTa (train on your machine) ")
    print("5. RoBERTa - transfer learning on music genre dataset")
    print("6. RoBERTa - transfer learning on music genre dataset (train on your machine)")
    print("--------------ALBERT-------------")
    print("7. ALBERT (pre-trained)")
    print("8. ALBERT (train on your machine)")
    print("9. ALBERT - transfer learning on music genre dataset")
    print("10. ALBERT - transfer learning on music genre dataset (train on your machine)")
    print("11. Exit program")
    print("##################################")

train_path= './Datasets/lyrics/train.csv' 
test_path = './Datasets/lyrics/test.csv'
cleaned_music_path = './Datasets/lyrics/clean_lyrics.csv'
reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
train_dataset_path=  './Datasets/lyrics/Roberta/train_dataset.pt'
val_dataset_path= './Datasets/lyrics/Roberta/val_dataset.pt'
test_dataset_path = './Datasets/lyrics/Roberta/test_dataset.pt'
batch_size = 16


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
            print("-----------------Full Dataset----------------")
            # train_path ='./Datasets/lyrics/train.csv'
            # test_path='./Datasets/lyrics/test.csv'
            cleaned_music_path = './Datasets/lyrics/clean_lyrics.csv'
            # cleaned_data, cleaned_music_path = prepare_and_clean_dataset(train_path=train_path, 
            #                                                              test_path=test_path,  
            #                                                              cleaned_music_path=cleaned_music_path)
            eda_lyrics(processed_lyrics_path=cleaned_music_path, plt_title1='Genre Distribution', plt_title2='Length of Lyrics', plt_title3='Average Length of Lyrics per Genre')
            
            print('...Sampled song lyrics data')
            reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
            reduced_lyrics_path=proportional_sampling(cleaned_music_path=cleaned_music_path, genre_col='Genre', min_samples_per_genre=200, target_samples=1000, reduced_lyrics_path ='./Datasets/lyrics/reduced_lyrics.csv')
            
            print("----------Exploratory data analysis----------")
            print("---------------Sampled Dataset---------------")   
            eda_lyrics(processed_lyrics_path='./Datasets/lyrics/reduced_lyrics.csv', plt_title1='Genre Distribution (sampled)', plt_title2= 'Length of Lyrics (sampled)', plt_title3='Average Length of Lyrics per Genre (sampled)')
        
        elif option == 2:
            print("Executing Naive Bayes classifier")
            reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
            predict_with_multinomialNB_lyrics(reduced_lyrics_path)

        elif option == 3:
            if os.path.exists('B/models/roberta__lyrics.bin'):
                print("Executing pre-trained Roberta on sampled dataset")
                reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
                dataset, unique_genres, num_labels = preprocess_lyrics_data(path='./Datasets/lyrics/reduced_lyrics.csv', tokenizer_name="roberta-base", max_length=512)
                
                print('Loading training, validation and test pytorch datasets...')
                train_dataset, val_dataset, test_dataset = load_datasets_from_files(train_dataset_path='./Datasets/lyrics/Roberta/train_dataset.pt', 
                                                                                    val_dataset_path ='./Datasets/lyrics/Roberta/val_dataset.pt', 
                                                                                    test_dataset_path ='./Datasets/lyrics/Roberta/test_dataset.pt')
            
                print('Creating dataloaders...')
                _, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

                print("Loading pre-trained model and making predictions on validation set:")
                load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/roberta__lyrics.bin', model_name="roberta-base")
                
                print("Making predictions on test set:")
                load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/roberta__lyrics.bin', model_name="roberta-base")
            
            else:
                print("Pre-trained model not found. Please train a model first.")
            

        elif option == 4:
            print('Training roberta on song lyrics...')
            # train_path ='./Datasets/lyrics/train.csv'
            # test_path='./Datasets/lyrics/test.csv'
            cleaned_music_path = './Datasets/lyrics/clean_lyrics.csv'
            reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
            train_dataset_path=  './Datasets/lyrics/Roberta/train_dataset.pt'
            val_dataset_path= './Datasets/lyrics/Roberta/val_dataset.pt'
            test_dataset_path = './Datasets/lyrics/Roberta/test_dataset.pt'

            # cleaned_data, cleaned_music_path = prepare_and_clean_dataset(train_path=train_path, 
            #                                                              test_path=test_path,  
            #                                                              cleaned_music_path=cleaned_music_path)
            print('...Cleaned dataset')

            reduced_lyrics_path=proportional_sampling(cleaned_music_path=cleaned_music_path, genre_col='Genre', min_samples_per_genre=200, target_samples=1000, reduced_lyrics_path =reduced_lyrics_path)
            print('...Sampled song lyrics data')

            dataset, unique_genres, num_labels = preprocess_lyrics_data(path='./Datasets/lyrics/reduced_lyrics.csv', tokenizer_name="roberta-base", max_length=512)
            print('...Tokenizing dataset')

            train_dataset, val_dataset, test_dataset= split_dataset(dataset, train_frac=0.7, test_frac=0.15, val_frac=0.15, seed=42, 
            train_dataset_path= train_dataset_path, val_dataset_path = val_dataset_path, test_dataset_path = test_dataset_path)
            print('...Splitting dataset')

            train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16)
            print('...Creating dataloaders')

            print("Training RoBERTa model.")
            training_stats = train_model(train_dataloader, valid_dataloader, num_labels, model_name="roberta-base", model_path= 'B/models/roberta__lyrics.bin', epochs=3, learning_rate=2e-5, epsilon=1e-8)
            
            plot_training_curves(training_stats)
            print("Loading pre-trained model and making predictions on validation set:")
            load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/roberta__lyrics.bin', model_name="roberta-base")
            
            print("Making predictions on test set:")
            load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/roberta__lyrics.bin', model_name="roberta-base")       


        elif option == 5:
            if os.path.exists('B/models/tf_roberta.bin'):
                print("Executing Transfer learned Roberta on sampled dataset")
                reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
                dataset, unique_genres, num_labels = preprocess_lyrics_data(path='./Datasets/lyrics/reduced_lyrics.csv', tokenizer_name="roberta-base", max_length=512)
                
                print('Loading training, validation and test pytorch datasets...')
                train_dataset, val_dataset, test_dataset = load_datasets_from_files(train_dataset_path='./Datasets/lyrics/Roberta/train_dataset.pt', 
                                                                                    val_dataset_path ='./Datasets/lyrics/Roberta/val_dataset.pt', 
                                                                                    test_dataset_path ='./Datasets/lyrics/Roberta/test_dataset.pt')
            
                print('Creating dataloaders...')
                _, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

                print("Loading pre-trained model and making predictions on validation set:")
                load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/tf_roberta.bin', model_name="roberta-base")
                
                print("Making predictions on test set:")
                load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/tf_roberta.bin', model_name="roberta-base")
                
            else:
                print("Pre-trained model not found. Please train a model first.")
            

        elif option == 6:
            # train_path= './Datasets/lyrics/train.csv' 
            # test_path = './Datasets/lyrics/test.csv'
            cleaned_music_path = './Datasets/lyrics/clean_lyrics.csv'
            reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
            train_dataset_path=  './Datasets/lyrics/Roberta/train_dataset.pt'
            val_dataset_path= './Datasets/lyrics/Roberta/val_dataset.pt'
            test_dataset_path = './Datasets/lyrics/Roberta/test_dataset.pt'
            print('Transfer learning roberta on song lyrics...')
            # cleaned_data, cleaned_music_path = prepare_and_clean_dataset(train_path=train_path, 
            #                                                              test_path=test_path,  
            #                                                              cleaned_music_path=cleaned_music_path)
            print('...Cleaned dataset')
            reduced_lyrics_path=proportional_sampling(cleaned_music_path=cleaned_music_path, genre_col='Genre', min_samples_per_genre=200, target_samples=1000, 
                                                      reduced_lyrics_path =reduced_lyrics_path)
            print('...Sampled song lyrics data')

            dataset, unique_genres, num_labels = preprocess_lyrics_data(path='./Datasets/lyrics/reduced_lyrics.csv', tokenizer_name="roberta-base", max_length=512)
            print('...Tokenizing dataset')

            train_dataset, val_dataset, test_dataset= split_dataset(dataset, train_frac=0.7, test_frac=0.15, val_frac=0.15, seed=42, 
            train_dataset_path= train_dataset_path, val_dataset_path = val_dataset_path, test_dataset_path = test_dataset_path)
            print('...Splitting dataset')

            train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16)
            print('...Creating dataloaders')

            print("Training RoBERTa model.")
            training_stats = train_model_tf(train_dataloader, valid_dataloader, num_labels, model_name="roberta-base", pre_trained_model_path= 'A/models/roberta_sm.bin', 
                                            model_path= 'B/models/tf_roberta.bin', epochs=3, learning_rate=2e-5, epsilon=1e-8)
            
            plot_training_curves(training_stats)
            print("Loading pre-trained model and making predictions on validation set:")
            load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/tf_roberta.bin', model_name="roberta-base")
            
            print("Making predictions on test set:")
            load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/tf_roberta.bin', model_name="roberta-base")        
    

        elif option == 7:
            if os.path.exists('B/models/albert_lyrics.bin'):     
                print("Executing pre-trained ALBERT on sampled dataset")
                reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
                dataset, unique_genres, num_labels = preprocess_lyrics_data(path='./Datasets/lyrics/reduced_lyrics.csv', tokenizer_name="albert-base-v2", max_length=512)
                
                print('Loading training, validation and test pytorch datasets...')
                train_dataset, val_dataset, test_dataset = load_datasets_from_files(train_dataset_path='./Datasets/lyrics/Albert/train_dataset.pt', 
                                                                                    val_dataset_path ='./Datasets/lyrics/Albert/val_dataset.pt', 
                                                                                    test_dataset_path ='./Datasets/lyrics/Albert/test_dataset.pt')
            
                print('Creating dataloaders...')
                _, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

                print("Loading pre-trained model and making predictions on validation set:")
                load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/albert_lyrics.bin', model_name="albert-base-v2")
                
                print("Making predictions on test set:")
                load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/albert_lyrics.bin', model_name="albert-base-v2")
            else:

                print("Pre-trained model not found. Please train a model first.")


        elif option == 8:
            print('Training Albert on song lyrics...')
            # train_path ='./Datasets/lyrics/train.csv'
            # test_path='./Datasets/lyrics/test.csv'
            cleaned_music_path = './Datasets/lyrics/clean_lyrics.csv'
            reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
            train_dataset_path=  './Datasets/lyrics/Albert/train_dataset.pt'
            val_dataset_path= './Datasets/lyrics/Albert/val_dataset.pt'
            test_dataset_path = './Datasets/lyrics/Albert/test_dataset.pt'

            # cleaned_data, cleaned_music_path = prepare_and_clean_dataset(train_path=train_path, 
            #                                                              test_path=test_path,  
            #                                                              cleaned_music_path=cleaned_music_path)
            # print('...Cleaned dataset')

            reduced_lyrics_path=proportional_sampling(cleaned_music_path=cleaned_music_path, genre_col='Genre', min_samples_per_genre=200, target_samples=1000, reduced_lyrics_path=reduced_lyrics_path)
            print('...Sampled song lyrics data')

            dataset, unique_genres, num_labels = preprocess_lyrics_data(path='./Datasets/lyrics/reduced_lyrics.csv', tokenizer_name="albert-base-v2", max_length=512)
            print('...Tokenizing dataset')

            train_dataset, val_dataset, test_dataset= split_dataset(dataset, train_frac=0.7, test_frac=0.15, val_frac=0.15, seed=42, 
            train_dataset_path= train_dataset_path, val_dataset_path = val_dataset_path, test_dataset_path = test_dataset_path)
            print('...Splitting dataset')

            train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16)
            print('...Creating dataloaders')

            print("Training RoBERTa model.")
            training_stats = train_model(train_dataloader, valid_dataloader, num_labels, model_name="albert-base-v2", model_path= 'B/models/albert_lyrics.bin', epochs=3, learning_rate=2e-5, epsilon=1e-8)
            
            plot_training_curves(training_stats)
            print("Loading pre-trained model and making predictions on validation set:")
            load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/albert_lyrics.bin', model_name="albert-base-v2")
            
            print("Making predictions on test set:")
            load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/albert_lyrics.bin', model_name="albert-base-v2")  


        elif option == 9:
            print('Executing Transfer learned ALBERT on song lyrics...')

            if os.path.exists('B/models/tf_albert.bin'):
                print("Executing pre-trained ALBERT on sampled dataset")
                reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
                dataset, unique_genres, num_labels = preprocess_lyrics_data(path='./Datasets/lyrics/reduced_lyrics.csv', tokenizer_name="albert-base-v2", max_length=512)
                
                print('Loading training, validation and test pytorch datasets...')
                train_dataset, val_dataset, test_dataset = load_datasets_from_files(train_dataset_path='./Datasets/lyrics/Albert/train_dataset.pt', 
                                                                                    val_dataset_path ='./Datasets/lyrics/Albert/val_dataset.pt', 
                                                                                    test_dataset_path ='./Datasets/lyrics/Albert/test_dataset.pt')
            
                print('Creating dataloaders...')
                _, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

                print("Loading pre-trained model and making predictions on validation set:")
                load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/tf_albert.bin', model_name="albert-base-v2")
                
                print("Making predictions on test set:")
                load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/tf_albert.bin', model_name="albert-base-v2")
                
            else:
                print("Pre-trained model not found. Please train a model first.")
        

        elif option == 10:
            # train_path= './Datasets/lyrics/train.csv' 
            # test_path = './Datasets/lyrics/test.csv'
            cleaned_music_path = './Datasets/lyrics/clean_lyrics.csv'
            reduced_lyrics_path = './Datasets/lyrics/reduced_lyrics.csv'
            train_dataset_path=  './Datasets/lyrics/Albert/train_dataset.pt'
            val_dataset_path= './Datasets/lyrics/Albert/val_dataset.pt'
            test_dataset_path = './Datasets/lyrics/Albert/test_dataset.pt'

            # cleaned_data, cleaned_music_path = prepare_and_clean_dataset(train_path=train_path, 
            #                                                              test_path=test_path,  
            #                                                              cleaned_music_path=cleaned_music_path)
            # print('...Cleaned dataset')
            print('Transfer learning ALBERT on song lyrics...')
            reduced_lyrics_path=proportional_sampling(cleaned_music_path=cleaned_music_path, genre_col='Genre', min_samples_per_genre=200, target_samples=1000, reduced_lyrics_path =reduced_lyrics_path)
            print('...Sampled song lyrics data')

            dataset, unique_genres, num_labels = preprocess_lyrics_data(path='./Datasets/lyrics/reduced_lyrics.csv', tokenizer_name="albert-base-v2", max_length=512)
            print('...Tokenizing dataset')

            train_dataset, val_dataset, test_dataset= split_dataset(dataset, train_frac=0.7, test_frac=0.15, val_frac=0.15, seed=42, 
            train_dataset_path= train_dataset_path, val_dataset_path = val_dataset_path, test_dataset_path = test_dataset_path)
            print('...Splitting dataset')

            train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=16)
            print('...Creating dataloaders')

            print("Training RoBERTa model.")
            training_stats = train_model_tf(train_dataloader, valid_dataloader, num_labels, model_name="albert-base-v2", pre_trained_model_path= 'A/models/albert.bin', model_path= 'B/models/tf_albert.bin', epochs=3, learning_rate=2e-5, epsilon=1e-8)
            
            plot_training_curves(training_stats)
            print("Loading pre-trained model and making predictions on validation set:")
            load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'B/models/tf_albert.bin', model_name="albert-base-v2")
            
            print("Making predictions on test set:")
            load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'B/models/tf_albert.bin', model_name="albert-base-v2")        
    

        elif option == 11:
            sys.exit()

        else:
            print("Invalid option. Please enter 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 or 11")
