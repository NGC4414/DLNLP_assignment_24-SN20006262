import sys
import os
import torch
from A.data_preprocessing_A import eda_books, load_datasets_from_files, preprocess_data, proportional_sampling_min, split_dataset, create_data_loaders
from A.modelA import plot_training_curves, predict_with_multinomialNB, train_model, load_model_and_predict


def print_menu_A():
    print("##################################")
    print("----------------EDA---------------")
    print('1. EDA - Exploratory Data Analysis')  
    print("-------------Baseline-------------")
    print("2. Naive Bayes classifier")
    print("--------------RoBERTa-------------")
    print("4. RoBERTa (pre-trained) - deprecated")
    print("5. RoBERTa (pre-trained) sampled")
    print("6. RoBERTa (train on your machine) - deprecated")
    print("7. RoBERTa (train on your machine) sampled")
    print("--------------ALBERT-------------")
    print("8. ALBERT (pre-trained) sampled")
    print("9. ALBERT (train on your machine) sampled")
    print("10. Exit program")
    print("##################################")


def run_task_A():     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache() # function to release all the memory that can be freed
    books_file_path = "./Datasets/books/books.csv"
    batch_size = 16

    while True:
        print_menu_A()
        try:
            option = int(input("Enter the option number: "))

        except ValueError:
            print("Invalid input. Please enter a number.")
            continue        

        if option == 1:
            print("----------Exploratory data analysis----------")
    
            eda_books(books_file_path, plt_title1 = 'Genre Distribution (Full Sample)', plt_title2='Summary Length Distribution (Full Sample)', plt_title3 = 'Average Summary Length per Genre (Full Sample)')

            print('EDA where each class is capped at 400 samples (modified)')
            reduced_books_file_path_400m = r"./Datasets/books/reduced_books_400m.csv"
            reduced_books_file_path_400m, reduced_data = proportional_sampling_min(books_file_path, reduced_books_file_path_400m, target_samples=400, minimum_samples=200)
            eda_books(reduced_books_file_path_400m, plt_title1 = 'Genre Distribution (max_samples_per_class=400)', plt_title2 = 'Summary Length Distribution (max_samples_per_class=400)', plt_title3= 'Average Summary Length per Genre (max_samples_per_class=400)')

        
        elif option == 2:
            print("Executing Naive Bayes classifier")
            reduced_books_file_path_400m = r"./Datasets/books/reduced_books_400m.csv"
            predict_with_multinomialNB(reduced_books_file_path_400m)
            

        elif option == 4:
            if os.path.exists('A/models/roberta_f.bin'):
                print("Executing pre-trained Roberta on full dataset")
                #reduced_books_file_path = r"./Datasets/reduced_books.csv"
                dataset, unique_genres, num_labels = preprocess_data(books_file_path, tokenizer_name="roberta-base", max_length=512)
                train_dataset_path_full  = './Datasets/books/Roberta/train_dataset_full.pt'  
                val_dataset_path_full  = './Datasets/books/Roberta/val_dataset_full.pt'
                test_dataset_path_full = './Datasets/books/Roberta/test_dataset_full.pt'

                #dataset, unique_genres, num_labels = preprocess_data(reduced_books_file_path, tokenizer_name="roberta-base", max_length=512)
                #train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42)

                # no need to split because the split was done in the training stage
                train_dataset, val_dataset, test_dataset =  load_datasets_from_files(train_dataset_path=train_dataset_path_full,
                val_dataset_path =val_dataset_path_full ,test_dataset_path =test_dataset_path_full) 
                
                # no need to split because the split was done in the training stage
                _, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

                print("Loading pre-trained model and making predictions on validation set:")
                load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'A/models/roberta_f.bin', model_name="roberta-base")
                
                print("Making predictions on test set:")
                load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'A/models/roberta_f.bin', model_name="roberta-base")

            else:

                print("Pre-trained model not found. Please train a model first.")


        elif option == 5:
            if os.path.exists('A/models/roberta_sm.bin'):
                print("Executing pre-trained Roberta on sampled dataset")
                reduced_books_file_path_400m = r"./Datasets/books/reduced_books_400m.csv"
                #reduced_books_file_path_400m, reduced_data = proportional_sampling_min(books_file_path, reduced_books_file_path_400m, target_samples=400, minimum_samples=200)
                dataset, unique_genres, num_labels = preprocess_data(reduced_books_file_path_400m, tokenizer_name="roberta-base", max_length=512)

                print('Loading training, validation and test pytorch datasets...')
                train_dataset, val_dataset, test_dataset = load_datasets_from_files(train_dataset_path='./Datasets/books/Roberta/train_dataset.pt', val_dataset_path ='./Datasets/books/Roberta/val_dataset.pt', test_dataset_path ='./Datasets/books/Roberta/test_dataset.pt')

                print('Creating dataloaders...')
                _, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

                print("Loading pre-trained model and making predictions on validation set:")
                load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'A/models/roberta_sm.bin', model_name="roberta-base")
                
                print("Making predictions on test set:")
                load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'A/models/roberta_sm.bin', model_name="roberta-base")

            else:
                print("Pre-trained model not found. Please train a model first.")


        elif option == 6:
            print('Training RoBERTa on full dataset')
            print("This may take a while ...")

            print('...Preprocessing the dataset')
            dataset, unique_genres, num_labels = preprocess_data(books_file_path, tokenizer_name="roberta-base", max_length=512)
            
            print('Splitting the dataset into training, validation and testing sets')
            train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42)
            
            print('Loading training, validation and test pytorch datasets...')
            train_dataset, val_dataset, test_dataset =  load_datasets_from_files(train_dataset_path='./Datasets/books/Roberta/train_dataset.pt',val_dataset_path ='./Datasets/books/Roberta/val_dataset.pt',test_dataset_path ='./Datasets/books/Roberta/test_dataset.pt')
            
            print('Creating dataloaders...')
            train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)
            
            # Train Roberta model on your machine
            print("Training Roberta model...")
            model_path='A/models/roberta_f.bin'
            training_stats = train_model(train_dataloader, valid_dataloader, num_labels, model_path, epochs=3, learning_rate=2e-5, epsilon=1e-8)
            plot_training_curves(training_stats)
            
            print("Loading pre-trained model and making predictions on validation set:")
            load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'A/models/roberta_f.bin', model_name="roberta-base")
            
            print("Making predictions on test set:")
            load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'A/models/roberta_f.bin', model_name="roberta-base")


        elif option == 7:
            print('Training RoBERTa on sampled dataset')
            print("This may take a while ...")

            print('...Sampling the dataset')
            reduced_books_file_path_400m, reduced_data = proportional_sampling_min(books_file_path, reduced_books_file_path = "./Datasets/books/reduced_books_400m.csv", target_samples=400)
            
            print('...Preprocessing the dataset')
            dataset, unique_genres, num_labels = preprocess_data(reduced_books_file_path_400m, tokenizer_name="roberta-base", max_length=512)
            
            print('Splitting the dataset into training, validation and testing sets')
            train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42, train_dataset_path= './Datasets/books/Roberta/train_dataset.pt', val_dataset_path ='./Datasets/books/Roberta/val_dataset.pt', test_dataset_path='./Datasets/books/Roberta/test_dataset.pt')
            
            print('Loading training, validation and test pytorch datasets...')
            train_dataset, val_dataset, test_dataset = load_datasets_from_files(train_dataset_path='./Datasets/books/Roberta/train_dataset.pt',val_dataset_path ='./Datasets/books/Roberta/val_dataset.pt',test_dataset_path ='./Datasets/books/Roberta/test_dataset.pt')
            
            print('Creating dataloaders...')
            train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)
            
            print("Training RoBERTa model.")
            training_stats = train_model(train_dataloader, valid_dataloader, num_labels, model_name= "roberta-base", model_path  = 'A/models/roberta_sm.bin', epochs=3, learning_rate=2e-5, epsilon=1e-8)

            plot_training_curves(training_stats)
            print("Loading pre-trained model and making predictions on validation set:")
            load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'A/models/roberta_sm.bin', model_name="roberta-base")
            
            print("Making predictions on test set:")
            load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'A/models/roberta_sm.bin', model_name="roberta-base")        


    #####################################################################################################################################################################################################
        elif option == 8:
            if os.path.exists('A/models/albert.bin'):
                print("Executing pre-trained ALBERT on sampled dataset")
                
                reduced_books_file_path_400m = r"./Datasets/books/reduced_books_400m.csv"
                #reduced_books_file_path_400m, reduced_data = proportional_sampling_min(books_file_path, reduced_books_file_path_400m, target_samples=400, minimum_samples=200)

                dataset, unique_genres, num_labels = preprocess_data(reduced_books_file_path_400m, tokenizer_name="albert-base-v2", max_length=512)

                print('Loading training, validation and test pytorch datasets...')
                train_dataset, val_dataset, test_dataset = load_datasets_from_files(train_dataset_path='./Datasets/books/Albert/train_dataset.pt',val_dataset_path ='./Datasets/books/Albert/val_dataset.pt',test_dataset_path ='./Datasets/books/Albert/test_dataset.pt')

                print('Creating dataloaders...')
                _, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

                print("Loading pre-trained model and making predictions on validation set:")
                load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'A/models/albert.bin', model_name="albert-base-v2")
                
                print("Making predictions on test set:")
                load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'A/models/albert.bin', model_name="albert-base-v2")

            else:
                print("Pre-trained model not found. Please train a model first.")

        
        elif option == 9:
            print('Training ALBERT on sampled dataset')
            print("This may take a while ...")

            print('...Sampling the dataset')
            reduced_books_file_path_400m, reduced_data = proportional_sampling_min(books_file_path, reduced_books_file_path = "./Datasets/books/reduced_books_400m.csv", target_samples=400)
            
            print('...Preprocessing the dataset')
            dataset, unique_genres, num_labels = preprocess_data(reduced_books_file_path_400m, tokenizer_name="albert-base-v2", max_length=512)
            
            print('Splitting the dataset into training, validation and testing sets')
            train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_frac=0.8, test_frac=0.1, val_frac=0.1, seed=42, train_dataset_path= './Datasets/books/Albert/train_dataset.pt', val_dataset_path ='./Datasets/books/Albert/val_dataset.pt', test_dataset_path='./Datasets/books/Albert/test_dataset.pt')
            
            print('Loading training, validation and test pytorch datasets...')
            train_dataset, val_dataset, test_dataset =  load_datasets_from_files(train_dataset_path='./Datasets/books/Albert/train_dataset.pt',val_dataset_path ='./Datasets/books/Albert/val_dataset.pt',test_dataset_path ='./Datasets/books/Albert/test_dataset.pt')
            
            print('Creating dataloaders...')
            train_dataloader, valid_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)
            
            # Train BERT model on your machine
            print("Training ALBERT model.")
            #model_path = 'A/models/roberta_sm.bin'
            training_stats = train_model(train_dataloader, valid_dataloader, num_labels, model_name= "albert-base-v2", model_path  = 'A/models/albert.bin', epochs=3, learning_rate=2e-5, epsilon=1e-8)
            #training_stats = train_model(train_dataloader, valid_dataloader, num_labels, model_path, epochs=3, learning_rate=2e-5, epsilon=1e-8)
            
            plot_training_curves(training_stats)
            print("Loading pre-trained model and making predictions on validation set:")
            load_model_and_predict(valid_dataloader, device, num_labels, unique_genres, phase="Validation", model_path = 'A/models/albert.bin', model_name="albert-base-v2")
            
            print("Making predictions on test set:")
            load_model_and_predict(test_dataloader, device, num_labels, unique_genres, phase="Test", model_path= 'A/models/albert.bin', model_name="albert-base-v2")
        

        elif option == 10:
            sys.exit()

        else:
            print("Invalid option. Please enter 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10")


