import sys
import os


def print_menu_B():
    print("##################################")
    print("----------------EDA---------------")
    print('1. EDA - Exploratory Data Analysis')  
    print("-------------Baseline-------------")
    print("2. Baseline Logistic Regression")
    print("3. Naive Bayes classifier")
    print("--------------RoBERTa-------------")
    print("4. RoBERTa (pre-trained)")
    print("5. RoBERTa (pre-trained) sampled")
    print("6. RoBERTa (train on your machine)")
    print("7. RoBERTa (train on your machine) sampled")
    print("--------------ALBERT-------------")
    print("8. ALBERT (pre-trained)")
    print("9. ALBERT (train on your machine)")
    print("10. Exit program")
    print("##################################")


lyrics_file_path = "./Datasets/lyrics/music_genre.csv"

def run_task_B():     
    while True:
        print_menu_B()
        try:
            option = int(input("Enter the option number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if option == 1:
            print("Executing Baseline Logistic Regression")
            #preprocess_data(file_path, tokenizer_name="roberta-base", max_length=512)

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
            pass


        elif option == 5:
            pass

        elif option == 6:
            sys.exit()

        else:
            print("Invalid option. Please enter 1, 2, 3, 4, 5 or 6")
