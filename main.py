import sys
from A.menuA import run_task_A
from B.menuB import run_task_B

def main():

    if len(sys.argv) != 2:
        print("Usage: python main.py <task>")
        sys.exit(1)

    task = sys.argv[1]

    if task == 'A':
        run_task_A() 

    elif task == 'B':
        run_task_B() 
        
    else:
        print("Invalid task. Please choose 'A' or 'B'.")
        sys.exit(1)

if __name__ == "__main__":
    main()




