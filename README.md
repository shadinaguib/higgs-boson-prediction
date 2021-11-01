# higgs-boson-prediction
Machine Learning HIggs Boson challenge for AICrowd

Team Members : 
* Léo Alvarez: leo.alvarez@epfl.ch  
* Daniel Gonczy: daniel.gonczy@epfl.ch
* Shadi Naguib: shadi.naguib@epfl.ch

# Environment
In order to run the code, the folder structure should be as follows: 

    .
    ├── Data                    # Data files, in .csv
        ├── train.csv
        └── test.csv
    ├── src                     # Source files
    └── README.md


# Code structure
The src folder contains several scritps and notebooks : 

**`costs.py`** : contains helper functions for regression models using gradient descent  
**`tools.py`** : contains helper functions for polynomial regression and cross validation  
**`crossvalidation.py`** : contains helper functions for cross validation  
**`implementations.py`** : contains definitions of regression models  
**`train_test.py`** : contains functions for training models using the different regression models  
**`run.py`** : contains code for generating prediction output CSV on unlabelled testing set   
**`notebook.ipynb`** : contains explatorary analysis of dataset and plots that are found in the report

# To generate prediction data

1. Clone the repo
2. Download the training and testing data and put it in the right folder 
3. Run `python3 run.py` in the home directory
