# higgs-boson-prediction
Machine Learning HIggs Boson challenge for AICrowd

Team Members : 
* Léo Alvarez: 
* Daniel Gonczy: 
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
**`implementations.py`** : contains regression models  
**`train_test.py`** : contains functions for training models and generating output csv file  
**`run.py`** : contains code for generating prediction output CSV on unlabelled testing set   
