# higgs-boson-prediction
Machine Learning HIggs Boson challenge for AICrowd

The Higgs Boson discovery, announced at the Large Hadron Collider at CERN in March 2013, was the center of a new machine learning challenge. This particle can result from the collision of two accelerated protons. Physicist were able to identify it by measuring the decay signature of a collision event. The challenge was to identify those signals among a noisy background. This report relates and details our methodology and results while trying to recreate the "discovery" of the Higgs Boson using Machine Learning methods, based on real data. By comparing different models and hyperparameters, we achieved $82.9\%$ of accuracy using the Ridge regression method.

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
        ├── helpers_p1.py
        ├── costs.py
        ├── tools.py
        ├── crossvalidation.py
        ├── implementations.py
        ├── train_test.py
        ├── run.py
        ├── project.ipynb
    └── README.md


# Code structure
The src folder contains several scritps and notebooks : 

**`helpers_p1.py`** : contains helper functions for reading and writing csv files  
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
