# COMP3314-Group-28

This repository contains the implementation and reproduction studies for Twin Support Vector Machines (TWSVM), organized by Group 28 for COMP3314.

The repository includes a StandardSVM.py implementation which serves as a baseline for performance comparison against the Twin SVM logic described in the associated research.

## Paper Details

This project is based on the following research paper:

Title: Twin Support Vector Machines for Pattern Classification

Authors: Jayadeva, R. Khemchandani, and Suresh Chandra

Venue: IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 5 (May 2007)

## Abstract Summary

The paper proposes Twin SVM (TWSVM), a binary classifier that determines two nonparallel planes by solving two related SVM-type problems. Unlike conventional SVMs, which solve a single large Quadratic Programming Problem (QPP), TWSVM solves two smaller QPPs.

Key advantages highlighted in the paper:

Speed: TWSVM is approximately 4 times faster than standard SVMs.

Generalization: It shows good generalization on benchmark datasets.

Structure: It generates two nonparallel planes such that each plane is closer to one of the two classes and as far as possible from the other.

##  How to Set Up and Run

### Prerequisites

Ensure you have Python 3.x installed. You will likely need the following standard machine learning libraries:

1. numpy

2. scikit-learn (for standard SVM comparison)

3. matplotlib (for visualization)

4. pandas (for processing the dataframe)

5. ucimlrepo (for downloading the dataset from the UCL Repository)

6. scipy (to calculate the generalized eigenvalue problem (GEP) in GEPSVM)

### Installation

1. Clone the repository:

git clone https://github.com/KoAung02/COMP3314-Group-28.git

cd COMP3314-Group-28

2. Install dependencies:

pip install numpy pandas scikit-learn matplotlib ucimlrepo scipy

### Running the Code

To run the Standard SVM baseline implementation:

python StandardSVM.py

To run the GEPSVM model implementation:

python GEPSVM.py

## Contributors 
1. Aung Phone Pyae 
2. Chung Kam Hong



