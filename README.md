Methods to Improve the Sensitivity of Wheat Genomic Prediction Models
=====================================================================

Below is a summary of my full midyear Scholar's Electives report detailing the project's progress up to January 2025. Read the full report [here](https://drive.google.com/file/d/1x_vt751px_VtHs26WwQTNsCm-EvxrKQ4/view?usp=sharing).

Overview
--------

As the demand for sustainable food production grows, researchers are looking for better ways to identify high-yield wheat varieties faster. Genomic prediction (GP) models use machine learning (ML) to predict a plant's performance based on its genetic makeup, which helps breeders focus on the most promising lines without having to grow and test every single one. However, these models often struggle with imbalanced datasets, which can make them miss some of the best potential crops.

Project Objectives
------------------

We're working on making these models more effective by:

-   **Using More Advanced ML Models:** Testing Support Vector Machines (SVM) and Extreme Gradient Boosting (XGB) to better capture complex genetic interactions.

-   **Tweaking Classification Thresholds:** Adjusting the cutoffs that define which wheat lines are "top performers" to improve model sensitivity.

-   **Comparing Different Approaches:** Evaluating three different model reformulations---Models R, B, and RO---to see which one balances accuracy and sensitivity the best.

I'm working under professor Mike Domaratzki of Western's CS Department.

Methods
-------

-   **Datasets:** We're using five elite yield trial (EYT) datasets from the International Maize and Wheat Improvement Center (CIMMYT).

-   **Model Variants:**

    -   **Model R:** A regression-based GP model using SVM and XGB.

    -   **Model B:** Reformulates GP as a classification problem.

    -   **Model RO:** Uses a combination of regression and classification with extra tuning after training.

-   **Training Process:**

    -   5-fold cross-validation with grid search for hyperparameter tuning.

    -   Performance measured using Pearson correlation, F1 score, sensitivity, and specificity.

Key Findings
------------

-   SVM generally outperformed XGBoost in sensitivity and F1 scores for Models R and B.

-   Lower classification thresholds helped balance sensitivity and specificity.

-   Optimizing classification thresholds after training made a big difference, especially for the GBLUP-based Models B and RO.

-   Future work could explore hybrid models that combine GBLUP with threshold tuning to push accuracy even further.

How to Use
----------

1.  Clone the repository.

2.  Open the Jupyter notebook.

3.  Run all the cells to train the models and check the results.

Next Steps
----------

Next, we're looking into **SMOTE (Synthetic Minority Over-sampling Technique)** and **SMOTR (Synthetic Minority Over-sampling Technique for Regression)** to help balance the dataset by generating synthetic examples. This should help improve sensitivity even more and ensure that the models don't overlook promising wheat lines.

References
----------

-   CIMMYT. (2018). CIMMYT Global Wheat Breeding Program EYT Dataset [Dataset]. [Link](https://data.cimmyt.org/dataset.xhtml?persistentId=hdl:11529/10548140)

-   Fernandez, A., Garcia, S., Herrera, F., & Chawla, N. V. (2018). SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary. *Journal of Artificial Intelligence Research, 61*, 863--905. [DOI](https://doi.org/10.1613/jair.1.11192)

-   Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (n.d.). *A Practical Guide to Support Vector Classification*.

-   Montesinos-López, O. A., Kismiantini, & Montesinos-López, A. (2023). Two simple methods to improve the accuracy of the genomic selection methodology. *BMC Genomics, 24(1),* 220\. [DOI](https://doi.org/10.1186/s12864-023-09294-5)

-   Uddin, S., Khan, A., Hossain, M. E., & Moni, M. A. (2019). Comparing different supervised machine learning algorithms for disease prediction. *BMC Medical Informatics and Decision Making, 19(1),* 281\. [DOI](https://doi.org/10.1186/s12911-019-1004-8)