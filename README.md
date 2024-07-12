Hey there! If you're reading this message, I'm currently putting the finishing touches on this repository. I just want to make sure that I won't miss any deadlines for sharing it.

## Table of Contents

1. [About This Repository](#about-this-repository)
2. [Note on the Selection Process Questionnaire](#note-on-the-selection-process-questionnaire)
3. [Project Structure](#project-structure)
4. [Answers to the Challenge Questions](#answers-to-the-challenge-questions)


=======
## About This Repository
I have written the entire modeling process in a single, heavily annotated [notebook](https://github.com/MatheusVazManzke/bix-intelligence-challenge/blob/main/notebooks/exploration/0.0-mvm-data-exploration.ipynb). This notebook is intended to showcase my current skills as a Data Scientist as I develop a quick proof-of-concept model for immediate demonstration. The steps reflect my thought process as I tackle the given problem. You will find the scripts for the data pipeline and model prediction [here](https://github.com/MatheusVazManzke/bix-intelligence-challenge/tree/main/bix-challenge). The answers to the 16 challenge questions are provided at the end of this README file.


## Note on the Selection Process Questionnaire

Apart from this repository, I also had to answer over 10 questions on a questionnaire. Many of these questions required me to analyze specific columns while disregarding NaN values. I couldn't find any of the expected answers, I only did when droping all NaN values across the entire dataset using .dropna(), but this approach removes any row containing a NaN value. With one column having over 70% NaN values, while others average between 1.5% and 4%, this method drastically reduced our dataframe from 60,000 to just 570 rows. I don't know if this is intended.
![question](https://github.com/MatheusVazManzke/bix-intelligence-challenge/blob/main/reports/figures/bix-sample-question.png)
![answer](https://github.com/MatheusVazManzke/bix-intelligence-challenge/blob/main/reports/figures/bix-question-answer.png)

=======

## Project Structure

I use [cookiecutter](https://github.com/drivendataorg/cookiecutter-data-science) here. You will find a description of how this project is structured bellow. I decided to keep everything so that you can have a full view of how my projects are usually structured.


```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for bix-challenge
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── bix-challenge                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes bix-challenge a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```

--------

## Answers to the Challenge Questions

To solve this problem we want you to answer the following questions:

**1. What steps would you take to solve this problem? Please describe as completely and clearly as possible all the steps that you see as essential for solving the problem.**
    
Please, refer to [this notebook](https://github.com/MatheusVazManzke/bix-intelligence-challenge/blob/main/notebooks/exploration/0.0-mvm-data-exploration.ipynb) for this answer to      this question. I will lay out the main steps below:
       - Exploratory Analysis / Data Transformation
            - Import Data
            - Check if the underlying probabilistic distributions are the same for both files
            - Splitting the data
            - Understand the nature of NaN's.
            - Checking for categorical variables
            - Measuring the kurtosis of the distributions
            - Imputing and droping NaN's 
       - Modeling
            - Comparing the performance of different models on our train set
            - Choosing a baseline model for futher optimization (CatBoostClassifier)
            - Feature selection
            - Hyperparameter tuning with Optuna
            - Model calibration
            - SHAP values
            - Test set. 
        - Data pipeline
            - Serialize our final model
            - Create data transformation classes and functions
            - Data transformation scripts
            - Test the scripts
            
**2. Which technical data science metric would you use to solve this challenge? Ex: absolute error, rmse, etc.**
    My go-to metric in this scenario is the F1-score. When a dataset is as imbalanced as the one we're dealing with, base accuracy will always be high if we assume every prediction belongs to the negative class. We might also get misleading results if we only look at Precision or Recall. We could achieve 100% Precision at the cost of too many false negatives.
   
**3. Which business metric would you use to solve the challenge?** 
    We were given the costs of true positives, false positives, and false negatives in the problem description. If we believe these are truly the only costs, we can create a 'Total Cost' function and minimize it with Optuna.

**4. How do technical metrics relate to the business metrics?** 
    Total Cost is a function of the business metrics which in turn are a function of the confusion matrix of our model. Whenever the number of false negatives increase, our F1-score may decrease (depending on the other variables) and this will affect our Total Cost function.
   
**5. What types of analyzes would you like to perform on the customer database?**    
    I'd like to ensure our sampling is of good quality, representative, and complete. I noticed that the NaNs don't seem to be randomly distributed, so I'd like to investigate this further. I would also check for the temporal characteristics of the data in the database.
    
**6. What techniques would you use to reduce the dimensionality of the problem?** 
    The most common dimensionality reduction techniques, like PCA, create a lower-dimensional representation of our data but make the model harder to interpret, as the new dimensions won't have the original meaning of our variables. So, given the task at hand, we will focus on feature selection (see below).
    
**7. What techniques would you use to select variables for your predictive model?**
    Minimum Redundancy Maximum Relevance (mRMR) and/or Permutation Feature Importance. 
    
**8. What predictive models would you use or test for this problem? Please indicate at least 3.**
    I tested several models with their default parameters (specifying that we are dealing with unbalanced data). The best results for these baseline models were CatBoostClassifier, LightGBMClassifier, and scikit-learn's NeuralNetwork. Given that we are dealing with a time series, once we have the appropriate time index, I would like to try using https://www.nixtla.io/open-source.
    
**9. How would you rate which of the trained models is the best?**
    It depends on our priorities. If we value predictive performance foremost, we would choose the model with the best metrics, established after rigorous statistical testing. Of course, different models have different computational requirements, and some are more complex than others. We should also consider the model's interpretability.

**10. How would you explain the result of your model? Is it possible to know which variables are most important?** 
    Permutation Feature Importance can tell us which features contribute most to model generalization. SHAP values will show which features are used for individual predictions. We should be aware that these methods do not directly describe real-world relationships. If the model is accurate enough, though, we can tell our clients with a certain level of confidence that certain features are more decisive than others in predicting the outcome. If we are really serious about establishing causality, causal models could me employed.

**11. How would you assess the financial impact of the proposed model?**
   We can compare our Total Cost function with a base maintenance cost (i.e., no model; no truck is repaired before it breaks down).

**12. What techniques would you use to perform the hyperparameter optimization of the chosen model?** 
    In my current knowledge, Optuna is the state-of-the-art tool for hyperparameter optimization. Please check the notebook for details.
    
**13. What risks or precautions would you present to the customer before putting this model into production?**
    All models are probabilistic, meaning there will necessarily be wrong predictions. But we are putting it into production because we believe we can keep wrong predictions to a manageable minimum that will, on average, dramatically reduce their costs. A warning, though: Data may change, impacting our model. That's why we will monitor the state of their data and retrain the model as needed.

**14. If your predictive model is approved, how would you put it into production?**
    Since I seriously want this job, I will be fully honest: I would need to be taught by someone more experienced in model deployment. I have an understanding of cloud computing, but I wouldn't be able to put an important project into production by myself. I do think I could learn quickly with the right guidance. To start, I would ensure my code can reliably ingest and transform our client's data for (re)training and prediction. 
    
**16. If the model is in production, how would you monitor it?**
    We could continuously test it for data drift, heteroscedasticity, and new trends and seasonality patterns.      
    
**18. If the model is in production, how would you know when to retrain it?** 
    I would refer to previous policy. It also depends on the cost of retraining. Even if it is not expensive, we don't want to be retraining it all the time, as that could add too much uncertainty. Probably the most sensible thing to do is to retrain it whenever we identify that there is significant data drift. If we get a sequence of bad results that are highly improbable even if we take into consideration the natural variabiltiy in a model's performance, we shoudl also rethink and retrain it.
