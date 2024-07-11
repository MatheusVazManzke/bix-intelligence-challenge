A note on the selection process' questionnaire:

Apart from this repository, I also had to answer over 10 questions on a questionnaire. Many of these questions required me to analyze specific columns while disregarding NaN values. I could only find the expected answers by droping all NaN values across the entire dataset using .dropna(), but this approach removes any row containing a NaN value. With one column having over 70% NaN values, while others average between 1.5% and 4%, this method drastically reduced our dataframe from 60,000 to just 570 rows. Instead, by handling NaN values only in the columns of interest, we obtained more meaningful results that better reflected the actual data.


=======
# bix-challenge

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

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

Challenge Activities
To solve this problem we want you to answer the following questions:

**1. What steps would you take to solve this problem? Please describe as completely and clearly as possible all the steps that you see as essential for solving the problem.
**   Please, check this notebook for a detailed walkthrough. 

**2. Which technical data science metric would you use to solve this challenge? Ex: absolute error, rmse, etc.
**   My go to metric in this scenario is the F1-score. When a dataset is as imbalanced as the one we are dealing with, base accuracy will always be high if we just assume that every prediction belongs to the negative class. We may also get things wrong if we just look at Precision or Recall. We might get 100% Precision at the cost of too many false negatives. 
   
**3. Which business metric  would you use to solve the challenge?
**   We were given the costs of true positives, false positives and false negatives in the problem's description. If we believe these are truly the only costs, we can create a 'Total Cost' function and optimize it. 

**4. How do technical metrics relate to the business metrics?
**   Total Cost is a function of the business metrics which in turn are a function of the confusion matrix of our model. Whenever the number of false negatives increase, our F1-score may decrease (depending on the other variables) and this will affect our Total Cost function.
   
**5. What types of analyzes would you like to perform on the customer database?
**    
**6. What techniques would you use to reduce the dimensionality of the problem?
**    
**7. What techniques would you use to select variables for your predictive model?
**   Minimum Redundancy Maximum Relevance (mRMR), Permutation Feature Importance. Since we want our variables to be interpretable, techniques like Principal Component Analysis won't help us here.

**8. What predictive models would you use or test for this problem? Please indicate at least 3.
**   I tested many models with their default parameters (except I specified we are dealing with unbalanced data). The better results for these baseline models were CatBoosClassifier, LightGBMClassifier and scikit-learn's NeuralNetwork. Given that we are dealing with a time series, once we had the appropriate time index, I would like to try using https://www.nixtla.io/open-source. (I read it it great, but I've never actually used it)
   
**9. How would you rate which of the trained models is the best?
**    It depends on our priorities. If we value predictive performance foremost, we would choose the model with the best metrics (which we could establish after some rigorous statistical testing). Of course, different models have different computational requisites and some are more complex than others. We should also think about how interpretable a model will be.

**10. How would you explain the result of your model? Is it possible to know which variables are most important?
**    Permutation Feature Importance can tell us which features are contributing the most to model generalization. SHAPLey values will tell us which features are being used for individual predictions. We should be aware that these methods are not directly describing real world relationships. If the model is accurate enough, though, we might tell our clients with a certain level of confidence that certain features are being more decisive than others in predicting the outcome. 

**11. How would you assess the financial impact of the proposed model?
**    We can compare out Total Cost function with a base maintenance cost (There is no model; no truck is repaired before it breaks down).

**12. What techniques would you use to perform the hyperparameter optimization of the chosen model?
**    In my current state of knowledge, Optuna is the state-of-the-art tool for hyperparameter optimization. Please check the notebook for details.

**13. What risks or precautions would you present to the customer before putting this model into production?
**    All models are probabilistic, this means that there will be wrong predictions necessarely. But we are only puting it into production because we believe that we will be able to keep wrong predictions to a manageable miniming that still will, on average, reduce their costs dramatically. A warning, thoough: Data may change and this will have an impact on our model. That's why we will be monitoring the state of their data and retraining the model as needed.

**14. If your predictive model is approved, how would you put it into production?
**    Since I seriously want this job, I will be fully honest: I would have to be taught that by someone who is better than me at model deployment. I do have an understanding of how cloud computing works, but I wouldn't be able to seriously put an important project into production by myself. I do think I could learn quickly with the right guidance, though. But, for starter, I would make sure that my code can reliably ingest and transform our client's data so that it can be used for (re)training and prediction. 
    
**16. If the model is in production, how would you monitor it?
**    We could continously be testing it for data drift, heteroskedacity, new trend and seasonality patterns.
        
**18. If the model is in production, how would you know when to retrain it?
**    I would refer to previous policy. It depends on the cost of retraining. Even if it is not expensive, we don't want to be retraining it daily, as that could add too much uncertainty. We should probably have a performance threshold where if our model starts to p
