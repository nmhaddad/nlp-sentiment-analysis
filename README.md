# Machine Learning Sentiment Analysis

## Wikipedia Attack Classifier REST API

- Nathaniel Haddad nhaddad2112@gmail.com

## Install
packages:
- `pip install pickle-mixin`
- `pip install -U scikit-learn`
- `pip install Flask`

run:
- train a model: (from the root folder) `python comment_clf_model.py`
- run the server: (from the root folder) `python comment_clf_app.py`
- go to the server home: `http://127.0.0.1:5000/v1/api`

## Notes
- Some of the resources I used:
  - https://www.kdnuggets.com/2019/01/build-api-machine-learning-model-using-flask.html
  - https://arxiv.org/abs/1912.02292

## Overview:
This project represents a series of machine learning models used to identify attacks on users on Wikipedia using natural language processing. Using Scikit-learn and other packages, I built several classifiers that were able to predict whether a comment was an attack or not with a high rate of accuracy.

![alt text](media/home.png "Home page")

## Methodologies:
**Data Cleaning**

One of the first text cleaning methods I tried was removing the stopwords from the comment column in the dataset. Stopwords are usually the most common words in a language, which can affect the meaning of phrases in a document. I used a package called the Natural Language Toolkit (nltk) that contains lists of stopwords. Using the imported list of stopwords, I removed them from each instance in comment using the Pandas method apply. Overall, I decided not to include my remove_stopwords function in my final code because it did not have a positive impact on my ROC AUC score for my models. I also used term frequency-inverse document frequency vectorizers (tfidf) later on in my experimentation, which removes stopwords when text mining - I will detail this later on in my findings.

Another text cleaning method I tried was to correct the spelling of words in the comment attribute of the dataset. In my research, I learned that spelling can sometimes affect text mining. Sometimes, this effect is positive, in that misspelled words are spelled correctly. Other times, this effect can be negative, as misspelled and correctly spelled words are incorrectly spelled because the words do not exist in the dataset of correctly spelled words. I thought that maybe corrected spelling would 'normalize' so to say, some of the comments. After implementing a spellchecker function with pyspellchecker, I saw no improvement to the ROC AUC score, which surprised me, so I decided not to include the spellchecker in the final code.

While implementing the above to functions, I built a parser method that applies specific transformations to the text as it is applied to each instance. Including the above two functions described, I also made all text lowercase and removed numbers and other nonalphabetical characters.

Again, even with all of these functions used independently, I saw no increase in ROC AUC score. Why? Because the code provided by the strawman model includes a `TfidfTransformer` that automatically lowercases text and removes stopwords.

**Feature Extraction and Construction**

I considered using several different types of feature extraction methods. First, I looked into fine-tuning the given strawman feature extraction methods, `CountVectorizer` and `TfidfTransformer`. Doing so, I saw an improvement in accuracy, albeit small.

Next, I used a `TfidfVectorizer` to combine the `CountVectorizer` and the `TfidfTransformer` above. I also used `GridSearchCV` to fine-tune this feature extraction method.

Finally, I added `FeatureUnion` of two `TfidfVectorizer` methods. I have already detailed this implementation in part a. Ultimately, I settled on using the resulting `combined_features` variable for `FeatureUnion` in my model testing.

I also considered using other features besides comments, such as `year`. `year` produced an ROC AUC of 0.95585, which was slightly worse than the strawman code. I also tried all of the other features in combination to `comment`; `logged_in`, `ns`, and `sample` negatively affected the AUC ROC score. I have since removed these features from the code.

**Optimizations**

One of the most important optimizations was combining training and validation datasets to create a new and bigger training set. Doing so made my training dataset bigger and therefore, gave it more unseen data to train on. I was also able to still create validation sets using built-in Sci-kit learn methods, so it was not a problem to combine the two.

Other optimizations I made include hyper-parameter tuning (discussed below), thoughtful analysis of the dataset to include specific feature extraction techniques (discussed above), and using functions to prevent code replication.

I used several different techniques for hyper-parameter tuning. The most important technique I used was to read the fine manual for all of the models. Reading the documentation was critical in making decisions about which hyper-parameters to choose for each model. For example, gradient descent is great for small datasets, but not for large ones. Here, stochastic gradient descent would be more applicable. Therefore, hyper-parameters should only include stochastic gradient descent, as our dataset is quite large. After handpicking, I used GridSearchCV to tune the hyper-parameters, passing in the values that would have the best effect on the models, as discussed above.

My accuracy increased ~1.5-2.0 percentage points after hyperparameter tuning.

## Results

![alt text](media/result_good.png "Results page")

For my first implementation, I chose to use the existing `LogisticRegression` machine learning model provided by the strawman code. Using the existing `pipeline`, I made alterations to the code: I combined validation and training sets to create a new training set with more instances that could then be used for validation using Sci-kit learn functions. One of the first functions I tried was `GridSearchCV`. Using this function took a very long time to train. With that in mind, I was careful to read the documentation for all of the implemented models to see what parameters really needed to be passed into `GridSearchCV`.

Another model I tested was `LogisticRegression` with a `FeatureUnion` of two `TfidfVectorizers`. This model also improved ROC AUC over the strawman code. Finally, I implemented `LogisticRegressionCV`. This model was even better than the previous two!

For my three new machine learning models, I tested `MLPClassifier`, `MultinomialNaiveBayes`, and a `RandomForestClassifier`.

I used a `get_metrics function` to keep track of all of my metrics for each experiment. As the project required, I implemented a confusion matrix, precision, recall, f-score, and ROC AUC. From these metrics I was able to learn more about the accuracy of my predictions for each of the classes, attack == 1 and attack == 0. I thought it was great to see that some models more accurately predict positive classes than negative classes and vice versa.

![alt text](media/result_bad.png "Results page")

My final result metrics are as follows:

LogisticRegression (strawman):
- Test Precision: 0.93923
- Test Recall: 0.94055
- Test F-Score: : 0.93396
- Test ROC AUC: 0.95697

LogisticRegressionCV 
- Test Precision: 0.94467
- Test Recall: 0.94685
- Test F-Score: : 0.94287
- Test ROC AUC: 0.96361

MLPClassifier 
- Test Precision: 0.94130
- Test Recall: 0.94404
- Test F-Score: : 0.93994
- Test ROC AUC: 0.95322

MultinomialNB
- Test Precision: 0.91163
- Test Recall: 0.90638 
- Test F-Score: : 0.87939
- Test ROC AUC: 0.86398

RandomForestClassifier
- Test Precision: 0.91932
- Test Recall: 0.91509
- Test F-Score: : 0.89451
- Test ROC AUC: 0.94202
