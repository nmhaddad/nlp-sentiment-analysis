# Understanding User Comments via Sentiment Analysis

*Includes analysis of a large corpus of positive and negative user comments, data cleaning, model selection, and deployment to a Flask REST API*

- Nathaniel Haddad haddad.na@northeastern.edu
- Northeastern University
- Disclosure: this is a academic project

---

## Install
packages:
- `pip3 install pickle-mixin`
- `pip3 install -U scikit-learn`
- `pip3 install Flask`
- `pip3 install nltk`
- `pip3 install pandas`

run:
- train a model: (from the root folder) `python comment_clf_model.py`
- run the server: (from the root folder) `python comment_clf_app.py`
- go to the server home: `http://127.0.0.1:5000/v1/api`



## Overview

This project represents a series of machine learning models used to identify attacks on users on Wikipedia using natural language processing. Using Scikit-learn and other packages, I built several classifiers that were able to predict whether a comment was an attack or not with a high rate of accuracy.

<br>

<div align="center">
   <img align="center" src="media/home.png">
   <img src="media/result_good.png" width="425"/> <img src="media/result_bad.png" width="425"/> 
</div>

## Notes
- Some of the resources I used:
  - https://www.kdnuggets.com/2019/01/build-api-machine-learning-model-using-flask.html
  - https://arxiv.org/abs/1912.02292



