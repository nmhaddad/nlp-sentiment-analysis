from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.externals import joblib
import pandas as pd
import urllib

def get_metrics(clf, test_comments) -> None:
    """
    function: get_metrics
    params: clf, a function
    returns: nothing
    does: prints out confusion matrix, precision, recall, f-score, and ROC AUC
    """
    y_pred = clf.predict(test_comments['comment'])
    auc = roc_auc_score(test_comments['attack'], clf.predict_proba(
      test_comments['comment'])[:,1])
    print('Test ROC AUC: %.5f' %auc)

def download_file(url: str, fname: str) -> None:
    """
    function: download_file
    params: url, a string; fname, a string
    returns: nothing
    does: downloads a given file to the current directory
    """
    urllib.request.urlretrieve(url, fname)

def main():
    # download annotated comments and annotations
    ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634'
    ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637'

    download_file(ANNOTATED_COMMENTS_URL, 'attack_annotated_comments.tsv')
    download_file(ANNOTATIONS_URL, 'attack_annotations.tsv')

    comments = pd.read_csv('attack_annotated_comments.tsv', sep='\t',
                           index_col=0)
    annotations = pd.read_csv('attack_annotations.tsv', sep='\t')

    # labels a comment as an atack if the majority of annoatators did so
    labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

    # join labels and comments
    comments['attack'] = labels

    # remove newline and tab tokens
    comments['comment'] = comments['comment']\
        .apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    comments['comment'] = comments['comment']\
        .apply(lambda x: x.replace("TAB_TOKEN", " "))

    train_comments = comments.query("split=='train'")
    val_comments = comments.query("split=='dev'")
    test_comments = comments.query("split=='test'")
    train_comments = pd.concat([val_comments, train_comments])

    # create feature union of character and word TFIDF vectorizers
    vectorizerW = TfidfVectorizer(lowercase=True,
                                  analyzer='word',
                                  stop_words=None,
                                  ngram_range=(1, 1),
                                  max_df=1.0,
                                  min_df=1,
                                  max_features=None,
                                  norm='l2')
    vectorizerC = TfidfVectorizer(lowercase=True,
                                  analyzer='char',
                                  stop_words=None,
                                  ngram_range=(1, 1),
                                  max_df=1.0,
                                  min_df=1,
                                  max_features=None,
                                  norm='l2')

    combined_features = FeatureUnion([('word', vectorizerW), ('char', vectorizerC)])

    # create and train a logistic regression with cross validation
    clf = Pipeline([
        ('features', combined_features),
        ('clf', LogisticRegressionCV(cv=3, max_iter=1000, solver='lbfgs'))
    ])
    clf = clf.fit(train_comments['comment'], train_comments['attack'])
    get_metrics(clf, test_comments)

    # save the model
    joblib.dump(clf, 'models/trained_model.pkl')
    print('Model Saved')

main()

