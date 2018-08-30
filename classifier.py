import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def clean_text(text):
    """

    :param text:
    :return: cleaned text
    """
    text = text.lower()
    text = text.strip(' ')
    return text

"""
Define train data and pass data to make predictions.
train - data to be trained
test - data to check (but file to classify can be used as well if the classifier is well tuned

"""
train = pd.read_csv('path/to/file.csv', encoding="ISO-8859-1")
test = pd.read_csv('path/to/file.csv', encoding="utf-8")
# Describe shape of data - optional
# print('Total rows in test is {}'.format(len(test)))
# print('Total rows in train is {}'.format(len(train)))

#specify columns to be as a target - dropping index and title, can be changed
cols_target = list(train)[2:]
data = train[cols_target]

test['title'] = test['title'].map(lambda com: clean_text(com))
train['title'] = train['title'].map(lambda com: clean_text(com))

#create matrices to be vectorized
X = train.title
test_X = test.title

vect = TfidfVectorizer(max_features=5000, stop_words='english')

#fit TF-IDF
X_dtm = vect.fit_transform(X)
test_X_dtm = vect.transform(test_X)

#apply logistic regression
logreg = LogisticRegression(C=12.0)

# create submission file
sample_submission = pd.read_csv('path/to/file.csv', encoding="utf-8")
for label in cols_target:
    print('... Processing {}'.format(label))
    y = train[label]
    # train the model using X_dtm & y
    logreg.fit(X_dtm, y)
    # compute the training accuracy
    y_pred_X = logreg.predict(X_dtm)
    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))
    # compute the predicted probabilities for X_test_dtm
    test_y_prob = logreg.predict_proba(test_X_dtm)[:, 1]
    sample_submission[label] = test_y_prob

#get statistics of the classified data
sample_submission.describe()

#export the data
sample_submission.to_csv('submission_binary.csv', index=False)

