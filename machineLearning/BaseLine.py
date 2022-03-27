import pandas as pd

from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def LogisticRegression(cross_validation, X_train, Y_train):
    """
    Class Imbalance is already taken into account of here.
    Logistic Regression with Cross validation
    """
    model  = LogisticRegressionCV(cv=cross_validation ,max_iter = 1000, class_weight= 'balanced', multi_class = 'multinomial')
    model.fit(X_train, Y_train)
    return model

def main():
    ## Change file location accordingly
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    ## Vectorizer 
    vectorizer = TfidfVectorizer(ngram_range=(1,2))

    X_train, X_val, y_train, y_val = train_test_split(train['whichever column'], train['whichever column'], test_size=0.3)

    y_train = train["change this"]

    model = LogisticRegression(10, X_train, y_train)

    ## Prediction
    y_pred = model.predict(X_val)

    score = f1_score(y_val, y_pred, average='macro')

    print("Validation Score is :" + str(score))


if __name__ == "__main__":
    main()
