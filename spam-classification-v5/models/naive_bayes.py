from sklearn.naive_bayes import MultinomialNB

def train_model(X_train, y_train):
    return MultinomialNB().fit(X_train, y_train)