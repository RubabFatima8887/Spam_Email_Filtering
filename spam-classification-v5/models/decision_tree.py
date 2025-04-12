from sklearn.tree import DecisionTreeClassifier

def train_model(X_train, y_train):
    return DecisionTreeClassifier().fit(X_train, y_train)