from sklearn.neighbors import KNeighborsClassifier

def train(X_train, y_train):
    return KNeighborsClassifier().fit(X_train, y_train)