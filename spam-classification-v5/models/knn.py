from sklearn.neighbors import KNeighborsClassifier

def train_model(X_train, y_train):
    return KNeighborsClassifier().fit(X_train, y_train)