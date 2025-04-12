from sklearn.tree import DecisionTreeClassifier

def train(X_train, y_train):
    return DecisionTreeClassifier().fit(X_train, y_train)

def inspect_tree(model):
    return {
        'root_feature': model.tree_.feature[0],
        'root_threshold': model.tree_.threshold[0]
    }