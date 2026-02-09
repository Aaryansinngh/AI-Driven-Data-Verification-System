from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def isolation_forest_model(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(data)
    return predictions

def one_class_svm_model(data):
    model = OneClassSVM(kernel="rbf", gamma="auto")
    predictions = model.fit_predict(data)
    return predictions
