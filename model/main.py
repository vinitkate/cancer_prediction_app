import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def get_clean_data():
    data = pd.read_csv("dataset/data.csv")
    # print(data.head())
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    
    return data
    
def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    scalar = StandardScaler()
    
    X = scalar.fit_transform(X)
    # splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    #train
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    #test
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_pred, y_test))
    print("Classification report:", classification_report(y_test, y_pred))
    
    return model, scalar

    
def main():
    data = get_clean_data()
    print(data.head())
    
    model, scalar = create_model(data)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scalar.pkl', 'wb') as f:
        pickle.dump(scalar, f)


if __name__ == '__main__':
    main()