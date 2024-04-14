from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from load_data import load_data
from preprocess import clean_data, prepare_data, data_split, vector
from model import train_and_evaluate_models, best_model
import pickle
import pathlib


if __name__ == "__main__":
    file_name = 'Sarcasm_Headlines_Dataset_v2.json'

    # load data
    df = load_data(file_name)

    # clean data
    df = clean_data(df)
    print(df)

    X_train, X_test, y_train, y_test = data_split (df)

    # vectorize
    X_train_vec, X_test_vec = vector(X_train, X_test)

    # using different models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }

    # train model
    results_df = train_and_evaluate_models(X_train_vec, y_train, X_test_vec, y_test, models)


    # find the best model
    best_model_instance = best_model(results_df, models)


    # save the best model
    model_dir = '../models'
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_model_path = f"{model_dir}/model.pkl"
    with open(save_model_path, "wb") as f:
        pickle.dump(best_model_instance, f)