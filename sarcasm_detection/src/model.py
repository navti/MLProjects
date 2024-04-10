import pandas as pd
from time import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def train_and_evaluate_models(X_train_vec, y_train, X_test_vec, y_test, prep_time, models):

    # Dictionaries for storing results
    results = {}

    # Training models and evaluating their performance
    for model_name, model in models.items():
        start_time = time()  # Запуск таймера
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        end_time = time()  # Остановка таймера

        # Saving results taking into account preparation time
        results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'time': (end_time - start_time) + prep_time
        }

    # Convert results to DataFrame for convenience
    results_df = pd.DataFrame(results).T

    # Output results
    print(results_df)

    return results_df

def plot_metrics(results_df):
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))

    # F1 Score
    axs[0, 0].bar(results_df.index, results_df['f1_score'], color='skyblue')
    axs[0, 0].set_title('F1 Score')
    axs[0, 0].set_ylabel('F1 Score')
    axs[0, 0].axhline(y=0.7, color='r', linestyle='--')  # Линия для 70%

    # Precision
    axs[0, 1].bar(results_df.index, results_df['precision'], color='lightgreen')
    axs[0, 1].set_title('Precision')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].axhline(y=0.7, color='r', linestyle='--')

    # Recall
    axs[1, 0].bar(results_df.index, results_df['recall'], color='salmon')
    axs[1, 0].set_title('Recall')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].axhline(y=0.7, color='r', linestyle='--')

    axs[1, 1].bar(results_df.index, results_df['time'], color='gold')
    axs[1, 1].set_title('Time to Train (seconds)')
    axs[1, 1].set_ylabel('Time (s)')

    plt.tight_layout()
    plt.show()

    return

def best_model(results_df, models):
    # Time normalization
    results_df['normalized_time'] = (results_df['time'] - results_df['time'].min()) / (
                results_df['time'].max() - results_df['time'].min())

    # KPI calculation
    results_df['KPI'] = (results_df['accuracy'] +
                         results_df['f1_score'] +
                         results_df['precision'] +
                         results_df['recall'] -
                         results_df['normalized_time'])

    print(results_df)

    # Search for a model with the highest KPI
    best_model_index = results_df['KPI'].idxmax()
    best_model_name = results_df.loc[best_model_index].name
    best_model = models[best_model_name]

    print("")
    print("*" * 100)
    print("best model:")
    print(best_model_name)

    return best_model
