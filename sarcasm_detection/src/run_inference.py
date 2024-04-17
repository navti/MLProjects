import pickle
from preprocess import *

"""
Run inference using the best model stored in model directory
"""
# load model and vectorizer
model_dir = '../models'
load_model_path = f"{model_dir}/model.pkl"
with open(load_model_path, "rb") as f:
    loaded_model = pickle.load(f)

vectorizer_path = f"{model_dir}/vectorizer.pkl"
with open(vectorizer_path, "rb") as v:
    vectorizer = pickle.load(v)

while True:
    prompt = input('\nEnter text to check for sarcasm or ctrl+c to exit.\n')
    clean_text = preprocess_text(prompt)
    vec = vectorizer.transform([clean_text])
    is_sarcasm = loaded_model.predict(vec).item()
    print("Sarcasm!\n" if is_sarcasm else "Looks like you are serious.\n")