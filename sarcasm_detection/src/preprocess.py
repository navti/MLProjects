import re
import nltk
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pathlib

# Initialize lemmatizer and get stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

def clean_data(df):
    """
    clean headlines data set
    :param df: dataframe containing the headlines and labels
    :return: cleaned dataframe
    """
    # Explore data
    df.info()
    df.head()
    # Remove duplicates
    df.drop_duplicates(ignore_index=True, inplace=True)
    # Drop the column with the link, because it has no weight
    df = df.drop(columns='article_link')
    # View several headers for the example
    list(df.head(10)['headline'])
    # Check for class imbalance
    sns.barplot(df.groupby('is_sarcastic').agg('count')['headline']);
    # Header lengths
    df.headline.str.len().describe()
    # Calculate row lengths
    df['length'] = df['headline'].str.len()
    # Calculate quartiles and IQR
    Q1 = df['length'].quantile(0.25)
    Q3 = df['length'].quantile(0.75)
    IQR = Q3 - Q1
    # Determine bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers
    df_cleaned = df[(df['length'] >= lower_bound) & (df['length'] <= upper_bound)]
    # Header lengths
    df_cleaned.headline.str.len().describe()
    # Remove column with row length
    df_cleaned = df_cleaned.drop(columns='length')

    return df_cleaned

def preprocess_text(text):
    """
    preprocess text before feeding to model
    :param text: type str
    :return: type str, after preprocessing
    """
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Keep only letters and spaces
    # Tokenization
    tokens = word_tokenize(text.lower()) # Convert to lowercase
    # Remove stop words and lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(cleaned_tokens)

def prepare_data(df):
    """
    prepare data in dataframe
    :param df: type pandas dataframe
    :return: type pandas dataframe after data processing
    """
    df['headline'] = df['headline'].apply(preprocess_text)
    return df

def data_split (df):
    """
    split data into training and testing sets
    :param df: type pandas dataframe
    :return: 
        X_train: train set features
        X_test: test set features
        y_train: train set labels
        y_test: test set labels
    """
    X_train, X_test, y_train, y_test = train_test_split(df['headline'], df['is_sarcastic'], test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def vector(X_train, X_test):
    """
    vectorize features
    :param X_train: input training features
    :param X_test: input test features
    :return: vectorized train and test features
    """
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model_dir = '../models'
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    vectorizer_path = f"{model_dir}/vectorizer.pkl"
    with open(vectorizer_path, "wb") as vectorizer_file:
        pickle.dump(vectorizer_path, vectorizer_file)
    return X_train_vec, X_test_vec