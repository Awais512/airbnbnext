# Importing Libraries
import os
import re
import string
import joblib
from joblib import dump, load
import pandas as pd
import numpy as np
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


'''This code filter the DataFrame df_train to create two new DataFrames named offensive and not_off based on the value in the label column. Specifically, offensive contains all the rows where the label column is equal to 'OFF', while not_off contains all the rows where the label column is equal to 'NOT'.'''

df_train = pd.read_csv(
    '/content/drive/MyDrive/CE807/Assignment2/2205000/train.csv')
offensive = df_train[df_train['label'] == 'OFF']
not_off = df_train[df_train['label'] == 'NOT']

'''
This code generates a word cloud of the tweets that are labeled as "offensive". The WordCloud() function from the wordcloud package is used to create the word cloud. The width, height, and background_color parameters are set to customize the size and background color of the word cloud. The generate() method is used to generate the word cloud from a string containing all the tweets that are labeled as "offensive". The string is created by joining the tweets together with a space separator using the ' '.join() method. Finally, the word cloud is plotted using the imshow() function from matplotlib and the axis() function is used to turn off the x and y axes. This visualization can help you identify the most frequent words used in offensive tweets and gain insights into the language used in these tweets.
'''
wordcloud = WordCloud(width=800, height=800, background_color='black').generate(
    ' '.join(offensive['tweet']))

plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

'''
This code creates a word cloud visualization of the tweets which are not offensive. The WordCloud function from the wordcloud package is used to generate a word cloud from the text data in the 'tweet' column of the not_off dataframe. The width, height, and background_color parameters are used to set the size and background color of the word cloud. The words in the word cloud are weighted based on their frequency in the text data. Finally, the imshow() function from matplotlib is used to display the word cloud as an image with the specified size, and the axis() function is used to turn off the axis labels. This visualization can help you quickly identify the most frequent words in the non-offensive tweets, providing useful insights into the language and topics used in these tweets.
'''
wordcloud = WordCloud(width=800, height=800, background_color='black').generate(
    ' '.join(not_off['tweet']))

plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

'''
This code is creating a bar chart to visualize the distribution of labels in a dataset. The code uses the seaborn package to create the visualization. The first line of the code sets the color palette to 'dark'. The color_palette() function is used to create a list of colors to be used in the visualization. The 'dark' palette is a list of dark shades that are suitable for creating a high contrast visualization. The next set of lines creates the bar chart itself. The countplot() function from seaborn is used to create the bar chart, with the x parameter set to 'label' to indicate that the labels are to be plotted on the x-axis. The data parameter specifies the dataset to be used, while the palette parameter is set to the colors list that was created earlier. The remaining lines of code are used to customize the appearance of the visualization. The title(), xlabel(), and ylabel() functions are used to add a title and labels to the x and y axes, respectively. Finally, the show() function is called to display the visualization.
'''

colors = sns.color_palette('dark')
plt.figure(figsize=(6, 6))
sns.countplot(x='label', data=df_train, palette=colors)
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

'''
This code is creating a histogram to show the frequency of tweets in a dataset. The plt.hist() function from the matplotlib package is used to plot a histogram of the length of each tweet in the df_train dataset. The bins parameter is set to 50, which means that the x-axis is divided into 50 equal bins. The color parameter is set to blue to make the histogram bars blue in color. The plt.title(), plt.xlabel(), and plt.ylabel() functions are used to add a title and labels to the plot, indicating that it shows the frequency of tweets in terms of the number of characters. Finally, the plt.show() function is called to display the plot. This histogram can provide insights into the distribution of tweet length in the dataset, which can be useful for text analytics tasks such as sentiment analysis or topic modeling.
'''
plt.figure(figsize=(8, 6))
plt.hist(df_train['tweet'].str.len(), bins=50, color='blue')
plt.title('Frequency of Tweets')
plt.xlabel('Number of Characters')
plt.ylabel('Count')
plt.show()

'''
Method 1
Defining a compute performance function which take two arguements y_true and y_pred. It will give us the confusion matrix and accuracy
'''


def compute_performance(y_true, y_pred):

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    # Print performance metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return f1


'''
This code defines a function train_naive_bayes that trains a Naive Bayes classifier on text data and saves the best model. The function takes three arguments: train_file (the path to the training data), val_file (the path to the validation data), and model_dir (the directory where the trained model will be saved). The function reads in the training and validation data using pd.read_csv and separates the input and output data. Then it preprocesses the text data using CountVectorizer which converts the text data into a matrix of token counts. The MultinomialNB algorithm is used to train the Naive Bayes classifier on the transformed text data. After training the model, the function makes predictions on the validation data and calculates evaluation metrics including accuracy, precision, recall, and F1-score. These evaluation metrics are printed to the console. Finally, the best model is saved in the specified directory using pickle.dump.
'''
model_dir = '/content/drive/MyDrive/CE807/Assignment2/2205000/models/1/nb_model.pkl'
output_file_dir = '/content/drive/MyDrive/CE807/Assignment2/2205000/models/1/output_nb/nb_output.csv'


def train_naive_bayes(train_file, val_file, model_dir):

    # Load train and validation data
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Separate input and output data
    X_train = train_df['tweet']
    y_train = train_df['label']
    X_val = val_df['tweet']
    y_val = val_df['label']

    # Preprocess the text data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    # Train the Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Predict on validation data and calculate evaluation metrics
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")

    # Save the best model
    # with open(model_dir, 'wb') as f:
    #     pickle.dump(clf, f)

    with open(model_dir, 'wb') as f:
        pickle.dump(clf, f)

# Here we are calling the functions which we define above.


train_file_path = '/content/drive/MyDrive/CE807/Assignment2/2205000/train.csv'
val_file_path = '/content/drive/MyDrive/CE807/Assignment2/2205000/valid.csv'
train_naive_bayes(train_file_path, val_file_path, f'{model_dir}')

'''
This code defines a function called test_naive_bayes that takes as input a path to a test file, a path to a saved Naive Bayes model, a path to a saved CountVectorizer (if available), and a path to an output file. The function reads the test data from the specified test file, loads the saved Naive Bayes model, and preprocesses the test data using the saved CountVectorizer (if available). Then, it predicts the labels for the test data using the loaded model and calculates the evaluation metrics using a function called compute_performance. The predicted labels and evaluation metrics are then saved to the specified output file (if available). This function is used to test the performance of a Naive Bayes model on new, unseen data.
'''


def test_naive_bayes(test_file, model_file, vectorizer_file=None, output_file=None):

    # Load test data
    test_df = pd.read_csv(test_file)

    # Separate input and output data
    X_test = test_df['tweet']
    y_test = test_df['label']

    # Load saved model
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)

    # Load saved vectorizer or fit a new one on the training data
    if vectorizer_file is not None:
        with open(vectorizer_file, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        vectorizer = CountVectorizer()
        X_train = pd.read_csv(train_file_path)['tweet']
        vectorizer.fit(X_train)

    # Preprocess the test data using the vectorizer
    X_test = vectorizer.transform(X_test)

    # Predict on test data and calculate evaluation metrics
    y_pred = clf.predict(X_test)

    # Print evaluation metrics and confusion matrix
    f1 = compute_performance(y_test, y_pred)

    # Save predicted labels and evaluation metrics to output file
    if output_file is not None:
        output_df = pd.DataFrame({'tweet': test_df['tweet'], 'predicted_labels': y_pred,
                                  'true_labels': y_test,
                                  })
        output_df.to_csv(output_file, index=False)


# Call the function for testing our model
test_naive_bayes(f'/content/drive/MyDrive/CE807/Assignment2/2205000/test.csv',
                 model_dir,
                 output_file=output_file_dir)

out = pd.read_csv(output_file_dir)
print(out[['predicted_labels']].value_counts())
print(out[['true_labels']].value_counts())

'''
This code trains and evaluates a Naive Bayes model on a dataset of tweets with binary classification labels (e.g. OFFENSIVE/NOT OFFENSIVE sentiment). The model is trained on subsets of the training data with varying sizes, and performance metrics (F1 score and accuracy) are computed for each subset. The results are plotted in a graph to visualize the effect of increasing training data size on model performance. The best model (i.e. the one trained on the largest subset of data) is saved to disk, along with its predicted labels on the test set. The saved model can be later used for inference on new data.
'''


def train_and_evaluate_nb(data_path, output_dir):
    """
    Train and evaluate the Naive Bayes model on different data sizes and save the models and results to output_dir.

    Args:
        data_path (str): path to the dataset file
        output_dir (str): path to the output directory

    Returns:
        None
    """
    # Load data
    df = pd.read_csv(data_path)

    # Split data into X and y
    X = df['tweet']
    y = df['label']

    # Split data into train and test sets with test size of 0.2
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize list to store performance results
    results = []

    # Define data sizes to use
    data_sizes = [0.25, 0.5, 0.75, 0.99]

    # Iterate over data sizes
    for size in data_sizes:
        # Split training data into smaller subset
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full, train_size=size, random_state=42)

        # Fit vectorizer on training data
        vectorizer = CountVectorizer()
        vectorizer.fit(X_train)

        # Transform training and testing data using the vectorizer
        X_train = vectorizer.transform(X_train)
        X_test_transformed = vectorizer.transform(X_test)

        # Initialize Naive Bayes classifier
        nb_clf = MultinomialNB()

        # Train the classifier
        nb_clf.fit(X_train, y_train)

        # Predict on test set
        y_pred = nb_clf.predict(X_test_transformed)

        # Compute performance metrics
        f1 = compute_performance(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Add results to list
        results.append({'data_size': size, 'f1': f1, 'accuracy': accuracy})

        # Save model and output to file
        model_file = os.path.join(
            output_dir, str(int(size*100)), f'nb_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(nb_clf, f)

        output_file = os.path.join(
            output_dir, str(int(size*100)), f'nb_output.csv')
        output_df = pd.DataFrame(
            {'tweet': X_test, 'true_labels': y_test, 'predicted_labels': y_pred})
        output_df.to_csv(output_file, index=False)

    # Plot results
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=results_df, x='data_size',
                 y='f1', marker='o', label='F1 Score')
    sns.lineplot(data=results_df, x='data_size',
                 y='accuracy', marker='o', label='Accuracy')
    plt.xlabel('Data Size')
    plt.ylabel('Performance')
    plt.title('Naive Bayes Performance vs Data Size')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'nb_results.png'))


# Call the function train_and_evaluate_nb
data_path = "/content/drive/MyDrive/CE807/Assignment2/2205000/test.csv"
output_dir = "/content/drive/MyDrive/CE807/Assignment2/2205000/models/1"
train_and_evaluate_nb(data_path, output_dir)


'''
Method 2
This code is defining two functions, preprocess_text() and train_method2(), to train a machine learning model to classify tweets as offensive or non-offensive. The preprocess_text() function takes a text string and applies a series of text preprocessing steps to it, including removing URLs and mentions, removing emojis, removing punctuation, converting to lowercase, tokenizing the text, removing stopwords, and joining the words back into a string. The train_method2() function loads the training and validation data from CSV files, cleans the tweet text using the preprocess_text() function, converts the label column to binary values, vectorizes the tweet text using the TfidfVectorizer, trains a Random Forest Classifier on the training set, makes predictions on the validation set, computes evaluation metrics including accuracy, precision, recall, and F1 score, saves the trained model and vectorizer to disk, and returns nothing.
'''


def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\S+", "", text)
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words("english")]
    # Join words
    text = " ".join(words)
    return text


def train_method2(train_file, val_file, model_dir):
    # Load the training data
    train_df = pd.read_csv(train_file)
    # Load the validation data
    val_df = pd.read_csv(val_file)

    # Clean the tweet text
    train_df['clean_text'] = train_df['tweet'].apply(preprocess_text)
    val_df['clean_text'] = val_df['tweet'].apply(preprocess_text)

    # Convert OFF to 0 and NOT to 1
    train_df['label'] = train_df['label'].apply(
        lambda x: 0 if x == 'OFF' else 1)
    val_df['label'] = val_df['label'].apply(lambda x: 0 if x == 'OFF' else 1)

    # Vectorize the tweet text
    vectorizer = TfidfVectorizer()
    train_vectors = vectorizer.fit_transform(train_df['clean_text'])
    val_vectors = vectorizer.transform(val_df['clean_text'])

    # Train the model
    clf = RandomForestClassifier()
    clf.fit(train_vectors, train_df['label'])

    # Make predictions on validation set
    val_preds = clf.predict(val_vectors)

    # Compute evaluation metrics
    accuracy = accuracy_score(val_df['label'], val_preds)
    precision = precision_score(val_df['label'], val_preds)
    recall = recall_score(val_df['label'], val_preds)
    f1 = f1_score(val_df['label'], val_preds)

    # Print the evaluation metrics
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1))

    # Save the model and vectorizer
    joblib.dump(clf, model_dir+'/random_forest.h5')
    joblib.dump(vectorizer, model_dir+'/vectorizer.h5')

    return


# Calling the function train_method2
train_method2('/content/drive/MyDrive/CE807/Assignment2/2205000/train.csv',
              '/content/drive/MyDrive/CE807/Assignment2/2205000/valid.csv',
              '/content/drive/MyDrive/CE807/Assignment2/2205000/models/2')

'''
This code defines a Python function named test_method2 that loads test data from a CSV file, cleans the tweet text by applying a text pre-processing function, converts the label values from 'OFF' to 0 and 'NOT' to 1, loads a saved machine learning model and vectorizer from disk, vectorizes the tweet text using the loaded vectorizer, and makes predictions on the test set using the loaded machine learning model. The code then computes evaluation metrics such as accuracy, precision, recall, and F1 score, and saves the predictions and evaluation metrics to a CSV file. Finally, the code displays the count of each label value and the total count of labels in the test data. The purpose of this code is to evaluate the performance of a machine learning model on a test set and generate evaluation metrics that can be used to assess the model's accuracy and effectiveness.
'''


def test_method2(test_file, model_dir):
    # Load the test data
    test_df = pd.read_csv(test_file)

    # Clean the tweet text
    test_df['clean_text'] = test_df['tweet'].apply(preprocess_text)

    # Convert OFF to 0 and NOT to 1
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 'OFF' else 1)

    # Load the saved model
    clf = joblib.load(
        '/content/drive/MyDrive/CE807/Assignment2/2205000/models/2/random_forest.h5')

    # Load the saved vectorizer
    vectorizer = joblib.load(
        '/content/drive/MyDrive/CE807/Assignment2/2205000/models/2/vectorizer.h5')

    # Vectorize the tweet text
    test_vectors = vectorizer.transform(test_df['clean_text'])

    # Make predictions on test set
    test_preds = clf.predict(test_vectors)

    # Compute evaluation metrics
    accuracy = accuracy_score(test_df['label'], test_preds)
    precision = precision_score(test_df['label'], test_preds)
    recall = recall_score(test_df['label'], test_preds)
    f1 = f1_score(test_df['label'], test_preds)

    # Print the evaluation metrics
    print("Accuracy: {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1))

    # Save the output in a CSV file
    output_df = pd.DataFrame(
        {'tweet': test_df['tweet'], 'true_label': test_df['label'], 'predicted_label': test_preds})
    output_df.to_csv(model_dir+'/output.csv', index=False)

    # Display the label value count and total count
    label_counts = test_df['label'].value_counts()
    total_count = test_df['label'].count()
    print("Label Value Count:")
    print(label_counts)
    print("Total Count:", total_count)


# Calling the function test_method2
test_method2('/content/drive/MyDrive/CE807/Assignment2/2205000/test.csv',
             '/content/drive/MyDrive/CE807/Assignment2/2205000/models/2')

# Checking the Output with true_label and predicted label
out_rf = pd.read_csv(
    '/content/drive/MyDrive/CE807/Assignment2/2205000/models/2/output.csv')
print(out_rf)

'''
This code defines a function train_test_models that trains and evaluates a machine learning model to predict the labels of tweets. The function takes two arguments, a path to a CSV file containing tweet data and a directory to save the trained models. The function first loads the tweet data from the CSV file, preprocesses the tweet text, and converts the labels from 'OFF' and 'NOT' to binary 0 and 1, respectively. The function then trains and tests the model for different data splits (25%, 50%, 75%, and 100% training data). For each data split, the function splits the data into training and testing sets, vectorizes the tweet text using TF-IDF, trains a Random Forest classifier, makes predictions on the test set, computes evaluation metrics such as accuracy, precision, recall, and F1 score, and saves the trained model, vectorizer, and test data predictions to a directory. The function then prints the evaluation metrics and classification report for each data split. The evaluation metrics are useful for comparing the performance of the model for different data splits and choosing the best split for the final model.
'''


def train_test_models(data_file, model_dir):
    # Load the data
    df = pd.read_csv(data_file)

    # Clean the tweet text
    df['clean_text'] = df['tweet'].apply(preprocess_text)

    # Convert OFF to 0 and NOT to 1
    df['label'] = df['label'].apply(lambda x: 0 if x == 'OFF' else 1)

    # Train and test models for different data splits
    splits = [0.25, 0.5, 0.75, 0.99]

    for split in splits:
        # Split the data
        train_data, test_data = train_test_split(
            df, test_size=1-split, random_state=42)

        # Vectorize the tweet text
        vectorizer = TfidfVectorizer(max_features=10000)
        train_vectors = vectorizer.fit_transform(train_data['clean_text'])
        test_vectors = vectorizer.transform(test_data['clean_text'])

        # Train the model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(train_vectors, train_data['label'])

        # Make predictions on test set
        test_preds = clf.predict(test_vectors)

        # Compute evaluation metrics
        accuracy = accuracy_score(test_data['label'], test_preds)
        precision = precision_score(test_data['label'], test_preds)
        recall = recall_score(test_data['label'], test_preds)
        f1 = f1_score(test_data['label'], test_preds)

        # Print the evaluation metrics and classification report
        print(
            "Evaluation Metrics for {:.0f}% Training Data:".format(split*100))
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1 Score: {:.4f}".format(f1))
        print("Label Value Counts:\n", test_data['label'].value_counts())
        print("Label Total Count:", len(test_data))

        # Save the model and vectorizer
        model_path = os.path.join(model_dir, str(int(split*100)))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        dump(clf, os.path.join(model_path, 'random_forest.joblib'))
        dump(vectorizer, os.path.join(model_path, 'vectorizer.joblib'))

        # Save the test data predictions to a CSV file
        output_df = pd.DataFrame(
            {'tweet': test_data['tweet'], 'true_label': test_data['label'], 'predicted_label': test_preds})
        output_path = os.path.join(model_path, 'output.csv')
        output_df.to_csv(output_path, index=False)


data_file = "/content/drive/MyDrive/CE807/Assignment2/2205000/train.csv"
model_dir = "/content/drive/MyDrive/CE807/Assignment2/2205000/models/2"

train_test_models(data_file, model_dir)
