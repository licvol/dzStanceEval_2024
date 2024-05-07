# -*- coding: utf-8 -*-

"""# **Libraries**"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import pandas as pd

"""# **Read dataset**"""

data = pd.read_csv("Mawqif_AllTargets_Train.csv", encoding="utf-8")

"""# **Normalisation**"""

import re

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def replace_emojis(text):
    emojis = {
        "\U0001F601": "فرح",
        "\U0001F602": "فرح",
        "\U0001F603": "فرح",
        "\U0001F604": "فرح",
        "\U0001F605": "حزن",
        "\U0001F606": "فرح",
        "\U0001F607": "فرح",
        "\U0001F60C": "فرح",
        "\U0001F60D": "حب",
        "\U0001F60E": "حزن",
        "\U0001F611": "حب",
        "\U0001F612": "حزن",
        "\U0001F613": "حزن",
        "\U0001F615": "حزن",
        "\U0001F618": "حب",
        "\U0001F619": "حب",
        "\U0001F61A": "حب",
        "\U0001F61B": "حب",
        "\U0001F61C": "حب",
        "\U0001F61D": "حب",
        "\U0001F61E": "حب",
        "\U0001F61F": "حب",
        "\U0001F620": "غضب",
        "\U0001F621": "غضب",
        "\U0001F622": "حزن",
        "\U0001F623": "فرح",
        "\U0001F624": "حزن",
        "\U0001F625": "حزن",
        "\U0001F626": "حزن",
        "\U0001F627": "غضب",
        "\U0001F628": "حزن",
        "\U0001F629": "حزن",
        "\U0001F62A": "حزن",
        "\U0001F62B": "حزن",
        "\U0001F62C": "حزن",
        "\U0001F62D": "حزن",
        "\U0001F62E": "حزن",
        "\U0001F62F": "حزن",
        "\U0001F630": "غضب",
        "\U0001F631": "حزن",
        "\U0001F632": "حزن",
        "\U0001F633": "حزن",
        "\U0001F634": "حزن وفرح",
        "\U0001F635": "حزن",
        "\U0001F636": "حزن",
        "\U0001F637": "فرح",
        "\U0001F638": "فرح",
        "\U0001F639": "فرح",
        "\U0001F63A": "فرح",
        "\U0001F63B": "فرح",
        "\U0001F63C": "فرح",
        "\U0001F63D": "فرح",
        "\U0001F63E": "فرح",
        "\U0001F63F": "فرح",
        "\U0001F640": "حزن",
        "\U0001F641": "حزن",
        "\U0001F642": "فرح",
        "\U0001F643": "فرح",
        "\U0001F644": "فرح",
        "\U0001F645": "فرح",
        "\U0001F646": "فرح",
        "\U0001F647": "فرح",
        "\U0001F648": "فرح",
        "\U0001F649": "فرح",
        "\U0001F64A": "فرح",
        "\U0001F64B": "فرح",
        "\U0001F64C": "فرح",
        "\U0001F64D": "فرح",
        "\U0001F64E": "فرح",
        "\U0001F64F": "فرح",
    }
    for emoji, arabic_equivalent in emojis.items():
        text = re.sub(emoji, arabic_equivalent, text)
    return text

# Applies normalization to texts
data['text'] = data['text'].apply(normalize_arabic)

# Replace emojis with their Arabic equivalent
data['text'] = data['text'].apply(replace_emojis)

print(data['text'])

"""# **Split data into training and developpement sets (70% train, 30% dev)**"""

textData = data['text'].astype(str)
stanceData = data['stance'].astype(str)

# Convert stances to uppercase
stanceData = [x.replace("nan", "None") for x in stanceData]
stanceData =[x.upper() for x in stanceData]  # Uppercase stances
print(data.head(5))

# Split data into training and testing sets (70% train, 30% test)
X_train, X_dev, y_train, y_dev = train_test_split(textData, stanceData, test_size=0.3, random_state=42)
test_ids = data.loc[X_dev.index, "ID"].tolist()  # Extract IDs for testing data
test_topics = data.loc[X_dev.index, "target"].tolist()  # Extract topics for testing data

# Load pre-trained model
model = SentenceTransformer("xlm-r-100langs-bert-base-nli-stsb-mean-tokens")

# Define models and corresponding tokenizers
# "xlm-r-bert-base-nli-stsb-mean-tokens": This model is based on the XLM-RoBERTa architecture and is
# fine-tuned for various tasks, including semantic textual similarity. It's pre-trained on a multilingual
#  corpus, which includes Arabic.


"""# **Train Model**"""

# Transform text data to embeddings
X_train_embeddings = model.encode(X_train.tolist())
X_dev_embeddings = model.encode(X_dev.tolist())

# Train a classifier (Logistic Regression)
classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

# Train the model
classifier.fit(X_train_embeddings, y_train)

"""# **Evaluate the model on the dev set**"""

from sklearn.metrics import classification_report

# Predict labels for test data
y_pred = classifier.predict(X_dev_embeddings)

report = classification_report(y_dev, y_pred, output_dict=True)

# Print classification report
print("Classification Report:")
print(classification_report(y_dev, y_pred, digits=4))

# Write gold labels (IDs, topic, text, uppercase stance) to a separate file
with open("gold_labels.txt", "w", encoding="utf-8") as outfile:
    for i, (topic, text, stance) in enumerate(zip(test_topics, X_dev, y_dev)):
        outfile.write(f"{test_ids[i]}\t{topic}\t{text}\t{stance.upper()}\n")

# Write predicted labels (IDs, topic as target, text, prediction) to a separate file
with open("predictions.txt", "w", encoding="utf-8") as outfile:
    for i, (prediction, text) in enumerate(zip(y_pred.tolist(), X_dev)):
        outfile.write(f"{test_ids[i]}\t{test_topics[i]}\t{text}\t{prediction}\n")  # Use topic for target

print("Stance prediction results with IDs, topics, and uppercase stances saved to separate files.")

# Now run StanceEval.py
!python "StanceEval.py" gold_labels.txt predictions.txt > obtained_results.log

"""**# Predict the blind data output**"""

blind = pd.read_csv("Mawqif_AllTargets_Blind Test.csv", encoding="utf-8")
textDataBlind = blind['text'].astype(str)

# Split data into training and testing sets (70% train, 30% test)
blind_ids = blind['ID'].astype(str)
blind_topics = blind['target'].astype(str)

X_blind_embeddings = model.encode(textDataBlind.tolist())

# Predict labels for blind data
y_pred_blind = classifier.predict(X_blind_embeddings)

import csv
# Save predictions of blind data to csv file
print('Number of comments per ID is:', len(blind_ids))
print('Number of comments per Target is:', len(blind_topics))
print('Number of predicted output is:', len(y_pred_blind))

# Create a CSV file
with open('dzStanceBlindTestPred.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['ID', 'Target', 'Stance'])

    # Write the data rows
    for i in range(len(blind_ids)):
        writer.writerow([blind_ids[i], blind_topics[i], y_pred_blind[i]])

print('CSV file created successfully!')