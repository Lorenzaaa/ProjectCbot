import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('popular') 
nltk.download('punkt',  download_dir='.')
nltk.download('punkt_tab', download_dir='.')
nltk.download('wordnet', download_dir='.')
nltk.download('omw-1.4', download_dir='.')
nltk.data.path.append(r'C:\Users\12062\MyProjects\ProjectChatbot\.venv')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('C:/Users/12062/MyProjects/ProjectChatbot/.venv/intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList=nltk.word_tokenize(pattern, language='english')
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters ]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb' ))

training = []
outputEmpty = [0] * len(classes)


for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)


training = np.array(training)
np.random.shuffle(training)


trainX = training[:, :len(words)]
trainY = training[:, len(words):]


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(trainX[0]),)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(trainY[0]), activation="softmax")
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

 
model.save('chatbot_model1.keras')

pickle.dump(hist.history, open('training_history.pkl', 'wb'))

print("Executed")