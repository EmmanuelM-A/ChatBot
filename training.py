import random
import json
import pickle
import numpy as np

import nltk
#nltk.download('punkt_tab')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# Loads and saves the intents data as a JSON object
intents = json.loads(open('intents.json').read())

words = []

# Refers to different tag classes like greetings, goodbye or news
classes = []

# Holds the combinations of the words and classes
documents = []

ignore_letters = ["?", "!", ".", ","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        '''
        Converts a sentence into list where each element is the word in the sentence. For example,
        "Hi, I am John" would be ["hi", "i", "am", "john"]
        '''
        word_list = nltk.word_tokenize(pattern)

        words.extend(word_list)

        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []

output_empty = [0] * len(classes)

# All the document data will be inside the training list to train the neural network
for document in documents:
    # For each combination of words we create an empty bag of words
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Create the bag of words for each document
    bag = [1 if word in word_patterns else 0 for word in words]

    # Output is a one-hot encoding of the class
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

training_x = list(training[:, 0])
training_y = list(training[:, 1])


# NEURAL NETWORK MODEL

model = Sequential()
model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(training_x), np.array(training_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras', hist)

#print("Done")
