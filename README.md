# Chatbot Project

This project is a chatbot application that uses machine learning to understand user inputs and generate appropriate responses. It is built using Python, Natural Language Processing (NLP) techniques, and a neural network model created with TensorFlow and Keras.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Additional Notes](#additional-notes)

## Requirements

The project depends on the following libraries:
- Python 3.x
- TensorFlow
- Keras
- NLTK
- Numpy

You can install all dependencies using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Required NLTK Data

To use the Natural Language Toolkit (NLTK), you need to download specific datasets. In the code, there are two commented lines for downloading NLTK datasets:

- punkt_tab for tokenizing words.
- wordnet for word lemmatization.

You can manually download these using the following commands inside your script or interactively:

```python
import nltk
nltk.downlaod('punkt')
nltk.download('wordnet')
```

## Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/chatbot-project.git
cd chatbot-project
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Download NLTK datasets as shown in the (#Requirements) section.

## Project Structure

The project directory is structured as follows:

```bash
chatbot-project/
│
├── intents.json        # Contains training data and patterns
├── train_chatbot.py    # Script to preprocess data and train the chatbot model
├── chatbot_model.h5    # Saved trained model
├── words.pkl           # Preprocessed words for the model
├── classes.pkl         # Classes (responses) for the model
├── chatbot.py          # Main script to interact with the chatbot
├── README.md           # Project documentation (this file)
└── requirements.txt    # Python dependencies
```
### Files Explanation

- `intents.json`: The file contains predefined intents (categories) and responses that the chatbot can use to understand user input and generate appropriate replies.
- `train_chatbot.py`: Script that preprocesses the `intents.json` file, builds and trains a neural network model, and saves the trained model and processed data.
- `chatbot_model.h5`: The trained model is saved in this file.
- `words.pkl` and classes.pkl: These pickle files store processed data for words and classes that the chatbot uses during prediction.

## How to Use

1. Training the Model

The first step is to train the chatbot model using the training script. Make sure your intents.json file is filled with proper training data.

Run the training script:

```bash
python train_chatbot.py
```

This will preprocess the training data, build a neural network model, and save the trained model in `chatbot_model.h5`. It will also save the processed words and classes in `words.pkl` and `classes.pkl` files, respectively.

2. Running the Chatbot

Once the model is trained, you can run the chatbot using the main script:

```bash
python chatbot.py
```

You can now interact with the chatbot via the terminal, and it will respond based on the model’s predictions.

### Model Training

The model is a Sequential neural network built with TensorFlow and Keras. The architecture consists of:

- Input layer: Takes the preprocessed training data.
- Hidden layers: Dense layers with activation functions.
- Output layer: Produces a probability for each class (intent) based on the input.

### Preprocessing Steps:

- Tokenization and Lemmatization: Input sentences are tokenized and lemmatized to reduce words to their root forms.
- Bag of Words Representation: Sentences are converted into a numerical bag of words representation.
- Training the Model: The preprocessed data is used to train the neural network, and the model is saved for future use.

## Additional Notes
- Customizing Responses: You can modify or add more intents to the `intents.json` file to expand the chatbot's responses. After making changes, you will need to retrain the model.
- Pickle Files: `words.pkl` and `classes.pkl` store the processed vocabulary and intent classes, which allow the chatbot to map user inputs to intents more efficiently.
