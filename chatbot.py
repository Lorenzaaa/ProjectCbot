import random
import json
import pickle
import numpy as np 
import nltk
nltk.download('punkt',  download_dir='.')
nltk.download('punkt_tab', download_dir='.')
nltk.download('wordnet', download_dir='.')
nltk.download('omw-1.4', download_dir='.')
nltk.data.path.append(r'C:\Users\12062\MyProjects\ProjectChatbot\.venv')

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('C:/Users/12062/MyProjects/ProjectChatbot/.venv/intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_model1.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence) :
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
    return np.array(bag, dtype=np.float32)

def predict_class(sentence):
    bow = bag_of_words((sentence))
    bow = np.array([bow], dtype=np.float32)
    res = model.predict(bow)[0]          
    
    
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results :
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})     
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:  
        return "I'm not sure how to respond to that."
    
    list_of_intents = intents_json['intents']
    tag = intents_list[0]['intent']
    for i in list_of_intents: 
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break 
        
    return result
print("Amazing work! Bot is up and running!")

try:
    while True:
        message = input("You: ")  
        if message.lower() == 'quit':  
            print("Bot: Goodbye!")
            break
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("Bot:", res)  
except EOFError:
    print("\nChat session ended.")
except KeyboardInterrupt:
    print("\nChat session ended.")