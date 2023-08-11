import tkinter as tk
from tkinter import scrolledtext, INSERT, END
import tkinter.font as tkFont
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
with open('./intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)
words = pickle. load (open('words.pkl', 'rb') )
classes = pickle. load(open("classes.pkl", 'rb') )
model = load_model('./chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

  

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate (res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response (intents_list ,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

# GUI class
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot GUI")
        self.root.configure(bg='black')

        self.chat_history = scrolledtext.ScrolledText(root, width=50, height=15, bg='black', fg='gold', insertbackground='gold')
        self.chat_history.pack(padx=10, pady=10)

        self.user_input = tk.Entry(root, width=50, bg='black', fg='gold', insertbackground='gold')
        self.user_input.pack(padx=10, pady=10)

        self.send_button = tk.Button(root, text="Send", command=self.send_message, bg='black', fg='gold')
        self.send_button.pack(padx=10, pady=10)

        self.user_input.bind('<Return>', lambda event=None: self.send_message())

    def send_message(self):
        user_message = self.user_input.get()
        self.chat_history.insert(tk.END, "You: " + user_message + "\n")
        self.user_input.delete(0, tk.END)

        bot_response = self.get_bot_response(user_message)
        self.chat_history.insert(tk.END, "Bot: " + bot_response + "\n")

    def get_bot_response(self, message):
        intents_list = predict_class(message)
        bot_response = get_response(intents_list, intents)
        return bot_response

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    print("GO! Bot is running!")
    root.mainloop()