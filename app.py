import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from PIL import Image, ImageTk
import pickle
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from tensorflow import keras
import re
import pygame
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# Load the trained classification models
modelLog = pickle.load(open('models/logistic.sav', 'rb'))
modelsvm = pickle.load(open('models/svm.sav', 'rb'))
modelrf = pickle.load(open('models/randomforest.sav', 'rb'))
modelbag = pickle.load(open('models/bagging.sav', 'rb'))
modelsoft = pickle.load(open('models/voting.sav', 'rb'))
modelbayes = pickle.load(open('models/bayes.sav', 'rb'))
modelLSTM = keras.models.load_model('models/LSTM.h5')
modelCNN = keras.models.load_model('models/CNN.h5')
modelAttention = keras.models.load_model('models/Attention.h5')
modelBiLSTM = keras.models.load_model('models/BiLSTM.h5')
modelBiLSTM_CNN = keras.models.load_model('models/BiLSTMCNN.h5')
modelGru = keras.models.load_model('models/GRU.h5')
modelBiLSTMCNNa = keras.models.load_model('models/BiCNNa.h5')
modelBiA = keras.models.load_model('models/BiA.h5')


# Load the trained Doc2Vec model
infermod=Doc2Vec.load("models/doc2vec.model")

# Load the music file
pygame.mixer.init()
click = pygame.mixer.Sound('music/click.wav')
click.set_volume(0.5)

# create the main window using customtkinter
window = ctk.CTk()
window.title('Fake News Detection')
window.geometry('1080x560')
window.resizable(False, False)
window.configure(bg='white')
window.eval('tk::PlaceWindow . center')
window.iconbitmap('images/icon.ico')

exit_button = ctk.CTkButton(master=window, text='Exit',fg_color="red",hover_color="red", command=window.destroy)
exit_button.place(relx=0.93, rely=0.05, anchor=tk.CENTER)

# Make A Title Label FAKE NEWS DETECTION
title_label = ctk.CTkLabel(master=window, text='FAKE NEWS DETECTION', font=('Calibri', 30, 'bold'))
title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

def switch_theme():
    # Switches between dark and light theme
    click.play(0)
    if ctk.get_appearance_mode() == 'Light':
        ctk.set_appearance_mode('dark')
    else:
        ctk.set_appearance_mode('light')

switch_button = ctk.CTkButton(master=window, text='Light / DARK',text_color=['white','black'],fg_color=['black','white'],hover_color=['black','white'], command=switch_theme)
switch_button.place(relx=0.07, rely=0.05, anchor=tk.CENTER)

# Make A Label For The Input Text
input_label = ctk.CTkLabel(master=window, text='Input Text :', font=('Calibri', 20, 'bold'))
input_label.place(relx=0.09, rely=0.23, anchor=tk.CENTER)
input_text = tk.Text(window, height=15, width=45, font=('Calibri', 13,))
input_text.place(relx=0.20, rely=0.5, anchor="center")

# create CTk scrollbar
textbox_scrollbar = ctk.CTkScrollbar(window, command=input_text.yview)
textbox_scrollbar.place(relx=0.36, rely=0.5, anchor=tk.W)
input_text.configure(yscrollcommand=textbox_scrollbar.set)

# Make A Label For The Output Text
output_label = ctk.CTkLabel(master=window, text='Judgement :', font=('Calibri', 20, 'bold'))
output_label.place(relx=0.735, rely=0.44, anchor=tk.CENTER)
output_text = tk.Text(window, height=1, width=20, font=('Calibri', 22))
output_text.place(relx=0.80, rely=0.5, anchor="center")
output_text.config(state="disabled")

# Make a function to clear the input and output text
def clear_text():
    click.play(0)
    input_text.delete('1.0', tk.END)
    output_text.config(state="normal")
    output_text.delete('1.0', tk.END)
    output_text.config(state="disabled")
    
# Make a button to clear the input and output text
clear_button = ctk.CTkButton(master=window, text='Clear',fg_color="gray",hover_color="blue", command=clear_text)
clear_button.place(relx=0.285, rely=0.77, anchor=tk.CENTER)

# Make a dropdown menu to select the model
optionmenu_var = ctk.StringVar(value="Logistic Regression")

def optionmenu_callback(choice):
    click.play(0)
    print("optionmenu dropdown clicked:", choice)
    
# Make A Label For The Output Text
model_label = ctk.CTkLabel(master=window, text='Model :', font=('Calibri', 18, 'bold'))
model_label.place(relx=0.44, rely=0.33, anchor=tk.CENTER)
combobox = ctk.CTkOptionMenu(master=window,text_color=['black','white'],values=["Logistic Regression","Soft Voting","SVM","Random Forest", "Bagging","Bayes","LSTM","CNN","Attention","BiLSTM","BiLSTM CNN","GRU","BiLSTM CNN Attention","BiLSTM Attention"],command=optionmenu_callback,fg_color=['white','black'],variable=optionmenu_var)
combobox.place(relx=0.47, rely=0.335, anchor=tk.W)

def preprocess_text(text):
    # remove the text with garbage values like emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # remove the text with html tags
    text = re.sub(r'<.*?>', '', text)

    # remove the text with url or links like http or www
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)

    # convert to lower case
    text = text.lower()

    # remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # remove special characters
    text = re.sub(r'\W+', ' ', text)

    text = re.sub(r'\s+', ' ', text)
    return text
def fake_news():
    click.play(0)
    output_text.config(state="normal")
    # Clear the output text entry widget
    output_text.delete("1.0","end")
    
    # Get the input text from the text entry widget as a string
    input_text_string = input_text.get("1.0","end-1c")
    input_text_string = preprocess_text(input_text_string)
    # Split the input text string into a list of strings
    words = input_text_string.split()
    
    # Convert the words into infer vector
    infer_vector = infermod.infer_vector(words)
    
    vectors = np.array(infer_vector).reshape(1, -1)
    
    # check which model is selected
    if optionmenu_var.get() == "Logistic Regression":
        prediction = modelLog.predict(vectors)[0]
    elif optionmenu_var.get() == "SVM":
        prediction = modelsvm.predict(vectors)[0]
    elif optionmenu_var.get() == "Random Forest":
        prediction = modelrf.predict(vectors)[0]
    elif optionmenu_var.get() == "Bagging":
        prediction = modelbag.predict(vectors)[0]
    elif optionmenu_var.get() == "Soft Voting":
        prediction = modelsoft.predict(vectors)[0]
    elif optionmenu_var.get() == "Bayes":
        prediction = modelbayes.predict(vectors)[0]
    elif optionmenu_var.get() == "LSTM":
        prediction = modelLSTM.predict(vectors)[0]
    elif optionmenu_var.get() == "CNN":
        prediction = modelCNN.predict(vectors)[0]
    elif optionmenu_var.get() == "Attention":
        prediction = modelAttention.predict([vectors,vectors])[0]
    elif optionmenu_var.get() == "BiLSTM":
        prediction = modelBiLSTM.predict(vectors)[0]
    elif optionmenu_var.get() == "BiLSTM CNN":
        prediction = modelBiLSTM_CNN.predict(vectors)[0]
    elif optionmenu_var.get() == "GRU":
        prediction = modelGru.predict(vectors)[0]
    elif optionmenu_var.get() == "BiLSTM CNN Attention":
        prediction = modelBiLSTMCNNa.predict(vectors)[0]
    elif optionmenu_var.get() == "BiLSTM Attention":
        prediction = modelBiA.predict(vectors)[0]
        
    # prediction = prediction[0][0]
        
    
    if optionmenu_var.get() == "Logistic Regression" or optionmenu_var.get() == "SVM" or optionmenu_var.get() == "Random Forest" or optionmenu_var.get() == "Bagging" or optionmenu_var.get() == "Soft Voting" or optionmenu_var.get() == "Bayes":
        prediction=int(prediction)
        if prediction == 1:
            # change the color of the output text to green
            output_text.config(fg="green")
            output_text.insert("end", "Real News")
        else:
            
            # change the color of the output text to red
            output_text.config(fg="red")
            output_text.insert("end", "Fake News")
    else:
        prediction=float(prediction)
        if prediction >= .5:
            # change the color of the output text to green
            output_text.config(fg="green")
            output_text.insert("end", "Real News")
        else:
            # change the color of the output text to red
            output_text.config(fg="red")
            output_text.insert("end", "Fake News")
    output_text.config(state="disabled")
    
    print(prediction)
# Make a button to detect the fake news
detect_button = ctk.CTkButton(master=window, text='Detect',fg_color="gray",hover_color="green", command=fake_news)
detect_button.place(relx=0.115, rely=0.77, anchor=tk.CENTER)
window.mainloop()