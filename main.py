import tkinter as tk
# Import module
from tkinter import *
from tkinter import font as tkFont
from PIL import ImageFont
import os
from tkinter import filedialog, messagebox
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pywt
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

norm=0
aiModel =0

accuracyLabeL=""
classificationLabel=""
test_up_Signals=[]
train_up_Signals=[]
train_down_Signals=[]
test_down_Signals=[]
train_up_wavelet_features = []
train_down_wavelet_features = []
test_up_wavelet_features = []
test_down_wavelet_features = []

file_path=''
lines=[]
testSignal=[]
def ReadingTrainSamples(file_name, data):
    with open(file_name, 'r') as f:
        lines = f.readlines()  # All lines are stored in a list
        for line in lines:
            EOG_Sample = []
            print(line.strip())
            L = line.strip()
            L = line.split('\t')
            for i in L:
                EOG_Sample.append(int(i))
            if data == "test_up":
                test_up_Signals.append(EOG_Sample)
            if data == "train_down":
                train_down_Signals.append(EOG_Sample)
            if data == "test_down":
                test_down_Signals.append(EOG_Sample)
            if data == "train_up":
                train_up_Signals.append(EOG_Sample)
            if data == "test":
                testSignal.append(EOG_Sample)

def preprocessing(signal, lowcut=0.5, highcut=20, fs=176, order=4, m=3):
    # Mean removal
    signal = signal - np.mean(signal)

    #  Bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    signal = filtfilt(b, a, signal)

    # Normalization
    if norm==1:
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    elif norm==2:
        signal = (2 * ((signal - signal.min()) / (signal.max() - signal.min())) - 1)

    #  Downsampling
    downsampled_signal = []
    for i in range(0, len(signal) - 1, m):
        downsampled_signal.append(signal[i])
    signal = np.array(downsampled_signal)

    return signal

def apply_wavelet(originalSignal):
    all_coeffs = pywt.wavedec(originalSignal, 'db3', level=8)
    desired_coeff = all_coeffs[1:7]
    features = []
    for coeff in desired_coeff:
        mean = np.mean(coeff)
        std = np.std(coeff)
        features.extend([mean, std])

    return np.array(features)



# Build Improved Neural Network Model
def neural_network(input_dim):
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim),
        Dropout(0.4),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0003),  # Lower learning rate for better convergence
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def predict_signal():
    global train_up_Signals
    global train_down_Signals
    global test_up_Signals
    global test_down_Signals
    global testSignal
    global train_up_wavelet_features
    global train_down_wavelet_features
    global test_up_wavelet_features
    global test_down_wavelet_features
    global testLabelll
    global lines
    global file_path
    train_up_Signals= []
    train_down_Signals = []
    test_up_Signals = []
    test_down_Signals = []
    train_up_wavelet_features =[]
    train_down_wavelet_features=[]
    test_up_wavelet_features = []
    test_down_wavelet_features = []
    test_features=[]
    testSignal=[]

    #read training data
    ReadingTrainSamples('train_down.txt', "train_down")
    ReadingTrainSamples('train_up.txt', "train_up")
    ReadingTrainSamples('test_down.txt', "test_down")
    ReadingTrainSamples('test_up.txt', "test_up")
    ReadingTrainSamples(file_path, "test")

    #process training data
    for i in range (0,(len(train_up_Signals))):
        train_up_Signals[i] = preprocessing(train_up_Signals[i])
    for i in range(0, (len(train_down_Signals))):
        train_down_Signals[i] = preprocessing(train_down_Signals[i])
    for i in range(0, (len(test_up_Signals))):
        test_up_Signals[i] = preprocessing(test_up_Signals[i])
    for i in range(0, (len(test_down_Signals))):
        test_down_Signals[i] = preprocessing(test_down_Signals[i])
    # Preprocess the signal
    for i in range(0, (len(testSignal))):
        testSignal[i] = preprocessing(testSignal[i])

    # Extract features
    for i in range(0, (len(train_up_Signals))):
        train_up_wavelet_features.append(apply_wavelet(train_up_Signals[i]))
    for i in range(0, (len(train_down_Signals))):
        train_down_wavelet_features.append(apply_wavelet(train_down_Signals[i]))
    for i in range(0, (len(test_up_Signals))):
        test_up_wavelet_features.append(apply_wavelet(test_up_Signals[i]))
    for i in range(0, (len(test_down_Signals))):
        test_down_wavelet_features.append(apply_wavelet(test_down_Signals[i]))
    # test_features = apply_wavelet(preprocessed_signal).reshape(1, -1)
    for i in range(0, (len(testSignal))):
        test_features.append(apply_wavelet(testSignal[i]))

    # Label 1 for 'Up'
    train_up_labels = np.ones(np.array(train_up_wavelet_features).shape[0])
    test_up_labels = np.ones(np.array(test_up_wavelet_features).shape[0])
    # Label 0 for 'Down'
    train_down_labels = np.zeros(np.array(train_down_wavelet_features).shape[0])
    test_down_labels = np.zeros(np.array(test_down_wavelet_features).shape[0])

    # Combine data and labels
    train_data = np.concatenate([train_down_wavelet_features, train_up_wavelet_features], axis=0)
    train_data_labels = np.concatenate([train_down_labels, train_up_labels], axis=0)
    test_data = np.concatenate([test_down_wavelet_features, test_up_wavelet_features], axis=0)
    test_data_labels = np.concatenate([test_down_labels, test_up_labels], axis=0)

    X_train = train_data
    X_test = test_data
    y_train = train_data_labels
    y_test = test_data_labels

    if aiModel==1:
      knn_model = KNeighborsClassifier(n_neighbors=8)
      knn_model.fit(X_train, y_train)
      # Predict label
      prediction = knn_model.predict(test_features)
      print(len(test_features))
      print(prediction)
      label = "Up" if prediction[int(sigNum_entry.get())] == 1 else "Down"
      accuracy = accuracy_score(y_test, knn_model.predict(X_test)) * 100
      predictionLabel.config(text=label)
      accuracyLabel.config(text=accuracy)
    elif aiModel==2:
        # Standardize Features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_data)
        test_features = scaler.transform(test_data)
        # Initialize and Train the Neural Network Model
        neural_net_model = neural_network(train_features.shape[1])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

        history = neural_net_model.fit(
            train_features, y_train,
            epochs=200,  # Allow longer training with early stopping
            batch_size=32,
            validation_data=(test_features, y_test),
            callbacks=[reduce_lr, early_stopping]
        )
        # Evaluate the Neural Network Model
        test_loss, test_accuracy = neural_net_model.evaluate(test_features, y_test)
        print(f"Neural Network Accuracy: {test_accuracy * 100:.2f}%")
        accuracyLabel.config(text=f"{test_accuracy * 100:.2f}%")


def browse_file():
    global file_path
    file_path=""
    file_path = filedialog.askopenfilename(title="Select Signal File", filetypes=[("Text Files", "*.txt")])


def pressedNorm1():
    global norm
    norm = 1
    norm1.config(image=radioPressedbtn)  # Change the image when button is pressed
    norm2.config(image=radioNotPressedbtn)

def pressedNorm2():
    global norm
    norm = 2
    norm1.config(image=radioNotPressedbtn)
    norm2.config(image=radioPressedbtn)  # Change the image when button is pressed


def go_to_homepage():
    test.pack(fill="both", expand=True)
    frame1.pack_forget()
    test.tkraise()


def pressedknn():
    global aiModel
    aiModel=1
    knnmodel.config(image=radioPressedbtn)  # Change the image when button is pressed
    neuralmodel.config(image=radioNotPressedbtn)


def pressedNeural():
    global aiModel
    aiModel=2
    knnmodel.config(image=radioNotPressedbtn)
    neuralmodel.config(image=radioPressedbtn)  # Change the image when button is pressed



def predict():
    if aiModel==1:
        print("helloo")
    elif aiModel==2:
        print("yokoo")

# Create the main window
root = tk.Tk()
root.geometry("1000x615")  # Set the size of the main window

# Create frame1
frame1 = tk.Frame(root, width=1000, height=615)
frame1.pack_propagate(False)
frame1.pack(fill="both", expand=True)

# Add image file
bg = PhotoImage(file="backgrounds/frame1bgresize.png")
# Show image using label
backgroundimage = Label(frame1, image=bg, width=1000, height=613)
backgroundimage.place(x=0, y=0)

startBtn_img = PhotoImage(file="backgrounds/strtbtn.png", width=260, height=75)
startBtn = tk.Button(frame1,image=startBtn_img,borderwidth=0,highlightthickness=5,command=go_to_homepage,relief="flat")
startBtn.place(x=245.0,y=490.0,width=260,height=75)


# Create the test frame
test = tk.Frame(root, width=1000, height=615)
test.pack_propagate(False)

bg2 = PhotoImage(file="backgrounds/testPage.png")
backgroundimage2 = Label(test, image=bg2, width=1000, height=613)
backgroundimage2.place(x=0, y=0)

predict_img = PhotoImage(file="backgrounds/predictbtnnew2.png", width=200, height=75)
predictBtn = tk.Button(test,image=predict_img,borderwidth=0,highlightthickness=5,command=predict_signal, relief="flat")
predictBtn.place(x=780.0,y=520.0,width=200,height=75)

choose_img = PhotoImage(file="backgrounds/chzFile.png", width=220, height=70)
chooseFileBtn = tk.Button(test,image=choose_img,borderwidth=0,highlightthickness=5,command=browse_file,relief="flat")
chooseFileBtn.place(x=637.0,y=47.0,width=220,height=70)

signalNum = PhotoImage(file="backgrounds/signalInput.png")
# Show image using label
inputSignalNum = Label(test, image=signalNum, width=195, height=100)
inputSignalNum.place(x=783, y=123)
sigNum_entry = tk.Entry(test, font=("Croissant One", 30), width=1)
sigNum_entry.place(x=862, y=175)

radioPressedbtn = PhotoImage(file="backgrounds/radioButtonimg.png", width=41, height=40)
radioNotPressedbtn = PhotoImage(file="backgrounds/nonPressed.png", width=41, height=40)

norm1 = tk.Button(test,image=radioNotPressedbtn,command=pressedNorm1,relief="flat")
norm1.place(x=473.0,y=273.0,width=41,height=40)

norm2 = tk.Button(test,image=radioNotPressedbtn,command=pressedNorm2,relief="flat")
norm2.place(x=473.0,y=328.0,width=41,height=40)

radioPressedbtnmodel = PhotoImage(file="backgrounds/radioButtonimg.png", width=41, height=40)
radioNotPressedbtnmodel = PhotoImage(file="backgrounds/nonPressed.png", width=41, height=40)

knnmodel = tk.Button(test,image=radioNotPressedbtnmodel,command=pressedknn,relief="flat")
knnmodel.place(x=473.0,y=489.0,width=41,height=40)

neuralmodel = tk.Button(test,image=radioNotPressedbtn,command=pressedNeural,relief="flat")
neuralmodel.place(x=473.0,y=543.0,width=41,height=40)

#######################################################

predictionLabel = Label(test,bg="#FFFFFF", width=8, height=1,text=classificationLabel,font=("Arial", 20))
predictionLabel.place(x=815, y=275)
accuracyLabel= Label(test,bg="#FFFFFF", width=8, height=1,text=accuracyLabeL,font=("Arial", 20))
accuracyLabel.place(x=822, y=380)

# Pack frame1 initially
frame1.pack(fill="both", expand=True)

# Run the main loop
root.mainloop()