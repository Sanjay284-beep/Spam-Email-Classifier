from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
from tkinter import simpledialog
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


main = tkinter.Tk()
main.title("Machine Learning for Email Spam Filtering")
main.geometry("1200x700")

global filename
global classifier
global X, Y
global dataset
accuracy = []
recall = []
precision = []
global X_train, X_test, y_train, y_test

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "spambase", filetypes=[("CSV files","*.csv"),("Data files","*.data"),("All files","*.*")])
    if not filename:
        return
    text.delete('1.0', END)
    try:
        dataset = pd.read_csv(filename)
        text.insert(END,filename+' Loaded\n\n')
        text.insert(END,str(dataset.head()))
    except Exception as e:
        text.insert(END,"Error loading dataset:\n")
        text.insert(END,str(e))

def preprocess():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    global X, Y
    global dataset
    if dataset is None:
        text.insert(END,"Please upload the Spambase dataset first.\n")
        return
    data = dataset.values
    cols = data.shape[1]-1
    X = data[:,0:cols]
    Y = data[:,cols]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    text.insert(END,"Total spam & non spam records found in dataset is : "+str(len(X))+"\n")
    text.insert(END,"Splitting dataset into train & test. ML will user 80% dataset size for training and 20% for testing\n")
    text.insert(END,"Splitted training records = "+str(len(X_train))+"\n")
    text.insert(END,"Splitted testing records  = "+str(len(X_test))+"\n")

def KNN_NB_MLP():
    text.delete('1.0', END)
    if X_train is None:
        text.insert(END,"Please preprocess the dataset first.\n")
        return
    accuracy.clear()
    recall.clear()
    precision.clear()
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test)
    knn_precision = precision_score(y_test, predict,average='macro') * 100
    knn_recall = recall_score(y_test, predict,average='macro') * 100
    knn_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"KNN Precision : "+str(knn_precision)+"\n")
    text.insert(END,"KNN Recall    : "+str(knn_recall)+"\n")
    text.insert(END,"KNN Accuracy  : "+str(knn_acc)+"\n\n")
    accuracy.append(knn_acc)
    recall.append(knn_recall)
    precision.append(knn_precision)

    nb = BernoulliNB(binarize=0.0)
    nb.fit(X_train, y_train)
    predict = nb.predict(X_test)
    nb_precision = precision_score(y_test, predict,average='macro') * 100
    nb_recall = recall_score(y_test, predict,average='macro') * 100
    nb_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"Naive Bayes Precision : "+str(nb_precision)+"\n")
    text.insert(END,"Naive Bayes Recall    : "+str(nb_recall)+"\n")
    text.insert(END,"Naive Bayes Accuracy  : "+str(nb_acc)+"\n\n")
    accuracy.append(nb_acc)
    recall.append(nb_recall)
    precision.append(nb_precision)

    mlp = MLPClassifier(random_state=42, max_iter=300)
    mlp.fit(X_train, y_train)
    predict = mlp.predict(X_test)
    mlp_precision = precision_score(y_test, predict,average='macro') * 100
    mlp_recall = recall_score(y_test, predict,average='macro') * 100
    mlp_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"MLP Precision : "+str(mlp_precision)+"\n")
    text.insert(END,"MLP Recall    : "+str(mlp_recall)+"\n")
    text.insert(END,"MLP Accuracy  : "+str(mlp_acc)+"\n")
    accuracy.append(mlp_acc)
    recall.append(mlp_recall)
    precision.append(mlp_precision)

def SVM_DT_Ada():
    text.delete('1.0', END)
    if X_train is None:
        text.insert(END,"Please preprocess the dataset first.\n")
        return
    sv = svm.SVC()
    sv.fit(X_train, y_train)
    predict = sv.predict(X_test)
    svm_precision = precision_score(y_test, predict,average='macro') * 100
    svm_recall = recall_score(y_test, predict,average='macro') * 100
    svm_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"SVM Precision : "+str(svm_precision)+"\n")
    text.insert(END,"SVM Recall    : "+str(svm_recall)+"\n")
    text.insert(END,"SVM Accuracy  : "+str(svm_acc)+"\n\n")
    accuracy.append(svm_acc)
    recall.append(svm_recall)
    precision.append(svm_precision)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    dt_precision = precision_score(y_test, predict,average='macro') * 100
    dt_recall = recall_score(y_test, predict,average='macro') * 100
    dt_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"Decision Tree Precision : "+str(dt_precision)+"\n")
    text.insert(END,"Decision Tree Recall    : "+str(dt_recall)+"\n")
    text.insert(END,"Decision Tree Accuracy  : "+str(dt_acc)+"\n\n")
    accuracy.append(dt_acc)
    recall.append(dt_recall)
    precision.append(dt_precision)

    ada = AdaBoostClassifier(random_state=42)
    ada.fit(X_train, y_train)
    predict = ada.predict(X_test)
    ada_precision = precision_score(y_test, predict,average='macro') * 100
    ada_recall = recall_score(y_test, predict,average='macro') * 100
    ada_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"AdaBoost Precision : "+str(ada_precision)+"\n")
    text.insert(END,"AdaBoost Recall    : "+str(ada_recall)+"\n")
    text.insert(END,"AdaBoost Accuracy  : "+str(ada_acc)+"\n\n")
    accuracy.append(ada_acc)
    recall.append(ada_recall)
    precision.append(ada_precision)

def RF_NN():
    text.delete('1.0', END)
    if X_train is None:
        text.insert(END,"Please preprocess the dataset first.\n")
        return
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    rf_precision = precision_score(y_test, predict,average='macro') * 100
    rf_recall = recall_score(y_test, predict,average='macro') * 100
    rf_acc = accuracy_score(y_test,predict)*100
    text.insert(END,"Random Forest Precision : "+str(rf_precision)+"\n")
    text.insert(END,"Random Forest Recall    : "+str(rf_recall)+"\n")
    text.insert(END,"Random Forest Accuracy  : "+str(rf_acc)+"\n\n")
    accuracy.append(rf_acc)
    recall.append(rf_recall)
    precision.append(rf_precision)

    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    nn_model = Sequential()
    nn_model.add(Dense(512, input_shape=(X_train.shape[1],)))
    nn_model.add(Activation('relu'))
    nn_model.add(Dropout(0.3))
    nn_model.add(Dense(512))
    nn_model.add(Activation('relu'))
    nn_model.add(Dropout(0.3))
    nn_model.add(Dense(2))
    nn_model.add(Activation('softmax'))
    nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(nn_model.summary())
    acc_history = nn_model.fit(X_train, y_train1, epochs=10, validation_data=(X_test, y_test1))
    test_loss, test_accuracy = nn_model.evaluate(X_test, y_test1)
    accuracy.append(test_accuracy * 100)
    predict = nn_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test1, axis=1)
    nn_precision = precision_score(testY, predict,average='macro') * 100
    nn_recall = recall_score(testY, predict,average='macro') * 100
    recall.append(nn_recall)
    precision.append(nn_precision)
    text.insert(END,"Neural Network Precision : "+str(nn_precision)+"\n")
    text.insert(END,"Neural Network Recall    : "+str(nn_recall)+"\n")
    text.insert(END,"Neural Network Accuracy  : "+str(test_accuracy*100)+"\n")

def accuracyGraph():
    if len(accuracy) != 8:
        text.delete('1.0', END)
        text.insert(END,"Please run all three algorithm groups first.\n")
        return
    height = [accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]]
    bars = ('KNN ACC', 'NB ACC','MLP ACC','SVM ACC','Decision Tree ACC','AdaBoost ACC','Random Forest ACC','Neural Network ACC')
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(12,6))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars, rotation=45, ha='right')
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.tight_layout()
    plt.show()

def recallGraph():
    if len(recall) != 8:
        text.delete('1.0', END)
        text.insert(END,"Please run all three algorithm groups first.\n")
        return
    height = [recall[0],recall[1],recall[2],recall[3],recall[4],recall[5],recall[6],recall[7]]
    bars = ('KNN Recall', 'NB Recall','MLP Recall','SVM Recall','Decision Tree Recall','AdaBoost Recall','Random Forest Recall','Neural Network Recall')
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(12,6))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars, rotation=45, ha='right')
    plt.ylabel("Recall (%)")
    plt.title("Recall Comparison")
    plt.tight_layout()
    plt.show()

def precisionGraph():
    if len(precision) != 8:
        text.delete('1.0', END)
        text.insert(END,"Please run all three algorithm groups first.\n")
        return
    height = [precision[0],precision[1],precision[2],precision[3],precision[4],precision[5],precision[6],precision[7]]
    bars = ('KNN Precision', 'NB Precision','MLP Precision','SVM Precision','Decision Tree Precision','AdaBoost Precision','Random Forest Precision','Neural Network Precision')
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(12,6))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars, rotation=45, ha='right')
    plt.ylabel("Precision (%)")
    plt.title("Precision Comparison")
    plt.tight_layout()
    plt.show()

def close():
  main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Machine Learning for Email Spam Filtering', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload SpamBase Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

firstButton = Button(main, text="Run KNN, Naive Bayes & Multilayer Perceptron Algorithms", command=KNN_NB_MLP)
firstButton.place(x=480,y=100)
firstButton.config(font=font1)

secondButton = Button(main, text="Run SVM, Decision Tree & AdaBoost Algorithms", command=SVM_DT_Ada)
secondButton.place(x=940,y=100)
secondButton.config(font=font1)

thirdButton = Button(main, text="Run Random Forest & Neural Network", command=RF_NN)
thirdButton.place(x=10,y=150)
thirdButton.config(font=font1)

accButton = Button(main, text="Accuracy Comparison Graph", command=accuracyGraph)
accButton.place(x=300,y=150)
accButton.config(font=font1)

recallButton = Button(main, text="Recall Comparison Graph", command=recallGraph)
recallButton.place(x=560,y=150)
recallButton.config(font=font1)

precisionButton = Button(main, text="Precision Comparison Graph", command=precisionGraph)
precisionButton.place(x=840,y=150)
precisionButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)

main.config(bg='light coral')
main.mainloop()
