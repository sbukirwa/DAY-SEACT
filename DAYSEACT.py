#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import time
import nltk
import numpy as np
import re, string
import sklearn.metrics as metrics
import datetime
from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize, sent_tokenize
from dateparser.search import search_dates
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('universal_tagset')
from datetime import datetime
string.punctuation = string.punctuation +"’"+"-"+"‘"+"-"

time_suspend = 1

dataWA = []
labelsWA = []

dataSS = []
labelsSS = []

dataSentiment = []
labelsSentiment = []

label_dir_sentiment = {
    "negative": "./sentiment/negative",
    "positive": "./sentiment/positive"
}

label_dir_summer_spring = {
    "summer": "./seasons/summer",
    "spring": "./seasons/spring"
}

label_dir_winter_autumn = {
    "winter": "./seasons/winter",
    "autumn": "./seasons/autumn"
}

def read_local_files_sentiment(label_dir_sentiment):
    for label in label_dir_sentiment.keys():
        for file_name in os.listdir(label_dir_sentiment[label]):
            filepath = f"{label_dir_sentiment[label]}/{file_name}"
            with open(filepath, mode='r', encoding='utf8', errors='ignore') as f:
                for line in f:
                    line = line.strip('\n')
                    dataSentiment.append(line)
                    labelsSentiment.append(label)
    return dataSentiment, labelsSentiment
  
# Reading the local files for winter autumn    
def read_local_files_winter_autumn(label_dir_winter_autumn):
    for label in label_dir_winter_autumn.keys():
        for file_name in os.listdir(label_dir_winter_autumn[label]):
            filepath = f"{label_dir_winter_autumn[label]}/{file_name}"
            with open(filepath, mode='r', encoding='utf8', errors='ignore') as f:
                for line in f:
                    line = line.strip('\n')
                    dataWA.append(line)
                    labelsWA.append(label)
    return dataWA, labelsWA
  
# Reading the local files for spring summer    
def read_local_files_spring_summer(label_dir_summer_spring):
    for label in label_dir_summer_spring.keys():
        for file_name in os.listdir(label_dir_summer_spring[label]):
            filepath = f"{label_dir_summer_spring[label]}/{file_name}"
            with open(filepath, mode='r', encoding='utf8', errors='ignore') as f:
                for line in f:
                    line = line.strip('\n')
                    dataSS.append(line)
                    labelsSS.append(label)
    return dataSS, labelsSS

dataWA,labelsWA = read_local_files_winter_autumn(label_dir_winter_autumn)
dataSS, labelsSS = read_local_files_spring_summer(label_dir_summer_spring)
dataSentiment,labelsSentiment = read_local_files_sentiment(label_dir_sentiment)


#Winter Activities
winter = {
    "Ice skating": "at the National Ice Centre. Link - https://www.national-ice-centre.com/",
    "Sledging": "at Bramcote Park. Link - https://www.broxtowe.gov.uk/for-you/parks-and-nature-conservation/parks-open-spaces/parks-in-my-area/bramcote-parks-and-open-spaces/",
    "Winter wonderland": "at Old Market Square. Link - https://nottinghamwinterwonderland.co.uk/"
}

#Summer Activities
summer = {
    "Swimming": "at Nottingham Swim Club. Link - https://www.nottinghamswimclub.com/",
    "Play miniature golf": "at the Lost city of Adventure Golf. Link - https://www.lostcityadventuregolf.com/nottingham/",
    "Build a castle": "at the beach at Nottingham Beach. Link - https://nottinghambeach.co.uk/"
}

#Spring Activities
spring = {
    "Picnic in the park": "at Nottingham Arboretum. Link - https://www.nottinghamshirewildlife.org/attenborough",
    "Go play golf": "at Gloryholes golf. Link - https://www.gloryholesgolfsheffield.co.uk/booknow",
    "Horse riding": "at Bassingfield Riding School. Link - http://www.bassingfieldriding.com/lessons/"
}

#Autumn Activities
autumn = {
    "Bird watching": "at Attenboro. Link - https://www.nottinghamshirewildlife.org/attenborough",
    "Go for a hike": "at Colwick Country Park. Link - https://nottsguidedwalks.co.uk/self-guided-walks/colwick-park-self-guided-walk/",
    "Jump in a pile of leaves": "in Bramwcote Parks. Link - https://www.broxtowe.gov.uk/for-you/parks-and-nature-conservation/parks-open-spaces/parks-in-my-area/bramcote-parks-and-open-spaces/"
}

#All activities
def all_activities():
    act={
    "Horse riding": "at Bassingfield Riding School. Link - http://www.bassingfieldriding.com/lessons/",
    "Swimming": "at Nottingham Swim Club. Link - https://www.nottinghamswimclub.com/",
    "Play miniature golf": "at the Lost city of Adventure Golf. Link - https://www.lostcityadventuregolf.com/nottingham/",
    "Ice skating": "at the National Ice Centre. Link - https://www.national-ice-centre.com/",
    }
    for key,value in act.items():
        print(" - ", key, value)
        time.sleep(time_suspend)

#Activities per season
def season_activities(season):
    print(season.title() + " activities:")
    if season.lower() == "autumn":
        for key,value in autumn.items():
            print(" - ", key, value)
            time.sleep(time_suspend)
    elif season.lower() == "spring":
        for key,value in spring.items():
            print(" - ", key, value)
            time.sleep(time_suspend)
    elif season.lower() == "summer":
        for key,value in summer.items():
            print(" - ", key, value)
            time.sleep(time_suspend)
    else:
        for key,value in winter.items():
            print(" - ", key, value)
            time.sleep(time_suspend)
    
#End conversation words
stoplist = ['bye', 'goodbye', 'stop']

#Appreciation words
appreciation = ['thanks', 'thank you']

#User Input
greeting_chabot_reply = "What is your name?"

#Yes Suggestions
yes_to_activities = ['yes','go', 'ahead','please','unfortunatley', 'do','unfortunately', 'fortunately','luckily','sure','thing', 'yeah', 'ok','ya', 'okay','oke','whatever','look','looks','like','it','seems','so','i','am','yup']

# Reply for greeting function
def reply_greeting(text):
    return (greeting_chabot_reply)
    
# Get person name function
def get_and_use_person_name(person_name_statement):
    global person_name
    py_stopword = set(stopwords.words('english'))
    s = list(py_stopword)
    s.append('name')
    s.append('call')
    py_stopword = set(s)
    py_txt = person_name_statement
    py_token = sent_tokenize(py_txt)
    for i in py_token:
        py_list_of_word = nltk.word_tokenize(i)
        py_list_of_word = [w for w in py_list_of_word if not w in py_stopword]
        py_list_of_word = [w for w in py_list_of_word if not w in string.punctuation + '.']
        py_tag = nltk.pos_tag(py_list_of_word)
        person_name = str(py_tag[0][0]).title()
        print("DAY-SEACT: Nice to meet you, "+ person_name +". How was your day?")
        continue_with_conversation_after_name(person_name)
        break

#function to classify sentiment                
def continue_with_conversation_after_name(person_name):
    user_input_user_day = input('\n' + person_name + ':')
    if user_input_user_day.lower() not in stoplist:
        user_input_user_day = user_input_user_day.lower()
        print(sentiment_classifier(dataSentiment, labelsSentiment, user_input_user_day, person_name))
    else:
        print("DAY-SEACT: Bye " + user_input_name)
        continue_convo = True

    #ASK ABOUT WEATHER function
    user_input_move_to_weather = input('\n' + person_name + ':')
    user_input_move_to_weather = user_input_move_to_weather.lower()
    find_out_about_the_weather(user_input_move_to_weather)
        
#Spring and summer classification function
def spring_summer_classifier(data, labels, query):
    global season_given
    X_train, X_test, y_train, y_test = train_test_split(data,labels,stratify=labels, test_size=0.25, random_state=42)
    count_vect = CountVectorizer(stop_words=stopwords.words('english'))
    X_train_counts=count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
    X_train_tf = tfidf_transformer.transform(X_train_counts)
    clf = LogisticRegression(random_state=0).fit(X_train_tf, y_train)
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    # print(confusion_matrix(y_test, predicted))
    # print(accuracy_score(y_test,predicted))
    # print(f1_score(y_test, predicted,pos_label="summer"))
    new_data = []
    new_data.append(query)
    processed_newdata = count_vect.transform(new_data)
    processed_newdata = tfidf_transformer.transform(processed_newdata)
    seasonFinal = clf.predict(processed_newdata)
    strFnal = "DAY-SEACT: Oh, you're in the " + seasonFinal[0].title() + " season. May I suggest some " + seasonFinal[0].title() + " activities?"
    season_given = seasonFinal[0]
    return strFnal

#Winter and autumn classification function
def winter_autumn_classifier(data, labels, query):
    global season_given
    X_train, X_test, y_train, y_test = train_test_split(data,labels,stratify=labels, test_size=0.25, random_state=42)
    count_vect = CountVectorizer(stop_words=stopwords.words('english'))
    X_train_counts=count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
    X_train_tf = tfidf_transformer.transform(X_train_counts)
    clf = LogisticRegression(random_state=0).fit(X_train_tf, y_train)
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    # print(confusion_matrix(y_test, predicted))
    # print(accuracy_score(y_test,predicted))
    # print(f1_score(y_test, predicted,pos_label="winter"))
    new_data = []
    new_data.append(query)
    processed_newdata = count_vect.transform(new_data)
    processed_newdata = tfidf_transformer.transform(processed_newdata)
    seasonFinal = clf.predict(processed_newdata)
    strFnal = "DAY-SEACT: Oh, you're in the " + seasonFinal[0].title() + " season. May I suggest some " + seasonFinal[0].title() + " activities?"
    season_given = seasonFinal[0]
    return strFnal

#Sentiment classification function
def sentiment_classifier(data, labels, query, person_name):
    global sentiment_given
    X_train, X_test, y_train, y_test = train_test_split(data,labels,stratify=labels, test_size=0.25, random_state=42)
    count_vect = CountVectorizer(stop_words=stopwords.words('english'))
    X_train_counts=count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
    X_train_tf = tfidf_transformer.transform(X_train_counts)
    clf = LogisticRegression(random_state=0).fit(X_train_tf, y_train)
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    # print(confusion_matrix(y_test, predicted))
    # print(accuracy_score(y_test,predicted))
    # print(f1_score(y_test, predicted,pos_label="positive"))
    new_data = []
    new_data.append(query)
    processed_newdata = count_vect.transform(new_data)
    processed_newdata = tfidf_transformer.transform(processed_newdata)
    sentimentFinal = clf.predict(processed_newdata)
    if((sentimentFinal[0].lower()) == 'negative'):
        strFnalSentiment = "DAY-SEACT: Sorry about that " + person_name + ". May I ask another question?" 
        sentiment_given = sentimentFinal[0]
    elif((sentimentFinal[0].lower()) == 'positive'):
        strFnalSentiment = "DAY-SEACT: I am glad it was a positive day " + person_name + ". May I ask another question?" 
    sentiment_given = sentimentFinal[0]
    return strFnalSentiment

#function to get user period for activity
def get_user_period(user_string_date):
    strPeriod = ""
    periodgiven = {}
    periodgiven = search_dates(user_string_date)
    if periodgiven != None:
        periodoutput = []
        for pdg in periodgiven:
            periodoutput.append(pdg)  
        strPeriod = "DAY-SEACT: The date will be " + str(periodoutput[0][1])
    else:
        print("DAY-SEACT: Sorry " + person_name + ", I didn't understand that. Please say that again.")
        user_input_suggest_again = input('\n' + person_name + ':')
        user_input_suggest_again = user_input_suggest_again.lower()
        get_user_period(user_input_suggest_again)
    return strPeriod

#function to get the user's response about the weather
def weather_function(user_input_longer_days):
    if user_input_longer_days.lower() not in stoplist:
        user_input_longer_days = user_input_longer_days.lower()
        user_input_longer_days_SPLIT = user_input_longer_days.split()
        for user_input_longer_days_split_done in user_input_longer_days_SPLIT:
            if user_input_longer_days_split_done in yes_to_activities:
                print("DAY-SEACT: Oh my! and how is the weather?")
                user_winter_autumn_reply = input('\n' + person_name + ':')
                user_winter_autumn_reply = user_winter_autumn_reply.lower()
                #winter or autumn response
                print(winter_autumn_classifier(dataWA, labelsWA, user_winter_autumn_reply))
                break
            else:
                print("DAY-SEACT: Oh my! and how is the weather?")
                user_spring_summer_reply = input('\n' + person_name + ':')
                user_spring_summer_reply = user_spring_summer_reply.lower()
                #summer or spring response
                print(spring_summer_classifier(dataSS, labelsSS, user_spring_summer_reply))
                break
    else:
        print("DAY-SEACT: Bye, thank you for your time " + person_name +".")
        continue_convo = True
            
    #Suggest season specific activities
    user_input_suggest_yes_no = input('\n' + person_name + ':')
    if user_input_suggest_yes_no.lower() not in stoplist:
        user_input_suggest_yes_no = user_input_suggest_yes_no.lower()
        user_input_suggest_yes_no_SLIPT = user_input_suggest_yes_no.split()
        for user_input_suggest_yes_act in user_input_suggest_yes_no_SLIPT:
            if user_input_suggest_yes_act in yes_to_activities:
                print("DAY-SEACT: Awesome! Here you go:")
                time.sleep(time_suspend)
                print('**************************************')
                season_activities(season_given)
                print('**************************************')
                time.sleep(time_suspend)
                activities_to_try_first()
                break
            else:
                print("DAY-SEACT: Bye, thank you for your time " + person_name +"." )
                continue_convo = True
    else:
        print("DAY-SEACT: Bye, thank you for your time " + person_name +"." )
        continue_convo = True

# function to find out about the weather and general activities output         
def find_out_about_the_weather(user_input_move_to_weather):
    if user_input_move_to_weather.lower() not in stoplist:
        user_input_move_to_weather = user_input_move_to_weather.lower()
        user_input_move_to_weather_SPLIT = user_input_move_to_weather.split()
        for user_input_move_to_weather_yes in user_input_move_to_weather_SPLIT:
            if user_input_move_to_weather_yes in yes_to_activities:
                print("DAY-SEACT: Alright. Are the days getting shorter, " + person_name + "?")
                user_input_longer_days = input('\n' + person_name + ':')
                user_input_longer_days = user_input_longer_days.lower()
                if user_input_longer_days not in stoplist:
                    #function for response if the days are getting shorter or longer
                    weather_function(user_input_longer_days)
                    break
                else:
                    print("DAY-SEACT: Bye, thank you for your time " + person_name +".")
                    continue_convo = True
            else:
                if(sentiment_given == 'negative'):
                    print("DAY-SEACT: The least I can do is recommend some activities to cheer you up. May I?")
                else:
                    print("DAY-SEACT: The least I can do is recommend some activities before this chat ends. May I?")
                user_input_activity_suggest_activity = input('\n' + person_name + ':')
                user_input_activity_suggest_activity = user_input_activity_suggest_activity.lower()
                if user_input_activity_suggest_activity not in stoplist:
                    user_input_activity_suggest_activity_SPLIT = user_input_activity_suggest_activity.split()
                    for user_input_activity_suggest_act in user_input_activity_suggest_activity_SPLIT:
                        if user_input_activity_suggest_act in yes_to_activities:
                            print("DAY-SEACT: Cool! There you go:")
                            time.sleep(time_suspend)
                            print('**************************************')
                            all_activities()
                            print('**************************************')
                            time.sleep(time_suspend)
                            activities_to_try_first()
                            break
                        else:
                            print("DAY-SEACT: Bye, thank you for your time " + person_name +"." )
                            continue_convo = True

                else:
                    print("DAY-SEACT: Bye, thank you for your time " + person_name +".")
                    continue_convo = True
    else:
        print("DAY-SEACT: Bye, thank you for your time " + person_name +".")
        continue_convo = True 
            
#function for Period when user will go for 
def activities_to_try_first():
    print("DAY-SEACT: Which activity will you try first?")
    user_input_activity = input('\n' + person_name + ':')
    if user_input_activity.lower() not in stoplist:
        print("DAY-SEACT: Great Choice! " + person_name + ". When will you go?")
        user_string_date = input('\n' + person_name + ':')
        print(get_user_period(user_string_date.lower()))
        print("DAY-SEACT: Enjoy, " + person_name)
    else:
        print("DAY-SEACT: Bye, thank you for your time " + person_name +".")
        continue_convo = True

#function to show current time when  display greeting depending on time    
def show_current_time():
    global greeting
    current_dateTime = datetime.now()
    current_time = "Current time: " + time.strftime("%H:%M:%S")
    if(current_dateTime.hour >= 0 and current_dateTime.hour <= 11):
        greeting = "Good morning"
    elif(current_dateTime.hour >= 12 and current_dateTime.hour <=16):
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"
        
    return current_time
        


# CHATBOT CONVERSATION
continue_convo = False
print("\nI am DAY-SEACT, I will suggest ACTivities you can do based on your DAY or SEAson you're in :)\n")
print(show_current_time())
while not continue_convo:
    #Greetings     
    user_input_greeting = input("DAY-SEACT: " + greeting + " to you.\n>>>")
    if user_input_greeting.lower() not in stoplist:
        user_input_greeting = user_input_greeting.lower()
        print("DAY-SEACT: "+ reply_greeting(user_input_greeting))
        
        #Get person Name
        user_input_name = input('\n>>>')
        if user_input_name.lower() not in stoplist:
            user_input_name = user_input_name.lower()
            get_and_use_person_name(user_input_name)
            continue_convo = True
        else:
            print("DAY-SEACT: Bye")
            continue_convo = True
    else:
        print("DAY-SEACT: Bye")
        continue_convo = True

    


