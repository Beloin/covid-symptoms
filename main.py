# This is a sample Python script.
import argparse
from typing import List, Union, Dict
from os import system
from tensorflow import keras
import numpy as np


def clear_console(): return system('clear')


valid_regular_symptoms = ['Fever', 'Tiredness',
                          'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat', 'None_Sympton']
valid_more_symptoms = ['Pains', 'Nasal-Congestion',
                       'Runny-Nose', 'Diarrhea', 'None_Experiencing']
valid_age = ['Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+']
valid_gender = ['Gender_Male', 'Gender_Female', 'Gender_Transgender']
valid_contact = ['Contact_Yes', 'Contact_No', 'Contact_Dont-Know']

symptoms = None
more_symptoms = None
age = None
gender = None
contact = None

pred_values = {'Fever': 0,
               'Tiredness': 0,
               'Dry-Cough': 0,
               'Difficulty-in-Breathing': 0,
               'Sore-Throat': 0,
               'None_Sympton': 0,
               'Pains': 0,
               'Nasal-Congestion': 0,
               'Runny-Nose': 0,
               'Diarrhea': 0,
               'None_Experiencing': 0,
               'Age_0-9': 0,
               'Age_10-19': 0,
               'Age_20-24': 0,
               'Age_25-59': 0,
               'Age_60+': 0,
               'Gender_Female': 0,
               'Gender_Male': 0,
               'Gender_Transgender': 0,
               'Contact_Dont-Know': 0,
               'Contact_No': 0,
               'Contact_Yes': 0
               }

severity_map = {0: 'Severity_Mild', 1: 'Severity_Moderate',
                2: 'Severity_Severe', 3: 'Severity_None'}


def print_choices(arr):
    for i in range(len(arr)):
        print(i, '-', arr[i])


def get_age(age: int):
    va = ''
    if age < 10:
        va = 'Age_0-9'
    if 10 <= age < 20:
        va = 'Age_10-19'
    if 20 <= age < 25:
        va = 'Age_20-24'
    if 25 <= age < 60:
        va = 'Age_25-59'
    if 60 <= age:
        va = 'Age_60+'

    return va


def get_item(value: Union[int, List[Union[int, str]]], value_arr=List[str]):
    if type(value) is list:
        to_ret = []
        for i in value:
            getter = i
            if type(getter) is str:
                getter = int(getter)

            to_ret.append(value_arr[getter])
        return to_ret

    return value_arr[value]


def change_pred_values(item: Union[str, List[str]], pred_values: Dict):
    if type(item) is list:
        for i in item:
            if pred_values[i] is not None:
                pred_values[i] = 1

    else:
        pred_values[item] = 1


def get_model():
    return keras.models.load_model('my_model.h5')


def main():
    clear_console()
    print('Welcome to Covid Severity Check (With 25% of efficience).\nPress "ENTER" to continue...')
    input()
    print('First lets study about your Symptoms')

    # Regular Symptoms
    print('\nWhich of those symptoms you have?')
    print_choices(valid_regular_symptoms)
    print('\nUse spaces to separate numbers.')
    symptoms = get_item(input().strip().split(' '), valid_regular_symptoms)

    clear_console()

    # More Symptoms
    print('Now, do you feel any more Symptom?')
    print_choices(valid_more_symptoms)
    print('\nUse spaces to separate numbers.')
    more_symptoms = get_item(input().strip().split(' '), valid_more_symptoms)
    clear_console()

    # Age
    print('Now, tell me your age. Dont be shy!')
    age = int(input())
    age = get_age(age)
    clear_console()

    # Gender
    print(
        "To keep going, we need to identify your gender. "
        "(Please don't have a dysphoria)"
    )
    print_choices(valid_gender)
    print('\nUse spaces to separate numbers.')
    gender = get_item(int(input().strip()), valid_gender)
    clear_console()

    # Contact
    print('Have you had any contact with someone accused to hace Covid-19?')
    print_choices(valid_contact)
    contact = get_item(int(input().strip()), valid_contact)

    change_pred_values(symptoms, pred_values)
    change_pred_values(more_symptoms, pred_values)
    change_pred_values(age, pred_values)
    change_pred_values(gender, pred_values)
    change_pred_values(contact, pred_values)

    to_pred = np.array(list(pred_values.values())).reshape((1, -1))
    preds = get_model().predict(to_pred)
    print(severity_map[np.argmax(preds)])


if __name__ == '__main__':
    main()
