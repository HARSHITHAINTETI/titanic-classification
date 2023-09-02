import streamlit as st
import os
import pandas as pd
import joblib as jb
heading_style = '''
<div style="color:red;" align='center'>
<h1>Titanic-Dataset</h1>
</div>
'''
def return_df(PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked):
    kbn={
    'PassengerId':[PassengerId],
	'Pclass':[Pclass],
    'Name':[Name],
    'Sex':[Sex],
	'Age':[Age],
	'SibSp':[SibSp],
    'Parch':[Parch],
    'Ticket':[Ticket],
    'Fare':[Fare],
    'Cabin':[Cabin],
	'Embarked':[Embarked]
    }   
    final_df=pd.DataFrame(kbn)
    return final_df
def base_model():
    bmodel=jb.load(os.path.join('finalized_model.pkl'))
    return bmodel
st.markdown(heading_style, unsafe_allow_html=True)
PassengerId=st.number_input('passengerId',min_value=0)
Pclass=st.slider('Pclass',1,2,3)
Name=st.text_input('enter your name')
Sex=st.selectbox('Select your gender',['Male','Female'])
Age=st.number_input('age', min_value=0)
SibSp=st.number_input('enter the SibSp number', min_value=0)
Parch=st.number_input('enter the parch number',min_value=0)
Ticket=st.number_input(' enter the Ticket number',min_value=0)
Fare=st.number_input('enter the Fare cost',min_value=0)
Cabin=st.selectbox('select the cabin number',['C85','C123','B42','C148'])
Embarked=st.selectbox('select the embarked id',['S','C','Q'])
df=return_df(PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)
if st.button('Submit'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	if predictions==1:
		st.write('Survived	')
	elif predictions==0:
		st.write(' Not Survived')

