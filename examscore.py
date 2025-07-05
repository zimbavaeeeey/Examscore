import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("data.csv")
x=df[['HoursStudied']]
y=df['ExamScore']
x_train, x_test, y_train, y_test, = train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
st.title("ExamScore predictor")
st.write("Enter hours studied to predict the exam score")
hours=st.number_imput("Hours Studied:",min_value=0.0,step=0.1)
if st.button("Predict Score"):
    predicted_score =model.predict([[hours]])[0]
    st.success(f"predicted score:(predicted_score:.2f)")
    st.write("###sample training data")
    st.dataframe(df)
