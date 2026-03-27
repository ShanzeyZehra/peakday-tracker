import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

st.title("PeakDay")

study_hours = st.slider("Hours studied",0,12)
sleep = st.slider("Hours slept",0,12)
productivity = st.slider("Productivity (1-10)",1,10)

file_name = "data.csv"

if st.button("Analyze & Save"):
    new_data = pd.DataFrame({
        "study_hours":[study_hours],
        "sleep":[sleep],
        "productivity":[productivity]
        })
    if os.path.exists(file_name):
        new_data.to_csv(file_name, mode="a",header=False,
                        index=False)
    else:
        new_data.to_csv(file_name,index=False)


    st.success("Data saved!")

if os.path.exists(file_name):
    df = pd.read_csv(file_name)

    st.subheader("Your Study Data")
    st.dataframe(df)

    st.subheader("Study Trends")
    st.line_chart(df)

    if len(df) > 3:
        X = df[["study_hours","sleep"]]
        y = df["productivity"]

        model = LinearRegression()
        model.fit(X,y)

        prediction = model.predict([[study_hours, sleep]])


        st.subheader("AI Prediction")
        st.write(f"Predicted Productivity:{prediction[0]:.2f}")
    else:
        st.info("Add more data (atleast 4 entries) to enable AI predictions.")

else:
    st.warning("No data yet. Start by adding your study information!")
                            
