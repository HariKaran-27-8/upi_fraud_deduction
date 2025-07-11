
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import datetime
from datetime import datetime as dt
import time
import base64
import pickle
from xgboost import XGBClassifier

"""
# Welcome to your own UPI Transaction Fraud Detector!

You have the option of inspecting a single transaction by adjusting the parameters below OR you can even check 
multiple transactions at once by uploading a .csv file in the specified format
"""

# Load and display the reference dataset
df = pd.read_csv("balanced_upi_data.csv")
fraud_count = df[df['fraud'] == 1].shape[0]
nonfraud_count = df[df['fraud'] == 0].shape[0]
ratio = nonfraud_count / fraud_count if fraud_count != 0 else 1

#st.write(f"Imbalance ratio (non-fraud/fraud): {ratio:.2f}")

pickle_file_path = "UPI Fraud Detection.pkl"
# Load the saved XGBoost model from the pickle file
loaded_model = pickle.load(open(pickle_file_path, 'rb'))

# If you ever retrain, use:
# model = XGBClassifier(scale_pos_weight=ratio)

tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment', 'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']

tran_date = st.date_input("Select the date of your transaction", datetime.date.today())
if tran_date:
    selected_date = dt.combine(tran_date, dt.min.time())
    month = selected_date.month
    year = selected_date.year

tran_type = st.selectbox("Select transaction type", tt)
pmt_gateway = st.selectbox("Select payment gateway", pg)
tran_state = st.selectbox("Select transaction state", ts)
merch_cat = st.selectbox("Select merchant category", mc)

amt = st.number_input("Enter transaction amount", step=0.1)

st.write("OR")
st.write("CSV Format:", df)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV:", df)

button_clicked = st.button("Check transaction(s)")
st.markdown(
    """
    <style>
    .stButton>button {
        position: fixed;
        bottom: 40px;
        left: 413px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if button_clicked:
    if uploaded_file is not None:
        with st.spinner("Checking transactions..."):
            def download_csv():
                csv = df.to_csv(index=False, header=True)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download Output CSV</a>'
                return href

            # Prepare features
            df[['Month', 'Year']] = df['Date'].str.split('-', expand=True)[[1, 2]]
            df[['Month', 'Year']] = df[['Month', 'Year']].astype(int)
            df.drop(columns=['Date'], inplace=True)
            df = df.reindex(columns=['Amount', 'Year', 'Month', 'Transaction_Type', 'Payment_Gateway', 'Transaction_State', 'Merchant_Category'])

            results = []
            for index, row in df.iterrows():
                # One-hot encoding for each categorical variable
                tt_oh = [0] * len(tt)
                pg_oh = [0] * len(pg)
                ts_oh = [0] * len(ts)
                mc_oh = [0] * len(mc)

                input = []
                input.append(row['Amount'])
                input.append(row['Year'])
                input.append(row['Month'])

                if row['Transaction_Type'] in tt:
                    tt_oh[tt.index(row['Transaction_Type'])] = 1
                if row['Payment_Gateway'] in pg:
                    pg_oh[pg.index(row['Payment_Gateway'])] = 1
                if row['Transaction_State'] in ts:
                    ts_oh[ts.index(row['Transaction_State'])] = 1
                if row['Merchant_Category'] in mc:
                    mc_oh[mc.index(row['Merchant_Category'])] = 1

                input = input + tt_oh + pg_oh + ts_oh + mc_oh
                prediction = loaded_model.predict([input])[0]
                results.append(prediction)

            df['fraud'] = results
            st.success("Checked transactions!")
            st.markdown(download_csv(), unsafe_allow_html=True)

    else:
        with st.spinner("Checking transaction(s)..."):
            tt_oh = [0] * len(tt)
            pg_oh = [0] * len(pg)
            ts_oh = [0] * len(ts)
            mc_oh = [0] * len(mc)

            if tran_type in tt:
                tt_oh[tt.index(tran_type)] = 1
            if pmt_gateway in pg:
                pg_oh[pg.index(pmt_gateway)] = 1
            if tran_state in ts:
                ts_oh[ts.index(tran_state)] = 1
            if merch_cat in mc:
                mc_oh[mc.index(merch_cat)] = 1

            input = []
            input.append(amt)
            input.append(year)
            input.append(month)
            input = input + tt_oh + pg_oh + ts_oh + mc_oh
            result = loaded_model.predict([input])[0]
            st.success("Checked transaction!")
            if result == 0:
                st.write("Congratulations! Not a fraudulent transaction.")
            else:
                st.write("Oh no! This transaction is fraudulent.")

        # prob = loaded_model.predict_proba([input])[0]
        # st.write(f"Prediction Probabilities: {prob}")