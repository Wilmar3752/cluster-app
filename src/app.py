import gradio as gr
import pandas as pd
import joblib

data = pd.read_csv('data/raw/CC GENERAL.csv', index_col='CUST_ID')
pipeline = joblib.load('./models/cluster_pipeline.joblib')

def sentence_builder(BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
       INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
       ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
       CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
       CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT,
       TENURE):
    ls = [BALANCE, BALANCE_FREQUENCY, PURCHASES, ONEOFF_PURCHASES,
       INSTALLMENTS_PURCHASES, CASH_ADVANCE, PURCHASES_FREQUENCY,
       ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY,
       CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX, PURCHASES_TRX,
       CREDIT_LIMIT, PAYMENTS, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT,
       TENURE]
    ls_df = pd.DataFrame(ls).T
    ls_df.columns = data.columns
   
    df_transform = pipeline.transform(ls_df)
    cluster = df_transform['label'][0]
    return f'El individuo pertenece al cluster {cluster}'



demo = gr.Interface(
    sentence_builder,
     [gr.Number(value=data[column][0]) for column in data.columns],
    "text")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0:$PORT")
