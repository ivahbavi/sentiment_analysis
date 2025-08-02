import streamlit as st
import pandas as pd
import plotly.express as px
import oracledb
from ultimate_sen_new import train_model, evaluate_model
import os
from dotenv import load_dotenv

load_dotenv()

# Function to get Oracle DB connection
def get_db_connection():
    """Create and return a database connection using env variables, Streamlit-friendly"""
    try:
        connection = oracledb.connect(
            user=os.getenv("ORACLE_USER"),
            password=os.getenv("ORACLE_PASSWORD"),
            dsn=os.getenv("ORACLE_DSN")
        )
        return connection
    except oracledb.Error as error:
        st.error(f"❌ Error connecting to Oracle Database: {error}")
        return None

# Function to fetch data from sentiment_analysis table
def fetch_sentiment_analysis_data():
    connection = get_db_connection()
    if connection is None:
        return pd.DataFrame()
    try:
        cursor = connection.cursor()
        query = """
            SELECT fed_ecode, user_remarks, sentiment, confidence, timestamp
            FROM sentiment_analysis
            ORDER BY timestamp DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        return pd.DataFrame(rows, columns=columns)
    except oracledb.Error as e:
        st.error(f"Error fetching data from database: {e}")
        return pd.DataFrame()
    finally:
        connection.close()

# Placeholder functions for model training and evaluation
def training():
    train_model()
    st.info("Train model logic executed.")

def evaluating():
    evaluate_model()
    st.info("Evaluate model logic executed.")

# Streamlit application
st.title("Sentiment Analysis Dashboard")

# Add Train and Evaluate Model Buttons
st.sidebar.title("Model Operations")

if st.sidebar.button("Train Model"):
    with st.spinner("Training the model, please wait..."):
        try:
            training()
            st.success("Model training completed successfully!")
        except Exception as e:
            st.error(f"Error during training: {e}")

if st.sidebar.button("Evaluate Model"):
    with st.spinner("Evaluating the model, please wait..."):
        try:
            evaluating()
            st.success("Model evaluation completed successfully!")
        except Exception as e:
            st.error(f"Error during evaluation: {e}")

# Fetch data from the database
df = fetch_sentiment_analysis_data()

if not df.empty:
    # Display the top 10 latest feedbacks
    st.subheader("Latest 10 Feedbacks")
    latest_feedbacks = df.tail(10)
    st.write(latest_feedbacks)

    # Visualization of sentiment distribution
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['SENTIMENT'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        title="Sentiment Distribution",
        labels={'Count': 'Number of Feedbacks', 'Sentiment': 'Sentiment'},
        color='Sentiment'
    )
    st.plotly_chart(fig)
else:
    st.warning("No data available in the sentiment_analysis table.")
