import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import oracledb
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Database functions for Oracle integration
def get_db_connection():
    """Create and return a database connection using environment variables"""
    try:
        connection = oracledb.connect(
            user=os.getenv("ORACLE_USER"),
            password=os.getenv("ORACLE_PASSWORD"),
            dsn=os.getenv("ORACLE_DSN")
        )
        print("✅ Successfully connected to Oracle Database")
        return connection
    except oracledb.Error as error:
        print(f"❌ Error connecting to Oracle Database: {error}")
        raise

def fetch_feedback_from_oracle():
    """Fetch feedback data from Oracle database"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        query = """
            SELECT FED_ECODE, USER_REMARKS
            FROM MAYUR_FEEDBACK
            WHERE USER_REMARKS IS NOT NULL
        """
        cursor.execute(query)
        
        # Fetch column names
        column_names = [col[0] for col in cursor.description]
        
        rows = cursor.fetchall()
        
        if not rows:
            print("No feedback found in the database.")
            return pd.DataFrame(columns=column_names)
        
        # Convert to DataFrame with explicit column names
        df = pd.DataFrame(rows, columns=column_names)
        
        return df
    
    except oracledb.Error as e:
        print(f"Detailed Database error: {e}")
        error, = e.args
        print(f"Oracle-Error-Code: {error.code}")
        print(f"Oracle-Error-Message: {error.message}")
        return pd.DataFrame(columns=['FED_ECODE', 'USER_REMARKS'])
    finally:
        if connection:
            connection.close()

def save_analysis_to_oracle(results):
    """Save sentiment analysis results back to Oracle database"""
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        
        # Create the sentiment_analysis table with necessary columns if it doesn't exist
        cursor.execute("""
            BEGIN
                EXECUTE IMMEDIATE 'CREATE TABLE sentiment_analysis (
                    sno NUMBER PRIMARY KEY,
                    fed_ecode NVARCHAR2(10),
                    user_remarks VARCHAR2(4000),
                    sentiment VARCHAR2(20),
                    confidence NUMBER,
                    secondary_categories VARCHAR2(200),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )';
            EXCEPTION
                WHEN OTHERS THEN
                    IF SQLCODE != -955 THEN
                        RAISE;
                    END IF;
            END;
        """)

        # Check existing records and get the maximum SNO
        cursor.execute("SELECT NVL(MAX(sno), 0) FROM sentiment_analysis")
        max_sno = cursor.fetchone()[0]

        # Check existing records in sentiment_analysis to avoid duplicates
        existing_feedbacks_query = """
            SELECT fed_ecode FROM sentiment_analysis
        """
        cursor.execute(existing_feedbacks_query)
        existing_feedbacks = {row[0] for row in cursor.fetchall()}
        
        # Filter out results that already exist in the database
        new_results = [r for r in results if r['feedback_id'] not in existing_feedbacks]
        
        if not new_results:
            print("All feedbacks have already been analyzed.")
            return
        
        # Insert new results into the sentiment_analysis table with incremented SNO
        insert_query = """
            INSERT INTO sentiment_analysis 
            (sno, fed_ecode, user_remarks, sentiment, confidence, 
            secondary_categories, timestamp)
            VALUES (:1, :2, :3, :4, :5, :6, :7)
        """
        
        data = [(
            max_sno + idx + 1,  # Generate new SNO by incrementing from max existing SNO
            r['feedback_id'],
            r['feedback'],
            r['sentiment'],
            r['confidence'],
            r['secondary_categories'],
            r['timestamp']
        ) for idx, r in enumerate(new_results)]
        
        cursor.executemany(insert_query, data)
        connection.commit()
        print(f"{len(new_results)} new results saved to database successfully")
        
    finally:
        if connection:
            connection.close()

# Load the pre-trained DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def encode_texts(texts):
    return tokenizer(
        texts,
        add_special_tokens=True,
        max_length=128,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Identify text and label columns
    text_column = None
    label_column = None
    
    for column in data.columns:
        if data[column].dtype == 'object' and len(data[column].unique()) > 10:
            text_column = column
        elif data[column].dtype == 'object' and len(data[column].unique()) <= 10:
            label_column = column
    
    if text_column is None or label_column is None:
        raise ValueError("Could not identify appropriate text and label columns")
    
    print(f"Using '{text_column}' as text column and '{label_column}' as label column")
    
    # Handle missing values
    data = data.dropna(subset=[label_column])
    
    # Encode labels
    le = LabelEncoder()
    data[label_column] = le.fit_transform(data[label_column])
    
    return data, text_column, label_column, le

# Load your dataset (example placeholder)
data, TEXT_COLUMN, LABEL_COLUMN, label_encoder = load_and_preprocess_data('new_corrected_reviews.csv')

num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
model.to(device)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data[TEXT_COLUMN], data[LABEL_COLUMN], test_size=0.2, random_state=42
)

# Encode the texts
train_encodings = encode_texts([str(text) for text in X_train.tolist()])
test_encodings = encode_texts(X_test.tolist())

# Convert the labels to tensors
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], y_train)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop for fine-tuning DistilBERT
def train_model(epochs=0):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch_input_ids, batch_attention_mask, batch_labels = [item.to(device) for item in batch]

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs} completed. Average loss: {total_loss / len(train_loader):.4f}')
    
    model.save_pretrained('./distilbert-feedback-model')
    tokenizer.save_pretrained('./distilbert-feedback-tokenizer')
# Evaluate the model
def evaluate_model():
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch_input_ids, batch_attention_mask, batch_labels = [item.to(device) for item in batch]

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            correct_predictions += (predictions == batch_labels).sum().item()
            total_predictions += len(batch_labels)

    accuracy = correct_predictions / total_predictions
    print(f"Test accuracy: {accuracy:.2f}")

# Function to predict feedback category of a single text
def predict_feedback(text):
    model.eval()
    inputs = encode_texts([text])
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)

    predicted_label = label_encoder.inverse_transform([prediction.item()])[0]
    confidence = probabilities.max().item()

    # Check for multiple categories
    threshold = 0.3  # Adjust this threshold as needed
    categories = []
    for i, prob in enumerate(probabilities[0]):
        if prob > threshold:
            categories.append(label_encoder.inverse_transform([i])[0])

    return predicted_label, confidence, categories

# Modified main block to include Oracle integration
if __name__ == '__main__':
    print("Training the DistilBERT model...")
    train_model()

    print("Evaluating the model...")  # Add code for evaluation if required
    evaluate_model()

    # Add new option for Oracle database processing
    while True:
        print("\nChoose an option:")
        print("1. Process feedback from Oracle database")
        print("2. Test with sample feedback")
        print("3. Interactive testing")
        print("4. Quit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
                try:
                    print("Fetching feedback from Oracle database...")
                    oracle_data = fetch_feedback_from_oracle()  # Do NOT overwrite this
                    
                    if oracle_data.empty:
                        print("No data found in the feedback table.")
                        continue
                    
                    results = []
                    for index, row in oracle_data.iterrows():
                        sentiment, confidence, categories = predict_feedback(row['USER_REMARKS'])
                        results.append({
                            'feedback_id': row['FED_ECODE'],
                            'feedback': row['USER_REMARKS'],
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'secondary_categories': ', '.join(categories),
                            'timestamp': datetime.now()
                        })

                    # Save results to Oracle database
                    save_analysis_to_oracle(results)
                    print("Sentiment analysis results saved to Oracle.")

                except Exception as e:
                    print(f"Error processing Oracle feedback: {e}")
        
        elif choice == '2':
            # Test the model with custom feedback
            print("\nTesting the model with new feedback:")
            test_feedback = [
                "The product exceeded my expectations. Great purchase!",
                "Delivery was late and the item was damaged. Very disappointing.",
                "How do I return this item?",
                "I suggest adding more color options.",
                "The product is okay, but nothing special.",
                "Excellent quality, but you should improve the packaging.",
            ]

            for feedback in test_feedback:
                label, confidence, categories = predict_feedback(feedback)  # Modified this line
                print(f"Feedback: {feedback}")
                print(f"Predicted category: {label} (Confidence: {confidence:.2f})")
                if len(categories) > 1:
                    print(f"Multiple categories detected: {', '.join(categories)}")
                print()  # Add a blank line between feedback items
        
        elif choice == '3':
            # Your existing interactive testing code
            while True:
                user_feedback = input("Enter your feedback (or 'quit' to exit): ")
                if user_feedback.lower() == 'quit':
                    break
                label, confidence, categories = predict_feedback(user_feedback)
                print(f"Predicted category: {label} (Confidence: {confidence:.2f}, Categories: {categories})")
        
        elif choice == '4':
            break
        else:
            print("Invalid choice, please try again.")
