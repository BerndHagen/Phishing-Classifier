import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tkinter as tk
from tkinter import Label, Button, messagebox, Frame, Text

model_filename = 'spam_classifier.pkl'
feedback_data_filename = 'feedback_data.csv'

feedback_data = pd.DataFrame()

def load_data():
    global feedback_data
    if not os.path.exists(model_filename):
        data = pd.read_csv('dataset.csv')
    else:
        data = pd.read_csv('dataset.csv')
        if os.path.exists(feedback_data_filename):
            feedback_data = pd.read_csv(feedback_data_filename)
    data['Length'] = data['Message'].str.len()
    data['Label'] = data['Category'].map({'spam': 1, 'ham': 0})
    if not feedback_data.empty:
        feedback_data['Length'] = feedback_data['Message'].str.len()
        data = pd.concat([data, feedback_data[['Message', 'Length', 'Label']]], ignore_index=True)
    return data

data = load_data()
X_train, X_test, y_train, y_test = train_test_split(data[['Message', 'Length']], data['Label'], test_size=0.2, random_state=42)

column_transform = ColumnTransformer([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=1.0, min_df=0.001), 'Message'),
    ('length', StandardScaler(), ['Length'])
])

pipeline = Pipeline([
    ('transform', column_transform),
    ('classifier', RandomForestClassifier(random_state=42))
])

parameters = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20],
}
search = GridSearchCV(pipeline, parameters, cv=5)
search.fit(X_train, y_train)
best_pipeline = search.best_estimator_
joblib.dump(best_pipeline, model_filename)

y_pred = best_pipeline.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

plt.figure(figsize=(10, 7))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=cmap, cbar=False, annot_kws={"size": 16})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", class_report)

class ReviewClassifications(tk.Tk):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.index = 0
        self.title("Classification Review")
        self.geometry("400x350")
        self.configure(bg='#333745')
        frame = Frame(self, bg='#333745')
        frame.pack(expand=True, fill='both', padx=20, pady=20)
        self.message_label = Text(frame, height=6, wrap='word', bg='#282A36', fg='#FFFFFF', font=('Arial', 12))
        self.message_label.pack(pady=10, fill='x')
        self.prediction_label = Label(frame, text="", fg="#8BE9FD", bg='#333745', font=('Arial', 14, 'bold'))
        self.prediction_label.pack(pady=10)
        status_label = Label(frame, text="Was this message correctly classified?", bg='#333745', fg='#F8F8F2', font=('Arial', 12))
        status_label.pack(pady=10)
        correct_button = Button(frame, text="Correct", command=lambda: self.save_feedback(True), font=('Arial', 12), bg='#50FA7B')
        correct_button.pack(fill='x', expand=True, padx=20, pady=5)
        incorrect_button = Button(frame, text="Incorrect", command=lambda: self.save_feedback(False), font=('Arial', 12), bg='#FF5555')
        incorrect_button.pack(fill='x', expand=True, padx=20, pady=5)
        self.update_message()

    def update_message(self):
        email_content, email_length = self.data[self.index]
        prediction = best_pipeline.predict(pd.DataFrame({'Message': [email_content], 'Length': [email_length]}))
        status = "Phishing Detected!" if prediction[0] == 1 else "No Threat Detected"
        self.message_label.delete('1.0', tk.END)
        self.message_label.insert(tk.END, email_content)
        self.prediction_label.config(text=f"Prediction: {status}")

    def save_feedback(self, is_correct):
        email_content, email_length = self.data[self.index]
        prediction = best_pipeline.predict(pd.DataFrame({'Message': [email_content], 'Length': [email_length]}))
        prediction_status = "Phishing Detected" if prediction[0] == 1 else "No Threat Detected"
        feedback_status = "Correct" if is_correct else "Incorrect"
        print(f"{email_content[:60]}... => {prediction_status} (User Feedback: {feedback_status})")

        if not is_correct:
            original_label = 'ham' if prediction[0] == 1 else 'spam'
            label = 0 if original_label == 'ham' else 1
            new_entry = pd.DataFrame({'Category': [original_label], 'Message': [email_content], 'Length': [email_length], 'Label': [label]})
            global feedback_data
            feedback_data = pd.concat([feedback_data, new_entry], ignore_index=True)
            feedback_data.to_csv(feedback_data_filename, index=False)
        self.next_message()

    def next_message(self):
        if self.index < len(self.data) - 1:
            self.index += 1
            self.update_message()
        else:
            if not feedback_data.empty and len(feedback_data) > 5:
                new_X = feedback_data[['Message', 'Length']]
                new_y = feedback_data['Label']
                best_pipeline.fit(new_X, new_y)
                joblib.dump(best_pipeline, model_filename)
            messagebox.showinfo("End", "All messages have been reviewed")
            self.destroy()

if __name__ == "__main__":
    examples = [
        "Just landed in New York. I'll give you a call once I'm settled in at the hotel. Maybe we can meet up for dinner if you would like? Let me know if you're around.",
        "Get €1000 for free as a welcome bonus for trading cryptocurrencies. Sign up here: crypto-bonus.de. Immediate action is required to claim this offer.",
        "Urgent security update available for your macOS. Protect your device by updating immediately at mac-security-update.apple-services.com. Immediate updating is crucial for your device's security.",
        "Don't forget about the weekly team meeting via the usual link. Looking forward to connecting with everyone and tackling this week's agenda! Hope to see you all there.",
        "We would like to inform you that we have shipped your order. Your shipment is now on its way; changes by you or our customer service are no longer possible. If you would like to return an item from your order or view or change other orders, you can easily do so via my orders on our website Amazon.de."
    ]
    data = [(message, len(message)) for message in examples]
    app = ReviewClassifications(data)
    app.mainloop()