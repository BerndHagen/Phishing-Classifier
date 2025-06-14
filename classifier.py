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

model_filename = 'classifier.pkl'
feedback_data_filename = 'feedback.csv'

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
X_train, X_test, y_train, y_test = train_test_split(data[['Message', 'Length']], data['Label'], test_size=0.2,
                                                    random_state=42)

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

plt.figure(figsize=(5, 5))
plt.style.use('default')

fig = plt.gcf()
fig.canvas.manager.window.resizable(False, False)
fig.canvas.manager.set_window_title('Confusion Matrix')


def center_window():
    mngr = fig.canvas.manager
    mngr.window.update_idletasks()

    screen_width = mngr.window.winfo_screenwidth()
    screen_height = mngr.window.winfo_screenheight()

    window_width = mngr.window.winfo_reqwidth()
    window_height = mngr.window.winfo_reqheight()

    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    mngr.window.wm_geometry(f"{window_width}x{window_height}+{x}+{y}")


fig.canvas.mpl_connect('draw_event', lambda event: center_window())

fig.patch.set_facecolor('#f8f9fa')

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

colors = ['#3498db', '#2980b9', '#1e3a8a']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('blue_gradient', colors, N=n_bins)

ax = sns.heatmap(conf_matrix,
                 annot=True,
                 fmt='g',
                 cmap=cmap,
                 cbar=False,
                 square=True,
                 linewidths=2,
                 linecolor='#ffffff',
                 annot_kws={"size": 18, "weight": "bold", "color": "#ffffff"},
                 vmin=0,
                 vmax=conf_matrix.max())

ax.set_xlabel('Predicted Labels', fontsize=10, fontweight='bold', color='#2c3e50')
ax.set_ylabel('True Labels', fontsize=10, fontweight='bold', color='#2c3e50')
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

ax.set_xticklabels(['0', '1'], fontsize=12, color='#2c3e50')
ax.set_yticklabels(['0', '1'], fontsize=12, color='#2c3e50', rotation=0)

ax.set_facecolor('#f8f9fa')

ax.tick_params(colors='#2c3e50', which='both')

plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
plt.show()

print("Classification Report:\n", class_report)


class ReviewClassifications(tk.Tk):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.index = 0
        self.title("Phishing Classification Review System")
        self.geometry("900x610")
        self.minsize(900, 610)
        self.configure(bg='#f8f9fa')
        self.resizable(False, False)
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 900) // 2
        y = (screen_height - 610) // 2 - 50
        self.geometry(f"900x610+{x}+{y}")

        header_frame = Frame(self, bg='#2c3e50')
        header_frame.pack(fill='x')

        title_label = Label(header_frame, text="Classification Review System",
                            bg='#2c3e50', fg='white', font=('Segoe UI', 18, 'bold'))
        title_label.pack(pady=20)
        main_frame = Frame(self, bg='#f8f9fa')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        message_section = Frame(main_frame, bg='#ffffff', relief='solid', bd=1)
        message_section.pack(fill='x', pady=(0, 15))

        message_header = Frame(message_section, bg='#ecf0f1')
        message_header.pack(fill='x')

        Label(message_header, text="Message Content", bg='#ecf0f1', fg='#2c3e50',
              font=('Segoe UI', 12, 'bold')).pack(anchor='w', padx=20,
                                                  pady=15)
        message_content_frame = Frame(message_section, bg='#ffffff')
        message_content_frame.pack(fill='x', padx=20, pady=20)

        self.message_label = Label(message_content_frame,
                                   bg='#ffffff', fg='#555555', font=('Segoe UI', 11, 'italic'),
                                   relief='flat', bd=0,
                                   justify='center', anchor='center',
                                   wraplength=800, height=3)
        self.message_label.pack(fill='both', expand=True)

        prediction_section = Frame(main_frame, bg='#ffffff', relief='solid', bd=1)
        prediction_section.pack(fill='x', pady=(0, 15))

        prediction_header = Frame(prediction_section, bg='#e8f4fd')
        prediction_header.pack(fill='x')

        Label(prediction_header, text="Classification Result", bg='#e8f4fd', fg='#2c3e50',
              font=('Segoe UI', 12, 'bold')).pack(anchor='w', padx=20, pady=15)

        prediction_content_frame = Frame(prediction_section, bg='#ffffff')
        prediction_content_frame.pack(fill='x', padx=20, pady=20)

        self.prediction_label = Label(prediction_content_frame, text="", bg='#ffffff', fg='#2c3e50',
                                      font=('Segoe UI', 13, 'bold'))
        self.prediction_label.pack()

        feedback_section = Frame(main_frame, bg='#ffffff', relief='solid', bd=1)
        feedback_section.pack(fill='x')

        feedback_header = Frame(feedback_section, bg='#fff3cd')
        feedback_header.pack(fill='x')

        Label(feedback_header, text="Validation Feedback", bg='#fff3cd', fg='#2c3e50',
              font=('Segoe UI', 12, 'bold')).pack(anchor='w', padx=20, pady=15)

        feedback_content = Frame(feedback_section, bg='#ffffff')
        feedback_content.pack(fill='x', padx=20, pady=20)

        self.status_label = Label(feedback_content, text="",
                                  bg='#ffffff', fg='#2c3e50', font=('Segoe UI', 11))
        self.status_label.pack(pady=(0, 15))
        button_frame = Frame(feedback_content, bg='#ffffff')
        button_frame.pack()

        correct_button = Button(button_frame, text="CORRECT", command=lambda: self.save_feedback(True),
                                font=('Segoe UI', 11, 'bold'), bg='#27ae60', fg='white',
                                relief='flat', bd=0, padx=40, cursor='hand2',
                                activebackground='#229954', width=18)
        correct_button.config(height=1)
        correct_button.pack(side='left', padx=(0, 15))

        incorrect_button = Button(button_frame, text="INCORRECT", command=lambda: self.save_feedback(False),
                                  font=('Segoe UI', 11, 'bold'), bg='#e74c3c', fg='white',
                                  relief='flat', bd=0, padx=40, cursor='hand2',
                                  activebackground='#c0392b', width=18)
        incorrect_button.config(height=1)
        incorrect_button.pack(side='left')

        self.update_message()

    def update_message(self):
        email_content, email_length = self.data[self.index]
        prediction = best_pipeline.predict(pd.DataFrame({'Message': [email_content], 'Length': [email_length]}))
        status = "Phishing Detected!" if prediction[0] == 1 else "No Threat Detected"
        status_color = "#e74c3c" if prediction[0] == 1 else "#27ae60"

        max_chars = 200
        if len(email_content) > max_chars:
            display_text = email_content[:max_chars - 3] + "..."
        else:
            display_text = email_content

        self.message_label.config(text=display_text)

        self.prediction_label.config(text=status, fg=status_color)

        self.status_label.config(text="Did the system classify this message correctly?")

    def save_feedback(self, is_correct):
        email_content, email_length = self.data[self.index]
        prediction = best_pipeline.predict(pd.DataFrame({'Message': [email_content], 'Length': [email_length]}))
        prediction_status = "Phishing Detected" if prediction[0] == 1 else "No Threat Detected"
        feedback_status = "Correct" if is_correct else "Incorrect"
        print(f"{email_content[:60]}... => {prediction_status} (User Feedback: {feedback_status})")

        if not is_correct:
            original_label = 'ham' if prediction[0] == 1 else 'spam'
            label = 0 if original_label == 'ham' else 1
            new_entry = pd.DataFrame(
                {'Category': [original_label], 'Message': [email_content], 'Length': [email_length], 'Label': [label]})
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