import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load Dataset
# -----------------------------
df = pd.read_csv("C:/Users/HP/Downloads/archive (11)/Match Dataset.csv")
print("DataSet:",df) 
print("Columns:",df.columns)
print("Preprocessing:")
print(df.isnull().sum())
df.dropna()

# -----------------------------
# 2) Select Relevant Columns
# -----------------------------
features = ['team1', 'team2', 'venue', 'city', 'toss_winner', 'toss_decision']
target = 'winner'

# Keep only selected columns
df = df[features + [target]]

# -----------------------------
# 3) One Hot Encoding with get_dummies()
# -----------------------------
X = pd.get_dummies(df[features])
y = df[target]

# -----------------------------
# 4) Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5) Train Model
# -----------------------------
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# -----------------------------
# 6) Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7) Accuracy and Metrics
# -----------------------------
print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8) Probability
# -----------------------------
# Get the test predictions (winners)
test_predictions = model.predict(X_test)

# Convert predictions to actual team names
predicted_winners = []

for i in range(len(test_predictions)):
    # If prediction is 1 -> team1 predicted to win
    if test_predictions[i] == 1:
        predicted_winners.append(df['team1'].iloc[X_test.index[i]])
    else:
        predicted_winners.append(df['team2'].iloc[X_test.index[i]])

# Convert to DataFrame
pred_df = pd.DataFrame(predicted_winners, columns=['predicted_winner'])

# Count wins
win_counts = pred_df['predicted_winner'].value_counts()

# Convert to probabilities
win_probability = win_counts / win_counts.sum()

print("\nCorrect WORLD CUP Winning Probability:\n")
print(win_probability)

# -----------------------------
# 9) Visualization
# -----------------------------

# Sort probabilities for better visualization
win_probability_sorted = win_probability.sort_values(ascending=False)

plt.figure(figsize=(12, 6)) # Set a good figure size
win_probability_sorted.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Predicted Winning Probability per Team in Test Set Matches')
plt.xlabel('Team')
plt.ylabel('Winning Probability')
plt.xticks(rotation=45, ha='right') 
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.tight_layout() 
plt.show()



