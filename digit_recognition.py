# digit_recognition.py

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


digits = datasets.load_digits()


plt.imshow(digits.images[3], cmap='gray')
plt.title(f"Digit Label: {digits.target[3]}")
plt.axis('off')
plt.show()



X = digits.data  
y = digits.target 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"\n Model Accuracy on Test Data: {accuracy * 100:.2f}%\n")


import random
#random.randint(0, len(X_test) - 1)
index = 2
plt.matshow(digits.images[index])
plt.title("Predicted Digit")
plt.show()

predicted = model.predict([X_test[index]])
print("🔍 Predicted digit:", predicted[0])
print("🧾 Actual digit   :", y_test[index])
