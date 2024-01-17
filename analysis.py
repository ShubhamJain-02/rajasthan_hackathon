import pandas as pd
data = [
    [0.6, 1, 0, 1, 1],
    [0.7, 2, 1, 1, 0],
    [0.5, 3, 2, 1, 0],
    [0.8, 1, 0, 1, 1],
    [0.9, 2, 1, 1, 0],
    [0.7, 3, 2, 1, 0],
    [0.6, 1, 0, 1, 1],
    [0.8, 2, 1, 1, 0],
    [0.9, 3, 2, 1, 0],
    [0.7, 1, 0, 1, 1],
    [0.6, 2, 1, 1, 0],
    [0.8, 3, 2, 1, 0],
    [0.7, 1, 0, 1, 1],
    [0.6, 2, 1, 1, 0],
    [0.9, 3, 2, 1, 0],
    [0.8, 1, 0, 1, 1],
    [0.7, 2, 1, 1, 0],
    [0.6, 3, 2, 1, 0],
    [0.7, 1, 0, 1, 1],
    [0.9, 2, 1, 1, 0],
    [0.8, 3, 2, 1, 0],
    [0.6, 1, 0, 1, 1],
    [0.7, 2, 1, 1, 0],
    [0.9, 3, 2, 1, 0],
    [0.8, 1, 0, 1, 1],
    [0.6, 2, 1, 1, 0],
    [0.7, 3, 2, 1, 0],
    [0.9, 1, 0, 1, 1],
    [0.8, 2, 1, 1, 0],
    [0.6, 3, 2, 1, 0],
    [0.7, 1, 0, 1, 1],
    [0.9, 2, 1, 1, 0],
    [0.8, 3, 2, 1, 0],
    [0.6, 1, 0, 1, 1],
    [0.7, 2, 1, 1, 0],
    [0.9, 3, 2, 1, 0],
    [0.8, 1, 0, 1, 1],
    [0.6, 2, 1, 1, 0],
    [0.7, 3, 2, 1, 0],
]
df=pd.DataFrame(data,columns=['reaction_time','num_attempts','num_wrong','num_correct','target'])
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
new_reaction_time = float(input('Enter reaction time (0-1): '))
new_num_attempts = int(input('Enter number of attempts: '))
new_num_wrong_answers = int(input('Enter number of wrong answers: '))
new_num_correct_answers = int(input('Enter number of correct answers: '))

new_data = [[new_reaction_time, new_num_attempts, new_num_wrong_answers, new_num_correct_answers]]

prediction = model.predict(new_data)

if prediction[0] == 0:
    print('FAIL')
else:
    print('PASS')
