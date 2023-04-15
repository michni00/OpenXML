import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_nn(n_layers, n_nodes, activation, optimizer, learning_rate):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=X_scaled.shape[1]))
    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(n_nodes, activation=activation))
    model.add(tf.keras.layers.Dense(y_encoded.shape[1], activation='softmax'))

    model.compile(optimizer=optimizer(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_scaled, y_encoded, epochs=20, validation_split=0.2, batch_size=128)

    return history

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
df = pd.read_csv(url, compression='gzip', header=None)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print('Logistic Regression accuracy:', lr_accuracy)
print('Decision Tree accuracy:', dt_accuracy)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1,1))

n_layers = [1]
n_nodes = [32]
activations = ['relu']
optimizers = [tf.keras.optimizers.Adam]
learning_rates = [0.01]
# n_layers = [1, 2]
# n_nodes = [32, 64]
# activations = ['relu', 'tanh']
# optimizers = [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD]
# learning_rates = [0.01, 0.001]

best_accuracy = 0
for nl in n_layers:
    for nn in n_nodes:
        for act in activations:
            for opt in optimizers:
                for lr in learning_rates:
                    print(f'Training model with {nl} layers, {nn} nodes, {act} activation, {opt.__name__} optimizer, and {lr} learning rate')
                    history = train_nn(nl, nn, act, opt, lr)
                    val_acc = history.history['val_accuracy'][-1]
                    if val_acc > best_accuracy:
                        best_accuracy = val_acc
                        best_history = history
                        best_hyperparams = {'n_layers': nl,
                                            'n_nodes': nn,
                                            'activation': act,
                                            'optimizer': opt.__name__,
                                            'learning_rate': lr}

plt.plot(best_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(best_history.history['loss'], label='Training Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training Curves')
plt.legend()
plt.show()