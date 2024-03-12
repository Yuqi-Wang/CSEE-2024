import numpy as np
from skopt import BayesSearchCV
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras

# Load dataset
mnist = fetch_openml('mnist_784', version=1)
X = np.array(mnist.data.astype('float32'))
y = keras.utils.to_categorical(np.array(mnist.target.astype('int')))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', num_layers=1, neurons1=32, neurons2=32, neurons3=32):
    model = keras.Sequential()
    model.add(keras.layers.Dense(neurons1, input_dim=784, activation='relu'))
    
    # Add additional hidden layers based on num_layers
    if num_layers > 1:
        model.add(keras.layers.Dense(neurons2, activation='relu'))
    if num_layers > 2:
        model.add(keras.layers.Dense(neurons3, activation='relu'))
    
    model.add(keras.layers.Dense(3, activation='linear'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Wrap Keras model so it can be used by scikit-learn
neural_network = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=10, batch_size=100, verbose=0)

# Define search space
param_space = {
    'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adam'],
    'num_layers': (1, 3),
    'neurons1': (10, 100),
    'neurons2': (10, 100),
    'neurons3': (10, 100),
    'batch_size': (10, 100),
    'epochs': (5, 15)
}

# Use BayesSearchCV
opt = BayesSearchCV(
    neural_network,
    param_space,
    n_iter=10,
    random_state=42,
    verbose=0
)

# Fit the model
opt.fit(X_train, y_train)

# Predict and evaluate the results
y_pred = opt.predict(X_test)
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)

print(f"Best parameters found: {opt.best_params_}")
print(f"Accuracy with best parameters: {accuracy:.2f}")
