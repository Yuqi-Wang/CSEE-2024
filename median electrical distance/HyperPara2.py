import numpy as np
from skopt import BayesSearchCV
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras
import config

initial_num = config.get_initial_num()
# Load the initial training data
f_set = np.load('3f_train_ws_woPC_OLTC_'+str(initial_num)+'.npy')
w_set = np.load('3w_train_ws_woPC_OLTC_'+str(initial_num)+'.npy') # all the three weights are given in original file


# Split the dataset into train and test sets
w_train, w_test, f_train, f_test = train_test_split(w_set, f_set, test_size=0.2)
np.save('3f_Bay_train_ws_woPC_OLTC_'+'.npy',f_train)
np.save('3w_Bay_train_ws_woPC_OLTC_'+'.npy',w_train)
np.save('3f_Bay_test_ws_woPC_OLTC_'+'.npy',f_test)
np.save('3w_Bay_test_ws_woPC_OLTC_'+'.npy',w_test)

# Function to create model, required for KerasClassifier
def create_model(num_layers=1, neurons1=32, neurons2=32, neurons3=32, neurons4=32, neurons5=32, neurons6=32, neurons7=32, neurons8=32, neurons9=32, loss_fun='mean_squared_error',learn_rate=0.01):
    model = keras.Sequential()
    model.add(keras.layers.Dense(neurons1, input_dim=3, activation='relu'))
    
    # Add additional hidden layers based on num_layers
    if num_layers > 1:
        model.add(keras.layers.Dense(neurons2, activation='relu'))
    if num_layers > 2:
        model.add(keras.layers.Dense(neurons3, activation='relu'))
    if num_layers > 3:
        model.add(keras.layers.Dense(neurons4, activation='relu'))
    if num_layers > 4:
        model.add(keras.layers.Dense(neurons5, activation='relu'))
    if num_layers > 5:
        model.add(keras.layers.Dense(neurons6, activation='relu'))
    if num_layers > 6:
        model.add(keras.layers.Dense(neurons7, activation='relu'))
    if num_layers > 7:
        model.add(keras.layers.Dense(neurons8, activation='relu'))
    if num_layers > 8:
        model.add(keras.layers.Dense(neurons9, activation='relu'))
    
    model.add(keras.layers.Dense(3, activation='linear'))
    model.compile(loss=loss_fun, optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate))
    return model

# Wrap Keras model so it can be used by scikit-learn
neural_network = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, epochs=10, batch_size=100, verbose=0)

# Define search space
param_space = {
    'num_layers': (1, 2),
    'neurons1': (1, 500),
    'neurons2': (1, 500),
    'neurons3': (1, 500),
    'neurons4': (1, 500),
    'neurons5': (1, 500),
    'neurons6': (1, 500),
    'neurons7': (1, 500),
    'neurons8': (1, 500),
    'neurons9': (1, 500),
    'batch_size': (1, 300),
    'epochs': (5, 50),
    'loss_fun': ['mean_squared_error'],
    'learn_rate': (0.005,0.05)
}
# , 'mean_absolute_error', 'logcosh', 'huber_loss'

# Use BayesSearchCV
opt = BayesSearchCV(
    neural_network,
    param_space,
    n_iter=100,
    random_state=42,
    verbose=0
)

# Fit the model
opt.fit(w_train, f_train)

# Predict and evaluate the results
f_trained = opt.predict(w_test)
f1_diff = np.max(np.abs(f_trained[:,0] - f_test[:,0]))
f2_diff = np.max(np.abs(f_trained[:,1] - f_test[:,1]))
f3_diff = np.max(np.abs(f_trained[:,2] - f_test[:,2]))

print(f"Best parameters found: {opt.best_params_}")
print('f1_diff: ',f1_diff)
print('f2_diff: ',f2_diff)
print('f3_diff: ',f3_diff)

