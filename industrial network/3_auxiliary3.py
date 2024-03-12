import pandas as pd
import shap
from keras import models
import config
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
import easygui

relax = config.get_relax()
MSE = config.get_MSE()
k = config.get_k()
MAE = config.get_MAE()
diff = config.get_diff()
test_prop = config.get_test_prop()
test_new = config.get_test_new()
num_iterations = config.get_num_iterations()
diff_max = config.get_diff_max()
initial_num = config.get_initial_num()
testing_num = config.get_testing_num()
BatchSize = config.get_BatchSize()
Epochs = config.get_Epochs()
LearningRate = config.get_LearningRate()

# Load the final training data
f_train_final = np.load('f_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
w_train_final = np.load('w_train_iteration_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.npy')
# load the model
model_3D = models.load_model('M_GK_'+str(initial_num)+'_iter_'+str(num_iterations)+'_MSE_'+str(MSE)+'_MAE_'+str(MAE)+'_diff_'+str(diff)+'.h5')

y_predict_final = model_3D.predict(w_train_final)
print('the expectation is: ', y_predict_final.sum(0)/len(y_predict_final))

feature_cols = [
    'w2',
    'w1',
    'w3'
]
label_cols = [
    'f2',
    'f1',
    'f3'
]
df_features = pd.DataFrame(data=w_train_final,columns=feature_cols)
df_labels = pd.DataFrame(data=f_train_final,columns=label_cols)

# Create the list of all labels for the drop down list
list_of_labels = df_labels.columns.to_list()

# Create a list of tuples so that the index of the label is what is returned
tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))

explainer = shap.KernelExplainer(model=model_3D.predict, data=df_features, link="identity")
w_train_final_shap = explainer.shap_values(X=df_features)

# Create a widget for the labels and then display the widget
current_label = widgets.Dropdown(options=tuple_of_labels, value=0, description="Select Label:")
shap.summary_plot(shap_values=w_train_final_shap[current_label.value], features=df_features)

current_label = widgets.Dropdown(options=tuple_of_labels, value=1, description="Select Label:")
shap.summary_plot(shap_values=w_train_final_shap[current_label.value], features=df_features)

current_label = widgets.Dropdown(options=tuple_of_labels, value=2, description="Select Label:")
shap.summary_plot(shap_values=w_train_final_shap[current_label.value], features=df_features)

exam_x = []
num_sample = 1000
for i in range(num_sample):
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1-x1)
    x3 = 1-x1-x2
    exam_x.append([x1,x2,x3])
exam_x = np.array(exam_x)
exam_y = model_3D.predict(exam_x)

df_features = pd.DataFrame(data=exam_x,columns=feature_cols)
df_labels = pd.DataFrame(data=exam_y,columns=label_cols)
exam_x_shap = explainer.shap_values(X=df_features)

current_label = widgets.Dropdown(options=tuple_of_labels, value=0, description="Select Label:")
shap.summary_plot(shap_values=exam_x_shap[current_label.value], features=df_features)

current_label = widgets.Dropdown(options=tuple_of_labels, value=1, description="Select Label:")
shap.summary_plot(shap_values=exam_x_shap[current_label.value], features=df_features)

current_label = widgets.Dropdown(options=tuple_of_labels, value=2, description="Select Label:")
shap.summary_plot(shap_values=exam_x_shap[current_label.value], features=df_features)

exam_x = []
num_sample = 10000
for i in range(num_sample):
    x1 = random.uniform(0,1)
    x2 = random.uniform(0,1-x1)
    x3 = 1-x1-x2
    exam_x.append([x1,x2,x3])
exam_x = np.array(exam_x)
exam_y = model_3D.predict(exam_x)

df_features = pd.DataFrame(data=exam_x,columns=feature_cols)
df_labels = pd.DataFrame(data=exam_y,columns=label_cols)
exam_x_shap = explainer.shap_values(X=df_features)

current_label = widgets.Dropdown(options=tuple_of_labels, value=0, description="Select Label:")
shap.summary_plot(shap_values=exam_x_shap[current_label.value], features=df_features)

current_label = widgets.Dropdown(options=tuple_of_labels, value=1, description="Select Label:")
shap.summary_plot(shap_values=exam_x_shap[current_label.value], features=df_features)

current_label = widgets.Dropdown(options=tuple_of_labels, value=2, description="Select Label:")
shap.summary_plot(shap_values=exam_x_shap[current_label.value], features=df_features)
