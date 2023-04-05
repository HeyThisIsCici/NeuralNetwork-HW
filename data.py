# # 数据导入和预处理

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# MNIST数据导入
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据初步处理
X_train = X_train / 255.
X_test = X_test / 255.

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

y_train = np.eye(10)[y_train.astype(int)]
y_test = np.eye(10)[y_test.astype(int)]