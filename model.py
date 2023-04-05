# # 初始化网络参数

# In[1]:


# 定义每层基本属性
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 10

# 初始化参数
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))


# # 定义激活函数、损失函数、前向传播、反向传播算法

# In[5]:


# 激活函数
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# 基于L2正则化定义交叉熵损失函数
def cross_entropy_loss(y_pred, y_true, reg_lambda):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / m
    loss += reg_lambda / (2 * m) * (np.sum(W1**2) + np.sum(W2**2))
    return loss

# 前向传播
def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    return a2

# 反向传播
def backward(X, y_true, y_pred, a1, reg_lambda):
    m = y_true.shape[0]
    
    dL_dz2 = (y_pred - y_true) / m
    dL_dW2 = np.dot(a1.T, dL_dz2)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
 
    dL_da1 = np.dot(dL_dz2, W2.T)
    dL_dz1 = dL_da1 * (a1 > 0)
    dL_dW1 = np.dot(X.T, dL_dz1) + reg_lambda / m * W1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
    
    return dL_dW1, dL_db1, dL_dW2, dL_db2


# # 定义超参数并开始训练和测试

# In[6]:


# 定义超参数
learning_rate = 0.01
num_epochs = 1000
batch_size = 128
reg_lambda = 0.01