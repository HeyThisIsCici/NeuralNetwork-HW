# 定义存储损失和精度数据的数组
train_losses = []
test_losses = []
train_accs = []
test_accs = []

# 训练&测试
for epoch in range(num_epochs):
    # Shuffle the training data
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    y_train = y_train[indices]

    # 训练模型
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # 前向传播
        y_pred = forward(X_batch)

        # 计算训练集损失和精度
        train_loss = cross_entropy_loss(y_pred, y_batch, reg_lambda)
        train_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

        # 反向传播计算梯度
        dL_dW1, dL_db1, dL_dW2, dL_db2 = backward(X_batch, y_batch, y_pred, relu(np.dot(X_batch, W1) + b1), reg_lambda)

        # 梯度下降更新权重和偏置
        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2

    # 测试集测试
    y_test_pred = forward(X_test)

    # 计算测试集损失和精度
    test_loss = cross_entropy_loss(y_test_pred, y_test, reg_lambda)
    test_acc = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))

    # 输出当前训练和测试的损失和精度
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 存储损失和精度数据
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# 存储模型
np.savez("mnist_model.npz", W1=W1, b1=b1, W2=W2, b2=b2)


# # 可视化模型表现

# In[13]:


# 训练和测试的loss曲线
x = np.arange(len(train_losses))
plt.title("Loss Curve")
plt.plot(x, train_losses, label='train loss')
plt.plot(x, test_losses, label='test loss', linestyle='--')
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig('loss_image.jpg')
plt.show()


# In[14]:


# 测试的accuracy曲线
plt.title("Accuracy Curve")
plt.plot(x, test_accs, label='test accuracy')
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.savefig('accuracy_image.jpg')
plt.show()

