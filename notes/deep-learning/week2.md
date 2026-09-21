## Dropout
- 训练过程中随机关闭一些神经元，使学习更分散，而不是集中在某几个神经元，减少过拟合，但是验证集的时候要用完整的神经网络

## epoch,batch,batch size
- epoch:训练的轮数
- batch:每次给模型的一批数据
- batch size:每个batch中有多少样本

## 训练集，验证集，测试集
- 训练集：用于训练模型的数据集，会更改模型参数，上面说的epoch,batch等都是用在训练集里面的，一般用` model.train() `表示开始训练
- 验证集：并不改变模型参数，用一些未参与训练的数据来评判模型的表现，` model.eval() `表示开始验证。验证阶段一般是一个epoch结束之后进行一次验证，也可以多个epoch之后验证一次，但要注意过拟合现象，或者说，验证集其实就是用来检验模型的学习的程度并防止一些过拟合现象的
- 一次训练与验证的完整过程如下：
```python
for epoch in range(num_epochs):
    model.train()#开始训练，可能会启动一些特殊的模式，比如Dropout等

    for X_batch,y_batch in train_loader:
        logits=model(X_batch)
        loss=loss_fn(logits,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()#开始验证，可能会关闭一些模式，改变某些层的工作模式，比如Dropout等

    with torch.no_gard():#关闭梯度计算
        for X_batch,y_batch in val_loader:
            logits=model(X_batch)
            val_loss=loss_fn(logits,y_batch)
 ```               
- 测试集：只在最后让模型见到一次，相当于期末考试，用来真正评估模型的能力

##  数据集的划分
- 训练集：70% ~ 80%
- 验证集：10% ~ 15%
- 测试集：10% ~ 15%
- 训练是我们同时设置了不同的模型，他们有不同的网络层数，不同的学习率等等，然后根据训练集和验证集的结果选择表现最好的模型，并用测试集进行最后的测试，但是如果出现数据泄露，比如：测试集数据混到训练集，根据测试集的结果去调整超参等，结果就不再可靠。所以，合理划分数据集，可以有效避免数据泄露。