import torch
from torch import nn, optim
import torch.utils.data as Data
import numpy as np
from model import TextRNN
from cnews_loader import read_vocab, read_category, process_file

train_data_path = 'cnews.train.txt'
test_data_path = 'cnews.test.txt'
val_data_path = 'cnews.val.txt'
vocab_path = 'cnews.vocab.txt'

train_epochs = 1000
batch_size = 256
lr = 0.001

#########################################
# 获取文本的类别及其对应id的字典
categories, cat_to_id = read_category()
print(categories)
# 获取训练文本中所有出现过的字及其所对应的id
words, word_to_id = read_vocab('cnews.vocab.txt')

vocab_size = len(words)

# 数据加载及分批
# 获取训练数据每个字的id和对应标签的one-hot形式
x_train, y_train = process_file('cnews.train.txt', word_to_id, cat_to_id, 600)
x_test, y_test = process_file('cnews.val.txt', word_to_id, cat_to_id, 600)

cuda = torch.device('cuda')
x_train, y_train = torch.LongTensor(x_train), torch.LongTensor(y_train)
x_test, y_test = torch.LongTensor(x_test), torch.LongTensor(y_test)

train_dataset = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
#################################################################################



def train(lr, train_loader, test_dataset):
    model = TextRNN().cuda()
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0

    for epoch in range(train_epochs):
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x, y = x_batch.cuda(), y_batch.cuda()

            # FF
            y_pred = model(x)
            loss = loss_fn(y_pred, y)


            # BF
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = np.mean((torch.argmax(y_pred,1) == torch.argmax(y,1)).cpu().numpy())
        print('Training epoch {:}, loss = {:}, acc = {:}'.format(epoch + 1, loss.item(), acc))

        if (epoch+1)%5 == 0:
            for step, (x_batch, y_batch) in enumerate(test_loader):
                x, y = x_batch.cuda(), y_batch.cuda()

                # FF
                y_pred = model(x)
                acc = np.mean((torch.argmax(y_pred, 1) == torch.argmax(y, 1)).cpu().numpy())
                # print('Test acc = {:}'.format(acc))
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), 'model_params.pkl')


train(lr, train_loader, (x_test, y_test))


