## 一、项目介绍

>在深度学习领域中，目标检测一直是一个备受关注的研究方向。为了更深入地理解深度学习目标检测的原理和实现，我写了一个简单的单目标检测项目。在这个项目中，
> 我用最简单的方式实现了数据迭代器、网络模型、预测脚本和训练模型脚本，以及一些辅助脚本，通过这个过程提高对目标检测的认识和实践能力。

> 项目地址：https://github.com/78778443/QingNet

数据集地址: 链接: https://pan.baidu.com/s/1Sv45FBxYUhCQioVN6qmuTg?pwd=4ijk 提取码: 4ijk  
![2cb275c57191210c34c09f65c48a7a9](https://github.com/78778443/QingNet/assets/8509054/3bb5cfda-08d3-4bec-86e0-d61b6e61f80c)


### 1.1 项目概要
要实现目标检测系统，离不开数据加载器，网络模型，训练脚本，预测脚本这四大项；

1. 数据加载器的作用是将数据集加载出来，并将数据集的标注数据给格式化，便于后续训练；
2. 网络模型的主要作用是提取网络特征，比如你给一张图，他把图里面的特征信息提取并返回给你；
3. 训练脚本的主要作用是桥接数据集和网络模型，通常是给模型一个图片，模型返回特征结果后，对结果进行偏差(损失)计算；
4. 预测脚本的主要作用是训练好一个模型(权重)后，将模型(权重)文件用于实际生产；

### 1.2 项目结构
这个项目的结构相对简单，主要涉及以下几个文件：

* data.py: 数据迭代器，负责加载和处理训练和测试数据。
* net.py: 网络模型的定义，包括卷积层、激活函数以及输出标签、位置、排序和置信度的信息。
* train.py: 训练模型的脚本，包括数据加载、模型训练、损失函数计算、优化器更新等过程。
* predict.py: 预测脚本，用训练好的模型进行单张图像的预测。
* tools.py: 辅助脚本，用于可视化预测结果。


### 1.3 项目运行
在运行项目时，只需执行`python train.py`命令即可。
如果缺少相关依赖包，可以通过使用pip进行安装。

```bash
python train.py 
train_loss 0===>> 0.8435055017471313
train_loss 10===>> 0.8142958283424377
train_loss 40===>> 0.8188565373420715
test_loss 0===>> 0.8148629665374756
test_loss 10===>> 0.8028237223625183
sort_acc 14==>> tensor(0.0397)
train_loss 0===>> 0.8068220615386963
```

## 二、数据集处理
在做这个项目之前，要准备一批数据集，我将数据集文件放在data文件夹下,文件名里面包含图片序号，是否有目标，目标的四个坐标点，并用逗号隔开
```zsh
(base) ➜ tree data 
data
├── test
    ├── 1.0.0.0.0.0.0.jpg
    ├── 1.1.163.54.290.181.6.jpg
    ├── ....过长省略......
└── train
    ├── 1000.0.0.0.0.0.0.jpg
    ├── 1000.1.64.90.229.255.8.jpg
    ├── ....过长省略......
```

### 2.1 数据加载

在项目里我写了一个自定义的`QingDataset`类来加载和处理训练和测试数据。首先，在初始化方法中，我遍历了指定目录下的所有文件名，并将它们拼接到数据集列表中：

```python
def __init__(self, root):
    self.dataset = []
    for filename in os.listdir(root):
        self.dataset.append(os.path.join(root, filename))
```

这样，`self.dataset`中存储了所有图像文件的路径。

### 2.2 数据处理

在`__getitem__`方法中，我通过读取图像数据，对其进行归一化处理，并转换为PyTorch张量：

```python
def __getitem__(self, index):
    img_path = self.dataset[index]
    img = cv2.imread(img_path)
    img = img / 255
    img = torch.tensor(img).permute(2, 0, 1)
    data_list = img_path.split('.')
    label = int(data_list[1])
    position = [int(i) / 300 for i in data_list[2:6]]
    sort = int(data_list[6]) - 1

    return np.float32(img), np.float32(label), np.float32(position), sort, img_path
```

这里我将图像进行了归一化处理，并从文件名中提取了标签、位置和排序信息。最后，返回了处理后的图像数据以及相应的标签、位置、排序和图像路径。

## 三、神经网络模型

>网络模型这里的`nn.Conv2d(3, 16, 3)`,`ReLU`,`MaxPool2d`里面的参数是我随意填写的，读者不用纠结参数的含义。
### 3.1 模型结构

在`net.py`中，我定义了神经网络模型`QingNet`。该模型的结构采用了`Sequential`容器，通过堆叠卷积层、激活函数以及池化层来提取图像特征：


```python
self.layers = nn.Sequential(
    nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(3),
    nn.Conv2d(16, 22, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(22, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 82, 3), nn.ReLU(),
    nn.Conv2d(82, 128, 3), nn.ReLU(),
    nn.Conv2d(128, 25, 3), nn.ReLU()
)
```
这些层的输出形成了模型的最后特征图。

### 3.2 输出信息
模型的最后几个层分别输出了标签、位置、排序和置信度的信息：
```python
self.label_layer = nn.Sequential(nn.Conv2d(25, 1, 3), nn.ReLU())
self.position_layers = nn.Sequential(nn.Conv2d(25, 4, 3), nn.ReLU())
self.sort_layers = nn.Sequential(nn.Conv2d(25, 20, 3), nn.ReLU())
self.confidence_layer = nn.Sequential(nn.Conv2d(25, 1, 3), nn.Sigmoid())
```

这些输出对应了单目标检测任务中所需的各个要素。

## 四、训练过程

>训练的过程其实就是将数据集丢给网络模型，网络模型会返回目标的位置信息，我会那这个结果与数据集的正确结果进行损失计算，并告诉网络模型损失值。

随着不断训练网络模型，网络模型会越来越靠近真实值，每训练一轮我都会把权重文件保存到磁盘中，这样电脑即使重启还可以接着上次的成果接着训练。

### 4.1 损失计算和反向传播

在`train.py`中，我对每个训练批次进行了循环迭代。对于每个批次，我计算了标签、位置和排序的损失，然后按照一定的权重组合得到了最终的训练损失：


```python
label_loss = self.label_loss(out_label, label)
position_loss = self.position_loss(out_position, position)
sort_loss = self.sort_loss(out_sort, sort)

train_loss = 0.2 * label_loss + position_loss * 0.6 + 0.2 * sort_loss
```

这里，我采用了`BCEWithLogitsLoss`、`MSELoss`和`CrossEntropyLoss`作为标签、位置和排序的损失函数。

### 4.2 模型保存


在每一轮训练结束后，我保存了模型的权重，方便后续的预测和部署：


```python
torch.save(self.net.state_dict(), f'param/{date_time}-{epoch}.pt')
```

这样，我们就可以在需要时加载训练好的模型进行预测。

## 五、预测和可视化

当我训练的效果达到满意后，我就可以把训练好的权重文件用于实际生产中了。

### 5.1 模型加载和预测

在`predict.py`中，我首先加载了训练好的模型权重，并将模型设置为评估模式：

```python
predictor = Predictor('param/' + max(os.listdir('param/')))
predictor.net.eval()
```

然后，通过`predict`方法对单张图像进行预测，获取标签、位置、排序和置信度的输出。

### 5.2 可视化工具

最后，通过`tools.py`中的`view_image`方法，我将原始图像与模型预测的标签、位置、排序进行可视化：

```python
tools.view_image(img_path, label, position, sort, out_label, out_position, out_sort)
```

这一步骤有助于直观地了解模型对于输入图像的处理效果，为进一步调优提供了参考。

## 六、关于我

作者：汤青松

微信：songboy8888

日期: 2024-02-02
