from torchvision import transforms
import torch.utils.data.dataloader
import torchvision
import torch

 # 创建Compose对象
transform_func_obj = transforms.Compose([
      transforms.ToTensor()
      # 1. 输入图像每个像素都归一化为[0.0, 1.0]
      # 2. 将这个图像转化为torch.tensor

    , transforms.Normalize((0.5, ), (0.5, ))
      # 1. 每个像素都会执行 pixel = (pixel - mean) / std
    ] 
    
    # 我们在compose初始化的时候，传入了一个list，其中包含两个对象，
    # 这两个对象在未来call这个函数对象的时候会被逐个作用在图片上
)

print(callable(transform_func_obj)) # 函数对象
print(type(transform_func_obj))


# 训练集
train_dataset = torchvision.datasets.FashionMNIST(root="./datasets/", 
                                      train=True, 
                                      transform=transform_func_obj,
                                      download=True)

# 测试集
test_dataset = torchvision.datasets.FashionMNIST(root="./datasets/",
                                     train=False,
                                     transform=transform_func_obj,
                                     download=True)


batch_size = 64
epochs = 20
my_lr = 0.01


# 包装成dataloader
train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=batch_size,
                                                shuffle=True)

test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

images, label = next(iter(train_data_loader))



class MySimpleClassifier(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(MySimpleClassifier, self).__init__(*args, **kwargs)
        self.fully_connected_layer_1 = torch.nn.Linear(28 * 28, 512)
        self.relu_1                  = torch.nn.ReLU  ()        
        self.fully_connected_layer_2 = torch.nn.Linear(512, 512)
        self.relu_2                  = torch.nn.ReLU  ()
        self.fully_connected_layer_3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.fully_connected_layer_1(x)
        x = self.relu_1(x)
        x = self.fully_connected_layer_2(x)
        x = self.relu_2(x)
        x = self.fully_connected_layer_3(x)
        return x
    

model = MySimpleClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=my_lr)

# 打印网络参数
for param in model.parameters():
    print(type(param), param.size())


# 训练前检查设备
# 1. 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 2. 查看当前设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# 3. 查看torch版本和配置
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")

# 4. GPU设备信息
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"当前GPU显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    print(f"当前GPU显存占用: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    print(f"当前GPU显存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")




def train():
    for cur_epoch in range(epochs):  # 每个 epoch
        total_loss = 0
        # 遍历所有 batch
        for batch_idx, (images, labels) in enumerate(train_data_loader):
            # 1. 前向传播：计算输出和损失
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 2. 反向传播：先清零梯度，再计算梯度
            optimizer.zero_grad()  # 清除之前的梯度
            loss.backward()        # 反向传播，计算每个参数的梯度
            
            # 打印本 batch 各层参数的梯度信息
            print(f"Epoch {cur_epoch}, Batch {batch_idx}:")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # 输出梯度的均值和范数，可以直观感受梯度大小
                    print(f"    {name}: grad mean = {param.grad.mean().item():.4f}, norm = {param.grad.norm().item():.4f}")
                else:
                    print(f"    {name}: No grad computed")
            
            # 3. 优化器更新参数（执行梯度下降或自适应更新）
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{cur_epoch}/{epochs}], Average loss: {total_loss/len(train_data_loader):.4f}")


def test():
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for image, labels in test_data_loader:
            outputs = model(image) # 前向传播一次
            _, predicted = torch.max(outputs, 1)
            total_count += labels.size(0)
            correct_count += (predicted == labels).sum().item()
    print(f"test done! accuracy: {100 * correct_count / total_count:.4f} %")



# 主函数
if __name__ == "__main__":
    train()
    test()
            