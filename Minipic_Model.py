import paddle
import paddle.nn as nn


class mymodel(paddle.nn.Layer):
    def __init__(self, image_shape):
        super(mymodel,self).__init__()

        s = image_shape[1]
        self.Down_sample = nn.MaxPool2D(kernel_size = 2 ,stride = 2)

        self.Active_f = nn.ReLU()

        self.Conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm2D(16)

        self.Conv2 = nn.Conv2D(16, 16, 3, 1, padding=1)
        self.BN2 = nn.BatchNorm2D(16)

        self.FC = nn.Sequential(# Rearrange('b c h w -> b (c h w)'),

                                nn.Linear(16*(s//4)**2 , 16),
                                nn.Dropout(0.5),
                                nn.ReLU(),

                                nn.Linear(16,10))
                                ## 这里linear第二个参数是一共有多少类

    def forward(self , input):
        x = self.Active_f(self.Down_sample(self.BN1(self.Conv1(input))))
        x = self.Active_f(self.Down_sample(self.BN2(self.Conv2(x))))
        output = self.FC(x)
        return output


# model = mymodel([3, 32, 32])
# img = torch.randn(1, 3, 32, 32)
# print(model(img).shape)


class DNN(nn.Layer):
    def __init__(self, n_classes=10):
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(3*32*32, 1024),     # 第一个全连接隐藏层
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, n_classes)  # 第二个全连接隐藏层
        )

    def forward(self, x):
        return self.model(x)