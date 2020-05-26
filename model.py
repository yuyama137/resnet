import torch
import torch.nn as nn
import torch.nn.functional as F

"""
[参考](https://github.com/kuangliu/pytorch-cifar/tree/master/models)

実装方法:
BasicBlock()
    - 単純な2層パスの一つのブロックを作る
    - 引数は、(inchanel, outchannel)
    - inchannelとoutchannelが違う時は、サイズを1/2する
ResBlock()
    - 一つのchannel数のブロック
    - BasicBlockを積み重ねる
    - 将来的に、bottleneckを積み重ねることも考えて実装
ResNet()
    - 最初の畳み込み層
    - resblockを積み重ねる
    - 引数として、それぞれのresblockが何層ずつ積み重なるかを持つ
    - 最後の畳み込み層
ResNet_18()
    - 18層のネットワークを作る
ResNet_34()
    - 34層のネットワークを作る
"""

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicBlock,self).__init__()

        # resblock
        if in_channel != out_channel:
            self.conv1 = nn.Conv2d(in_channel, out_channel,kernel_size=3,stride=2,padding=1)
        else:
            self.conv1 = nn.Conv2d(out_channel,out_channel,3,1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel,out_channel,3,1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # shortcut
        if in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channel)
            )
        else:
            self.shortcut = nn.Sequential()
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print("a")
        out = self.bn2(self.conv2(out))
        # print("b")
        # print(out.size())
        out += self.shortcut(x)
        # print("c")
        # print(out.size())
        out = F.relu(out)
        return out
        

class ResBlock(nn.Module):
    """
    args:
        block : basicblock or botolneck
        in_channel
        out_channel
    """
    def __init__(self, block, blocknum, in_channel, out_channel):
        super(ResBlock, self).__init__()

        layers = []
        self.block = block# 将来的にbotoleneckを使う時用

        for i in range(blocknum):
            if i==0 and in_channel!=out_channel:# サイズ変更ない時は除外(56->56の時)
                layers.append(BasicBlock(in_channel,out_channel))# 1回目(サイズを変える)
            else:
                layers.append(BasicBlock(out_channel,out_channel))# サイズを変えない時
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# class PlainNet():
#     def __init__(self, block, layers):# input size = [1,3,224,224]
#         self.layers = layers
#         self.block = block

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=5)
    
#     def _make_layers
        


class ResNet(nn.Module):
    """
    args:
        block : basicblock or botolneck
        layernum : number of layer at each resblock
        num_class : number of classes to identify
    """
    def __init__(self, block, layernum, num_class=10):
        super(ResNet, self).__init__()
        self.layernum = layernum
        self.block = block

        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1)# 3 -> 64
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)#64->64
        self.conv2_x = ResBlock(block, layernum[0],64,64)# 64->64
        self.conv3_x = ResBlock(block, layernum[1],64,128)# 64 -> 128
        self.conv4_x = ResBlock(block, layernum[2],128,256)# 128 -> 256
        self.conv5_x = ResBlock(block, layernum[3],256,512)# 256 -> 512

        # avepool,全結合層(1000),softmax
        self.linear = nn.Linear(512,num_class)
    
    def forward(self, x):
        # print(x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.size())
        out = self.conv2_1(out)
        # print("pass conv2_1")
        # print(out.size())
        out = self.conv2_x(out)
        # print("pass conv2_x")
        # print(out.size())
        out = self.conv3_x(out)
        # print("pass conv3_x")
        # print(out.size())
        out = self.conv4_x(out)
        # print("pass conv4_x")
        # print(out.size())
        out = self.conv5_x(out)
        # print("pass conv4_x")
        # print(out.size())
        out = F.avg_pool2d(out,out.size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.softmax(out,dim=1)
        return out




def ResNet_18(num_class):# 外部から呼ばれる
    return ResNet(block="BasicBlock",layernum=[2,2,2,2], num_class=num_class)

def ResNet_34(num_class):
    return ResNet(block="BasicBlock",layernum=[3,4,6,3], num_class=num_class)

# def PlainNet_18():
#     return PlainNet(layernum=[2,2,2,2])

# def PlainNet_34():
#     return PlainNet(layernum=[3,4,6,3])

if __name__ == "__main__":
    x = torch.randn(2,3,32,32)
    net = ResNet_18(10)
    y = net(x)
    print(y.size())

