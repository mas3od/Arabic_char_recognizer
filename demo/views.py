from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import re
from PIL import Image
import io
import torch
import base64
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
import pandas as pd

size = 32,32
# Create your views here.
def index(request):
    if request.is_ajax():
        print('ajax')
        #return HttpResponse('hellloooo')
        return JsonResponse({
        'hello' : 'hhhh',
        'success' : 'yes'
    })
    print('index')
    return render(request, 'index.html')

def update(request):
    
    if request.method == 'GET':
            print('yes')
            image = request.GET.get('img', None)
            char = int(request.GET.get('char', None))
            
            image_str = re.search(r'data:image/png;base64,(.*)', image).group(1)
            #print(image_str)
            #padding = len(image)%4
            #image += '=' * (len(image) % 4)
            #print((image_str))
            image_bytes = io.BytesIO(base64.b64decode(image_str))
            im = Image.open(image_bytes).transpose(2).transpose(Image.FLIP_TOP_BOTTOM)
            im.thumbnail(size, Image.ANTIALIAS)
            tensr = transforms.ToTensor()(im)[3].float()
            newDataset = tensr.view(1024).numpy()
            print(newDataset.shape)
            #columns = ['image','label']
            #df = pd.read_csv('dataset.csv')
            #res = [newDataset,[char]]
            #res.append(np.array(char))
            df1 = pd.DataFrame([newDataset])
            df2 = pd.DataFrame([char])
            #df.append([res])
            df1.to_csv('image.csv',mode='a',header=False,index=False,sep=';')
            df2.to_csv('labels.csv',mode='a',header=False,index=False,sep=';')

            
    data = {
            'success' : True,
            
        }  
    return JsonResponse(data) 


def ajax(request):
        #request_csrf_token = request.POST.get('csrfmiddlewaretoken', '')
        #request_getdata = request.POST.get('getdata', None) 
        number = 0
        models ={
            'cnn1'  : {
                'algo' : ConvNet,
                'link' : 'models/CnnWithValidation.pt'

            },
            'cnn2'  : {
                'algo' : ConvNet1,
                'link' : 'models/CnnWithValidation1225.pt'

            },
            'rnn' : {
                'algo' : RNN,
                'link' : 'models/rnn.pt'
            },
            'resnet' : {
                'algo' : ResNet,
                'link' : 'models/resnetBest.pt'
            },

        }
        net_args = {
        "block": ResidualBlock,
        "layers": [2, 2, 2, 2]
            }
        if request.method == 'GET':
            image = request.GET.get('img', None)
            algo = request.GET.get('algo', None)
            if algo == 'resnet': model =  models[algo]['algo'](**net_args)
            else : model = models[algo]['algo']()
            model.load_state_dict(torch.load(models[algo]['link'],map_location='cpu'))
            model.eval()
            #print((image))
            image_str = re.search(r'data:image/png;base64,(.*)', image).group(1)
            #print(image_str)
            #padding = len(image)%4
            #image += '=' * (len(image) % 4)
            #print((image_str))
            image_bytes = io.BytesIO(base64.b64decode(image_str))
            im = Image.open(image_bytes).transpose(2).transpose(Image.FLIP_TOP_BOTTOM)
            im.thumbnail(size, Image.ANTIALIAS)
            tensr = transforms.ToTensor()(im)[3].float()
            newDataset = tensr.view(1024).numpy()
            print(newDataset.shape)
            if algo == 'rnn': test_data = tensr.view(32,32)
            else : test_data = tensr.view(1,32,32)
            c = test_data.unsqueeze(0)
            a = model(c)
            sm = torch.nn.Softmax(dim=1)
            b = sm(a) 
            top3,top3i2 = torch.topk(b.data,3)

            top = top3.squeeze(0).numpy().tolist()
            index = top3i2.squeeze(0).numpy().tolist()
            #[:,:,0]
            #image = re.sub('data:image/png;base64,', '',image)
            #lst = [ord(c) for c in image]
            #number =  int(request.GET['input'] )
        data = {
            'success' : True,
            'result' : number*5,
            'top3' : json.dumps(top),
            'array' : json.dumps(index)
        }   
        return JsonResponse(data)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
            
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(36*128, 100)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 28)
        #self.fc3 = nn.Linear(50, 28)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out =self.conv2_drop(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out =self.drop(out)
        out = self.fc2(out)
        #out = self.fc3(out)
        return out

class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        
            
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(32))
        self.fc1 = nn.Linear(128, 28)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, 28)
        #self.fc3 = nn.Linear(50, 28)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out =self.conv2_drop(out)
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        #out =self.drop(out)
        #out = self.fc2(out)
        #out = self.fc3(out)
        return out

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.bn2(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=28):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv = conv3x3(1, 32)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        #self.layer0 = self.make_layer(block, 16, layers[0])
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[0])
        self.layer3 = self.make_layer(block, 128, layers[0],2)
        self.avg_pool = nn.AvgPool2d(16)
        self.max_pool = nn.MaxPool2d(4)
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(128, num_classes)
        self.fc1 = nn.Linear(100, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        #out = self.relu(out)
        out = self.bn(out)
        #out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        #out = self.act(out)
        #out = self.fc1(out)
        return out

BATCH_SIZE = 32
TIME_STEP = 32          # rnn time step / image height
INPUT_SIZE = 32         # rnn input size / image width
LR = 0.01 
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=128,         # rnn hidden unit
            num_layers=3,           # number of rnn layer
            #dropout=0.05, 
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.fc1= nn.Linear(128,64)
        self.dout = nn.Dropout(0.3)
        self.out = nn.Linear(64, 28)
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(64)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        #out = self.ln1(r_out[:, -1, :])
        #out = self.act(out)
        out = self.fc1(r_out[:, -1, :])
        
        out = self.ln2(out)
        out = self.act(out)
        
        
        out = self.dout(out)
        out = (self.out(out))
        #out = self.act(out)
        return out