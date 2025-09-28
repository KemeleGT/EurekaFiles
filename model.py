import torch
import torch.nn as nn
from typing import Any
#from torch.hub import load_state_dict_from_url
import urllib

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        '''
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        '''

      
  
        self.conv1      = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool1   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2      = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool2   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3      = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu       = nn.ReLU(inplace=True)
        self.conv4      = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu       = nn.ReLU(inplace=True)
        
        self.conv5      = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.relu       = nn.ReLU(inplace=True)
        self.maxpool3   = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.avgpool    = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.final_fcn = nn.Linear(4096, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.features(x)
        
        
        
        conv1_out = self.conv1(x)  
        x = self.relu(conv1_out)    
        x = self.maxpool1(x)
        conv2_out = self.conv2(x)   
        x = self.relu(conv2_out)    
        x = self.maxpool2(x)
        
        conv3_out = self.conv3(x)   
        
        x = self.relu(conv3_out)    
        
        conv4_out = self.conv4(x)   
        
        x = self.relu(conv4_out)    
        
        conv5_out = self.conv5(x)   
        
        x = self.relu(conv5_out)    
        x = self.maxpool3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.final_fcn(x)
        x = self.softmax(x)
        return x, conv1_out,conv2_out, conv3_out, conv4_out, conv5_out
        
    def forward1(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.features(x)
        x = self.conv1(x)  
        return x
    def forward2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x) 
        return x
    def forward3(self, x: torch.Tensor) -> torch.Tensor:        
        x = self.maxpool1(x)
        return x
    def forward4(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv2(x)
        return x
    def forward5(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.relu(x)    
        return x
    def forward6(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.maxpool2(x)
        return x
    def forward7(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv3(x)   
        return x
    def forward8(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.relu(x)    
        return x
    def forward9(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv4(x)   
        return x
    def forward10(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.relu(x)    
        return x
    def forward11(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.conv5(x)   
        return x
    def forward12(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.relu(x)   
        return x
    def forward13(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.maxpool3(x)
        return x
    def forward14(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.avgpool(x)
        return x
    def forward15(self, x: torch.Tensor) -> torch.Tensor: 
        x = torch.flatten(x, 1)
        return x
    def forward16(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.classifier(x)
        return x
    def forward17(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.final_fcn(x)
        return x
    def forward18(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.softmax(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        #state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        urllib.request.urlretrieve(model_urls['alexnet'], './alexnet-owt-7be5be79.pth')
        model_name = './alexnet-owt-7be5be79.pth'
        pretrained_weights = torch.load(model_name)#, map_location=torch.device('cuda:'+str(cudanum)))
        model.load_state_dict(pretrained_weights)
    return model
