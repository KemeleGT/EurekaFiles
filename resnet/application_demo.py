# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import warnings
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import utils as vutils
import os
import shutil
# import model
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from IPython import embed
import datetime
import struct
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
from resnet import vizualize_model


 


w = 600
h = 800

found_model = False


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
 
        # setting title
        self.setWindowTitle("EUREKA ")
 
        # setting geometry
        self.setGeometry(100, 100, w, h)
 
        # calling method
        self.UiComponents()
 
        # showing all the widgets
        self.show()
 
    # method for widgets
    def UiComponents(self):
 
        # creating load model button
        button = QPushButton("Load model", self)
        button.setGeometry(200, 50, 200, 30)
        button.clicked.connect(self.open_file_dialog)


        #creating a label with geometry
        nameLabel = QLabel('Click here to load model',self)
        nameLabel.setGeometry(w//2 - w//4, 0, w//2, 30)
        
        # mode_status = QLabel('no mode selected', self)
        # mode_status.setGeometry(w//2 - w//4, 200,w//2, 30)

        #creating labels for parameters
        bit_label = QLabel('bit number: ', self)
        type_label = QLabel('distance level: ', self)
        attack_label = QLabel('attack layer: ', self)
        error_label = QLabel('error number: ', self)


        bit_label.setGeometry(50, 200, 200, 30)
        type_label.setGeometry(50, 250, 200, 30)
        attack_label.setGeometry(50, 300, 200, 30)
        error_label.setGeometry(50,350,200,30)



        # Creating TextBox

        self.bit_text = QLineEdit(self)
        self.type_text = QLineEdit(self)
        self.attack_text = QLineEdit(self)
        self.error_text = QLineEdit(self)

        self.bit_text.setGeometry(300,200,200,30)
        self.type_text.setGeometry(300,250,200,30)
        self.attack_text.setGeometry(300,300,200,30)
        self.error_text.setGeometry(300,350,200,30)

        #crearing button for evaluation
        evaluate = QPushButton("evaluate model", self)
        evaluate.setGeometry(200, 450, 200, 30)
        evaluate.clicked.connect(self.evaluation)
        
 
    # action method

    def explore_fun(self):
        pass

    def load_model(self):
 
        # printing pressed
        print("pressed")
        vizualize_model.main()

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName( self, "Select a File")
        if filename:
            path = Path(filename)

        if "resnet" in str(path):
            vizualize_model.main()
            found_model = True

    def evaluation(self):
        bit_num = self.bit_text.text()
        attack_layer = self.attack_text.text()
        distance_level = self.type_text.text()
        error_num = self.error_text.text()

        cmd = "python /home/kmejbaulislam/Desktop/eureka/model_evaluations/resnet/resnet_eval.py --params="
        cmd+=error_num+","+distance_level+","+attack_layer+","+bit_num
        print("evaluate")
        os.system(cmd)

        
        # print(cmd)

     
# create pyqt5 app
App = QApplication(sys.argv)
 
# create the instance of our Window
window = Window()
 
# start the app
sys.exit(App.exec())