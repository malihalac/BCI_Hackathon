# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:00:23 2021

@author: delor
"""

import os

directory = "C:/Users/delor/OneDrive/Desktop/Drexel University/General/Hackathon/BCI2021/BCI_hackathon-main/"

for file in os.listdir(directory):
    if ".edf" not in file and ".md" not in file and ".py" not in file:
        print(file)
        os.rename(file, "C:/Users/delor/OneDrive/Desktop/Drexel University/General/Hackathon/BCI2021/BCI_hackathon-main/" + file.replace(" ", "")+ ".edf")