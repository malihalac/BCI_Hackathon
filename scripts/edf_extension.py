# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 10:00:23 2021

@author: delor
"""

import os

directory = "/Users/malihalac/desktop/BCI_Hackathon/github/BCI_Hackathon/edf_data/"

for file in os.listdir(directory):
    if ".edf" not in file and ".md" not in file and ".py" not in file:
        print(file)
        os.rename("/Users/malihalac/desktop/BCI_Hackathon/github/BCI_Hackathon/edf_data/" + file, "/Users/malihalac/desktop/BCI_Hackathon/github/BCI_Hackathon/edf_data/" + file.replace(" ", "")+ ".edf")