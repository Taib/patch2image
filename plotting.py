# -*- coding: utf-8 -*-
"""Usefull methods for patch and image visualization.
"""

# Author: Taibou BIRGUI SEKOU <taibou.birgui_sekou@insa-cvl.fr>

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def plot_images(images, nrows=None, ncols=None,
             title='', figsize=(4.2, 4), cmap=None):
    plt.figure(figsize=figsize)  
    c = np.ceil(np.sqrt(len(images))) if ncols is None else ncols
    l = len(images)//c if nrows is None else nrows
    for i, im in enumerate(images): 
        plt.subplot(l, c, i+1)  
        plt.imshow(np.squeeze(im), cmap=cmap)  
        plt.xticks(())
        plt.yticks(()) 
    plt.suptitle(title)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()

def plot_3D(data): 
    
    fig, ax = plt.subplots()
    plt.subplots_adjust( bottom=0.25)  
    plt.imshow(data[10]) 
    
    axcolor = 'lightgoldenrodyellow'
    axZ = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor) 
    
    sZ = Slider(axZ, 'Z', 1, data.shape[0], valinit=0) 
     
    def update(val): 
        Z = int(sZ.val) 
        ax.imshow(data[Z]) 
    sZ.on_changed(update) 
    
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
     
    def reset(event):
        sZ.reset() 
    button.on_clicked(reset) 
    plt.show()


def plot_4D(data): 
    
    fig, ax = plt.subplots()
    plt.subplots_adjust( bottom=0.25)  
    plt.imshow(data[10, 10]) 
    
    axcolor = 'lightgoldenrodyellow'
    axZ = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    axT = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    
    sZ = Slider(axZ, 'Z', 1, data.shape[1], valinit=0)
    sT = Slider(axT, 'T', 1, data.shape[0], valinit=0) 
    
    def update(val):
        T = int(sT.val)
        Z = int(sZ.val) 
        ax.imshow(data[T, Z]) 
    sZ.on_changed(update)
    sT.on_changed(update)
    
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975') 
    
    def reset(event):
        sZ.reset()
        sT.reset()
    button.on_clicked(reset)
     
    plt.show()