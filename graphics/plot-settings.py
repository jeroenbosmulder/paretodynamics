import matplotlib.pyplot as plt
import numpy as np

# COLORS
black = np.array([0,0,0])/256
orange = np.array([230,159,0])/256
skyblue = np.array([86,180,233])/256
bluishgreen = np.array([0,158,115])/256
yellow = np.array([240,228,66])/256
blue = np.array([0,114,178])/256
vermillion= np.array([213,94,0])/256
reddishpurple= np.array([204,121,167])/256

color1 = 'darkblue'
color2 = 'dodgerblue'
color3 = 'darkorange'
color4 = 'navajowhite'
color5 = 'grey'
color6 = np.array([230,159,0])/256
color6 = 'red'
color7 = 'orangered'


# SETTINGS

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title