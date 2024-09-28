# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:59:50 2024

@author: davis
"""

import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your actual data)
x = np.logspace(0, 4, 100)  # Logarithmic scale for x
y = np.log(x)  # Example y values (replace with your actual data)

# Create the figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y, 'k.')  # Your actual plot

# Setting the scales for the plot
ax.set_xscale('log')
ax.set_yscale('linear')

# Axis labels
ax.set_xlabel(r"$y^+$")
ax.set_ylabel(r"$\frac{\bar{u}}{U_f}$")

# Annotations below the x-axis
ax.annotate('Viscous\nsublayer', xy=(5, -5), xycoords='data', textcoords='data',
            ha='center', va='top', fontsize=10)
ax.annotate('Logarithmic\nlayer', xy=(30, -5), xycoords='data', textcoords='data',
            ha='center', va='top', fontsize=10)

arrowstyle = dict(arrowstyle="->", 
                  connectionstyle="angle,angleA=0,angleB=90")

# Drawing arrows for the regions
ax.annotate('', xy=(1, -5), xytext=(5, -5),
            arrowprops=dict(arrowstyle='<->', lw=1.5))
ax.annotate('', xy=(5, -5), xytext=(30, -5),
            arrowprops=dict(arrowstyle='<->', lw=1.5))
ax.annotate('', (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', arrowprops=arrowstyle)

# Set the y-axis lower limit to make space for the annotations and arrows
ax.set_ylim(-10, 30)

# Customize the axis limits and ticks
ax.set_xlim(1, 1e4)

# Show the grid and plot
ax.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
fig.savefig(fname="test.svg")