# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:43:39 2024

@author: davis
"""

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 14, 12, 18, 16]

# Create a scatter plot with unfilled markers
plt.scatter(x, y, facecolors='none', edgecolors='blue', marker='o', s=100)

# Add labels and title
plt.title('Scatter Plot with Unfilled Markers')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()

plt.savefig(fname="test.svg")