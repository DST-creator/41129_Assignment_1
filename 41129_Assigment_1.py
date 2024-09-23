#%% Imports
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import os

#Regression model imports
from scipy import optimize
from scipy import integrate



#%%Global plot settings
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
# mpl.rcParams['ztick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Optional, for math symbols


#%% Input data

replot_tasks = dict(T1=False, 
                    T4=False,
                    T5=True,
                    T6=True,
                    T7=True,
                    T8=True,
                    )

h = .07         #[m]
b = .3          #[m]
nu = 1e-6       #[m^2/s]

structs = scipy.io.loadmat("Exercise1.mat", struct_as_record=False, squeeze_me=True)["Channel"]

#%% Action item 1
#Calculate mean u-velocities for all channels
struct_len = len(structs)
u_means = np.zeros(struct_len)
y = np.zeros(struct_len)
for i in range(struct_len):
    u = structs[i].u
    tt = structs[i].tt
    u_means[i] = float(np.sum(np.multiply(u,tt)) / np.sum(tt))
    y[i] = structs[i].y

#Manually insert start and end values
y = np.append(np.insert(y, 0, 0), .07)
u_means = np.append(np.insert(u_means, 0, 0), .3)

if replot_tasks["T1"]:
    fig1, ax1 = plt.subplots(figsize=(16, 10))
    ax1.plot(u_means, y)
    
    #Formatting
    ax1.set_title('Mean velocity')
    ax1.set_ylabel('$y\:[m]$')
    ax1.set_xlabel(r'$\bar{u}\:[m/s]$')
    ax1.grid()
    
    fig1.savefig(fname="Task_1_plot.svg",
                bbox_inches = "tight")
else:
    print("Plot for Task 1 not replotted")

#%% Action item 2
#Calculate depth-averaged velocity

V = 1/h* scipy.integrate.cumulative_trapezoid(u_means, y)[-1]
#V should be 0.264 roughly

#%% Action item 3
#Calculate the friction velocity

A = h*b                     #[m^2]
P = 2*h + b                 #[m]
r_h = A/P                   #[m]
Re = r_h*V/nu               #[-]

f = .0557/Re**.25           #[-]
U_f_est = np.sqrt(f/2)*V        #[m/s]

#%% Action item 4

#Calculate y+
y_plus = y*U_f_est/nu

#Determine the logarithmic region approximation function
Re_tau = h*U_f_est/nu
i_upper_bound = np.where(y<=.1*h)
i_lower_bound = np.where(y_plus>=30)
i_log = np.intersect1d(i_lower_bound, i_upper_bound)

popt = np.polyfit (np.log(y[i_log]), u_means[i_log], deg=1)
func_log_layer = lambda y: popt[0]*np.log(y) + popt[1]



if replot_tasks["T4"]:
    fig4, ax4 = plt.subplots(figsize=(10, 10))
    ax4.scatter(u_means, y, label = "true data", marker = "+")
    ax4.plot(func_log_layer(y[i_log]),
            y[i_log],
            label="Approximation function")
    
    #Formatting
    ax4.set_title('Mean velocity')
    ax4.set_ylabel('$y\:[m]$')
    ax4.set_xlabel(r'$\bar{u}\:[m/s]$')
    ax4.set_yscale("log")
    ax4.grid()
    ax4.legend(loc="lower right")
    fig4.savefig(fname="Task_4_plot.svg",
                bbox_inches = "tight")
else:
    print("Plot for Task 4 not replotted")

#Recalculate U_f (Eq. 3.15 - should be the same for all)
U_f = np.mean(y_plus[1:] * nu / y[1:])

#%% Action Item 5

if replot_tasks["T5"]:
    fig5, ax5 = plt.subplots(figsize=(16, 10))
    ax5.scatter(u_means/U_f, y_plus, label = "true data", marker = "+")
    ax5.plot(func_log_layer(y[i_log])/U_f,
            y_plus[i_log],
            label="Approximation function")
    rect_visc_sub = mpl.patches.Rectangle((-10,0), 40, 5, 
                                          fc = "b", alpha = .1,
                                          label= "Viscous sublayer")
    rect_visc_buffer = mpl.patches.Rectangle((-10,5), 40, 30-5, 
                                             fc = "r", alpha = .1,
                                             label= "Buffer layer")
    rect_log = mpl.patches.Rectangle((-10,30), 40, .1*h*U_f_est/nu-30, 
                                     fc = "g", alpha = .1,
                                     label= "Logarithmic sublayer")
    rect_out = mpl.patches.Rectangle((-10, .1*h*U_f_est/nu), 
                                     40, 1000-.1*h*U_f_est/nu, 
                                     fc = "c", alpha = .1,
                                     label= "Outer region")
    
    ax5.add_patch(rect_visc_sub)
    ax5.add_patch(rect_visc_buffer)
    ax5.add_patch(rect_log)
    ax5.add_patch(rect_out)
    
    #Formatting
    dx_ticks = 2
    ax5.set_title('Mean velocity')
    ax5.set_ylabel('$y^+\:[-]$')
    ax5.set_xlabel(r'$\frac{\bar{u}}{U_f}\:[-]$')
    ax5.set_yscale("log")
    ax5.set_xlim([0, max(u_means/U_f)*1.05])
    ax5.set_xticks(np.arange(0, int(np.ceil(u_means[-1]/U_f/dx_ticks))*dx_ticks, 
                            dx_ticks))
    ax5.grid()
    ax5.legend(loc="lower right")
    fig5.savefig(fname="Task_5_plot.svg",
                bbox_inches = "tight")
else:
    print("Plot for Task 5 not replotted")

#%% Action Item 6

def dfunc_3_108(y_p, kappa=.4, A_d=25):
    u_mean = 2*U_f*np.divide (1,
                              1 + np.sqrt(1 + 4*np.power(kappa,2)
                                               *np.power(y_p,2)
                                               *np.power(1-np.exp(-y_p/A_d), 2)
                                          )
                              )
    return u_mean
                                                  
u_vD = integrate.cumulative_trapezoid (y=dfunc_3_108(y_plus, kappa=.4, A_d=25),
                                       x=y_plus)
y_plus_vd = np.array([(y_plus[i+1]+y_plus[i])/2 for i in range(len(y_plus)-1)])

ax5.plot(u_vD/U_f,
        y_plus_vd,
        label="van Driest velocity")
ax5.legend(loc="lower right")
fig5.savefig(fname="Task_6_plot.svg",
            bbox_inches = "tight")


