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

#Lines and markers
mpl.rcParams['lines.linewidth'] = 1.2
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['scatter.marker'] = "+"
mpl.rcParams['lines.color'] = "k"
mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])
# Cycle through linestyles with color black instead of different colors
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler('color', ['k', 'k', 'k', 'k'])\
#                                 + mpl.cycler('linestyle', ['-', '--', '-.', ':'])

#Text sizes
mpl.rcParams['font.size'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['legend.fontsize'] = 20

#Padding
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
# mpl.rcParams['ztick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20

#Latex font
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Optional, for math symbols

#Custom overline function (cf. https://tex.stackexchange.com/questions/22100/the-bar-and-overline-commands)
#Note: this slows down the code extremely sometimes 
plt.rcParams['text.latex.preamble'] = r"""
\newcommand{\ols}[1]{\mskip.5\thinmuskip\overline{\mskip-.5\thinmuskip {#1} \mskip-.5\thinmuskip}\mskip.5\thinmuskip}""" 

#Export
mpl.rcParams['savefig.bbox'] = "tight"

#%% Input data

replot_tasks = dict(T1=False, 
                    T4=False,
                    T5=False,
                    T6=False,
                    T7=False,
                    T8=False,
                    T9=False,
                    T10=False,
                    T11=True,
                    )

h = .07         #[m]
b = .3          #[m]
nu = 1e-6       #[m^2/s]

structs = scipy.io.loadmat("Exercise1.mat", struct_as_record=False, squeeze_me=True)["Channel"]

#%% Action item 1
#Calculate mean velocities for all channels
struct_len = len(structs)
u_means = np.zeros(struct_len)
v_means = np.zeros(struct_len)
y = np.zeros(struct_len)
for i in range(struct_len):
    t = structs[i].t
    u_means[i] = integrate.trapezoid(structs[i].u, t) / t[-1]
    v_means[i] = integrate.trapezoid(structs[i].v, t) / t[-1]
    y[i] = structs[i].y

#Manually insert start and end values
y = np.append(np.insert(y, 0, 0), .07)
u_means = np.append(np.insert(u_means, 0, 0), .3)

if replot_tasks["T1"]:
    fig1, ax1 = plt.subplots(figsize=(16, 10))
    ax1.scatter(u_means, y, s=150, linewidths=1.5)
    
    #Formatting
    ax1.set_title('Mean velocity')
    ax1.set_ylabel('\[y\:[m]\]')
    ax1.set_xlabel(r'\[\overline{u}\:[m/s]\]')
    ax1.grid()
    
    fig1.savefig(fname="Task_1_plot.svg")
    plt.close(fig1)
else:
    print("Plot for Task 1 not replotted")

#%% Action item 2
#Calculate depth-averaged velocity

V = 1/h* integrate.trapezoid(u_means, y)
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
    ax4.scatter(u_means, y, label = "true data")
    ax4.plot(func_log_layer(y[i_log]),
            y[i_log],
            label="Approximation function",
            ls="-", c='k')
    
    #Formatting
    ax4.set_ylabel('\[y\:[m]\]')
    ax4.set_xlabel(r'\[\overline{u}\:[m/s]\]')
    ax4.set_yscale("log")
    ax4.grid()
    ax4.legend(loc="lower right")
    fig4.savefig(fname="Task_4_plot.svg")
    plt.close(fig4)
else:
    print("Plot for Task 4 not replotted")

#Recalculate U_f (Eq. 3.15 - should be the same for all)
U_f = np.mean(y_plus[1:] * nu / y[1:])

#%% Action Item 5

if replot_tasks["T5"]:
    fig5, ax5 = plt.subplots(figsize=(16, 10))
    plt5_sc1 = ax5.scatter(u_means/U_f, y_plus, label = "True data", 
                           s=150, linewidths=1.5)
    plt5_line1 = ax5.plot(func_log_layer(y[i_log])/U_f,
                          y_plus[i_log],
                          label="Approximation function",
                          ls="--", c='k', lw=1.5)
    
    plt5_lgd1 = plt.legend(handles=[plt5_sc1, plt5_line1[0]], 
                           loc='upper left')
    plt5_lgd1 = ax5.add_artist(plt5_lgd1)
    
    #Regions
    rect_visc_sub = mpl.patches.Rectangle((-10,0), 40, 5, 
                                          hatch="//\\\\", fc = "1", alpha = .35,
                                          label= "Viscous sublayer")
    rect_visc_buffer = mpl.patches.Rectangle((-10,5), 40, 30-5, 
                                             hatch="//", fc = "1", alpha = .45,
                                             label= "Buffer layer")
    rect_log = mpl.patches.Rectangle((-10,30), 40, .1*h*U_f_est/nu-30, 
                                     hatch="\\\\/",  fc = "1", alpha = .35,
                                     label= "Logarithmic layer")
    rect_out = mpl.patches.Rectangle((-10, .1*h*U_f_est/nu), 
                                     40, .9*h*U_f_est/nu, 
                                     hatch="\\\\", fc = "1", alpha = .45,
                                     label= "Outer region")
    
    ax5.add_patch(rect_visc_sub)
    ax5.add_patch(rect_visc_buffer)
    ax5.add_patch(rect_log)
    ax5.add_patch(rect_out)
    
    plt5_lgd2 = plt.legend(handles=[rect_visc_sub, rect_visc_buffer,
                                    rect_log, rect_out], 
                           loc='lower right')
    ax5.add_artist(plt5_lgd2)
    #Formatting
    dx_ticks = 2
    ax5.set_ylabel('\[y^+\:[-]\]')
    ax5.set_xlabel(r'\[\frac{\overline{u}}{U_f}\:[-]\]')
    ax5.set_yscale("log")
    ax5.set_xlim([0, max(u_means/U_f)*1.05])
    ax5.set_xticks(np.arange(0, int(np.ceil(u_means[-1]/U_f/dx_ticks))*dx_ticks, 
                            dx_ticks))
    ax5.grid()
    fig5.savefig(fname="Task_5_plot.svg")
    
    if not replot_tasks["T6"]:
        plt.close(fig5)
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

if replot_tasks["T6"]:
    if replot_tasks["T5"]:
        plt6_line1 = ax5.plot(u_vD/U_f,
                               y_plus_vd,
                               label="van Driest velocity", 
                               ls = "--", marker = "v", ms=7)
        plt5_lgd1 = plt.legend(handles=[plt5_sc1, plt5_line1[0], plt6_line1[0]], 
                               loc='upper left')
        plt5_lgd1 = ax5.add_artist(plt5_lgd1)
        fig5.savefig(fname="Task_6_plot.svg",
                    bbox_inches = "tight")
        plt.close(fig5)
    else:
        print("Task 6 can only be replotted if Task 5 is plotted")
else:
    print("Plot for Task 6 not replotted")

#%% Action Item 7

#Calculate turbulence quantities
u_var_mean = np.zeros(struct_len)
v_var_mean = np.zeros(struct_len)

turb_quant_u = np.zeros(struct_len)
turb_quant_v = np.zeros(struct_len)
turb_quant_uv = np.zeros(struct_len)

for i in range(struct_len):
    t = structs[i].t
    
    u_var = structs[i].u-u_means[i]
    v_var = structs[i].v-v_means[i]
    
    u_var_mean[i] = integrate.trapezoid(np.power(u_var, 2), t) / t[-1]
    v_var_mean[i] = integrate.trapezoid(np.power(v_var, 2), t) / t[-1]
    
    turb_quant_u[i] = np.sqrt(u_var_mean[i]) / U_f
    turb_quant_v[i] = np.sqrt(v_var_mean[i]) / U_f
    turb_quant_uv[i] = np.sqrt(-integrate.trapezoid(u_var*v_var, t) / t[-1])/U_f    



if replot_tasks["T7"]:
    fig7, ax7 = plt.subplots(figsize=(16, 10))
    ax7.plot(y_plus[1:-1],
                turb_quant_u,
                label=r"\[\sqrt{\overline{u^{'2}}}/U_f\]", 
                ls = "--", marker = "+")
    ax7.plot(y_plus[1:-1],
                turb_quant_v,
                label=r"\[\sqrt{\overline{v^{'2}}}/U_f\]",
                ls = "--", marker = "x")
    ax7.plot(y_plus[1:-1],
                turb_quant_uv,
                label=r"\[\sqrt{\overline{u^{'}v^{'}}}/U_f\]",
                ls = "--", marker = "v", ms=7)
    
    #Empty scatter (needed for the distribution of the labels in the columns 
    # of the legend)
    ax7.scatter(1,1, label=' ', c='#ffffff') 
    
    #Regions
    rect_visc_sub = mpl.patches.Rectangle((0,0), 5, 5, 
                                          hatch="//\\\\", fc = "1", alpha = .35,
                                          label= "Viscous sublayer")
    rect_visc_buffer = mpl.patches.Rectangle((5, 0), 30-5, 5, 
                                             hatch="//", fc = "1", alpha = .45,
                                             label= "Buffer layer")
    rect_log = mpl.patches.Rectangle((30,0), .1*h*U_f/nu-30, 5, 
                                     hatch="\\\\/",  fc = "1", alpha = .35,
                                     label= "Logarithmic layer")
    rect_out = mpl.patches.Rectangle((.1*h*U_f/nu, 0), 
                                     .9*h*U_f/nu, 5, 
                                     hatch="\\\\", fc = "1", alpha = .45,
                                     label= "Outer region")
     
    ax7.add_patch(rect_visc_sub)
    ax7.add_patch(rect_visc_buffer)
    ax7.add_patch(rect_log)
    ax7.add_patch(rect_out)
    
    #Formatting
    ax7.set_xlim([0,100])
    ax7.set_ylim([0,4.5])
    ax7.set_xlabel('\[y^+\:[-]\]')
    ax7.set_ylabel(r'Turbulence quantities $[-]$')
    ax7.grid()
    ax7.legend(loc="upper right", ncols=2)
    fig7.savefig(fname="Task_7_plot.svg")
    plt.close(fig7)
else:
    print("Plot for Task 7 not replotted")

#%% Action Item 8
if replot_tasks["T8"]:
    fig8, ax8 = plt.subplots(figsize=(16, 10))
    ax8.plot(y[1:-1]/h,
                turb_quant_u,
                label=r"\[\sqrt{\overline{u^{'2}}}/U_f\]",
                ls = "--", marker = "+")
    ax8.plot(y[1:-1]/h,
                turb_quant_v,
                label=r"\[\sqrt{\overline{v^{'2}}}/U_f\]",
                ls = "--", marker = "x")
    ax8.plot(y[1:-1]/h,
                turb_quant_uv,
                label=r"\[\sqrt{\overline{u^{'}v^{'}}}/U_f\]",
                ls = "--", marker = "2")
    
    #Formatting
    ax8.set_xlim([0,1])
    ax8.set_xlabel('\[y/h\:[-]\]')
    ax8.set_ylabel(r'Turbulence quantities $[-]$')
    ax8.grid()
    ax8.legend(loc="upper right")
    fig8.savefig(fname="Task_8_plot.svg",
                bbox_inches = "tight")
    plt.close(fig8)
else:
    print("Plot for Task 8 not replotted")
    
#%% Action Item 9
#Calculate the turbulent kinetic energy
tke = .5*(u_var_mean+v_var_mean*2.8)/U_f**2

if replot_tasks["T9"]:
    fig9, ax9 = plt.subplots(figsize=(16, 10))
    ax9.scatter(y[1:-1]/h,
            tke,
            label=r"\[k/U_f^2\]")
    
    #Formatting
    ax9.set_xlim([0,1])
    ax9.set_xlabel('\[y/h\:[-]\]')
    ax9.set_ylabel(r'\[k/U_f^2\:[-]\]')
    ax9.grid()
    fig9.savefig(fname="Task_9_plot.svg")
    plt.close(fig9)
else:
    print("Plot for Task 9 not replotted")
    
#%% Action Item 10
#Calculate the Reynolds stress
rho = 1000 #[kg/m^3] - assumed water density
du_dy = np.gradient(u_means, y)

tau_mean = rho*U_f**2*(1-y/h)
rs = tau_mean-rho*nu*du_dy

if replot_tasks["T10"]:
    fig10, ax10 = plt.subplots(figsize=(16, 10))
    ax10.plot(y/h,
            rs/U_f**2, 
            marker = "+", ls="--",
            label = "Approximation")
    ax10.plot(y[1:-1]/h,
            np.power(turb_quant_uv*U_f, 2)*rho/U_f**2,
            marker="x", ls="--",
            label = "Calculation from measurements")
    
    
    #Formatting
    ax10.set_xlim([0,1])
    ax10.set_xlabel('\[y/h\:[-]\]')
    ax10.set_ylabel(r"\[\frac{-\rho\overline{u^{'}v^{'}}}{U_f^2}\:\:[-]\]")
    ax10.grid()
    ax10.legend(loc="upper right")
    fig10.savefig(fname="Task_10_plot.svg")
    plt.close(fig10)
else:
    print("Plot for Task 10 not replotted")

#%% Action Item 11
if replot_tasks["T11"]:
    fig11, ax11 = plt.subplots(figsize=(16, 10))
    ax11.scatter(y[1:-1]/h,
                 np.power(turb_quant_uv*U_f, 2)*rho*du_dy[1:-1])
    
    #Formatting
    ax11.set_xlim([0,1])
    ax11.set_xlabel('\[y/h\:[-]\]')
    ax11.set_ylabel(r"\[-\rho\overline{u^{'}v^{'}} "
                    r"\cdot \frac{\partial\overline{u}}{\partial y}\:\:[-]\]")
    ax11.grid()
    fig11.savefig(fname="Task_11_plot.svg")
    plt.close(fig11)
else:
    print("Plot for Task 11 not replotted")