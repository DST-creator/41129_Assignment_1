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
mpl.rcParams['font.size'] = 20
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 25

#Padding
mpl.rcParams['figure.subplot.top'] = .94    #Distance between suptitle and subplots
mpl.rcParams['xtick.major.pad'] = 5         
mpl.rcParams['ytick.major.pad'] = 5
# mpl.rcParams['ztick.major.pad'] = 5
mpl.rcParams['axes.labelpad'] = 20

#Latex font
mpl.rcParams['text.usetex'] = True          #Use standard latex font
mpl.rcParams['font.family'] = 'serif'  # LaTeX default font family
mpl.rcParams["pgf.texsystem"] = "pdflatex"  # Use pdflatex for generating PDFs
mpl.rcParams["pgf.rcfonts"] = False  # Ignore Matplotlib's default font settings
mpl.rcParams['text.latex.preamble'] = "\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
                                                 r'\usepackage{siunitx}'])

# #Custom overline function (cf. https://tex.stackexchange.com/questions/22100/the-bar-and-overline-commands)
# #Note: this slows down the code extremely sometimes 
# plt.rcParams['text.latex.preamble'] = r"""
# \newcommand{\ols}[1]{\mskip.5\thinmuskip\overline{\mskip-.5\thinmuskip {#1} \mskip-.5\thinmuskip}\mskip.5\thinmuskip}""" 

#Export
mpl.rcParams['savefig.bbox'] = "tight"

#%% Input data

replot_tasks = dict(T1=True, 
                    T4=True,
                    T5=True,
                    T6=True,
                    T7=True,
                    T8=True,
                    T9=True,
                    T10=True,
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
    tt = structs[i].tt

    u_means[i] = float(np.sum(structs[i].u*tt))/np.sum(tt)
    v_means[i] = float(np.sum(structs[i].v*tt))/np.sum(tt)
    y[i] = structs[i].y

del tt

#Manually insert start and end values
y = np.append(np.insert(y, 0, 0), .07)
u_means = np.append(np.insert(u_means, 0, 0), .3)

if replot_tasks["T1"]:
    fig1, ax1 = plt.subplots(figsize=(16, 10))
    ax1.scatter(u_means, y, s=150, linewidths=1.5, zorder=2)
    
    #Formatting
    ax1.set_title('Mean velocity')
    ax1.set_ylabel('$y\:[m]$')
    ax1.set_xlabel(r'$\overline{u}\:[m/s]$')
    ax1.grid(zorder=1)
    
    fname = "Task_1_plot"
    fig1.savefig(fname=fname+".svg")
    fig1.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig1.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
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
    ax4.scatter(u_means, y, label = "Measurements", s=100, zorder=2)
    ax4.plot(func_log_layer(y[i_log]),
            y[i_log],
            label="Approximation function",
            ls="-", c='k', zorder=2)
    
    #Formatting
    ax4.set_ylabel('$y\:[m]$')
    ax4.set_xlabel(r'$\overline{u}\:[m/s]$')
    ax4.set_yscale("log")
    ax4.grid(zorder=1)
    ax4.legend(loc="lower right")
    
    fname = "Task_4_plot"
    fig4.savefig(fname=fname+".svg")
    fig4.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig4.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
    plt.close(fig4)
else:
    print("Plot for Task 4 not replotted")

#Recalculate U_f (Eq. 3.43 & 3.44) & y_plus
U_f = popt[0]/2.5
y_plus = y*U_f/nu

#%% Action Item 5

if replot_tasks["T5"]:
    fig5, ax5 = plt.subplots(figsize=(16, 10))
    plt5_sc1 = ax5.scatter(u_means/U_f, y_plus, label = "Measurements", 
                           s=170, linewidths=1.8, zorder=2)
    plt5_line1 = ax5.plot(func_log_layer(y[i_log])/U_f,
                          y_plus[i_log],
                          label="Approximation function",
                          ls="--", c='k', lw=1.5, zorder=2)
    
    plt5_lgd1 = plt.legend(handles=[plt5_sc1, plt5_line1[0]], 
                           loc='upper left')
    plt5_lgd1 = ax5.add_artist(plt5_lgd1)
    
    #Regions
    rect_visc_sub = mpl.patches.Rectangle((-10,0), 40, 5, 
                                          hatch="-\\", fc = "1", alpha = .32,
                                          ec="k", lw=1,
                                          label= "Viscous sublayer", zorder=0)
    rect_visc_buffer = mpl.patches.Rectangle((-10,5), 40, 30-5, 
                                             hatch="/", fc = "1", alpha = .42,
                                             ec="k", lw=1,
                                             label= "Buffer layer", zorder=0)
    rect_log = mpl.patches.Rectangle((-10,30), 40, .1*h*U_f_est/nu-30, 
                                     hatch="\\/",  fc = "1", alpha = .32,
                                     ec="k", lw=1,
                                     label= "Logarithmic layer", zorder=0)
    rect_out = mpl.patches.Rectangle((-10, .1*h*U_f_est/nu), 
                                     40, .9*h*U_f_est/nu, 
                                     hatch="\\", fc = "1", alpha = .42,
                                     ec="k", lw=1,
                                     label= "Outer region", zorder=0)
    
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
    ax5.set_ylabel('$y^+$')
    ax5.set_xlabel(r'$\frac{\overline{u}}{U_f}$')
    ax5.set_yscale("log")
    ax5.set_xlim([0, max(u_means/U_f)*1.05])
    ax5.set_xticks(np.arange(0, int(np.ceil(u_means[-1]/U_f/dx_ticks))*dx_ticks, 
                            dx_ticks))
    ax5.grid(zorder=1)
    fname = "Task_5_plot"
    fig5.savefig(fname=fname+".svg")
    fig5.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig5.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
    
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
# y_plus_vd = np.array([(y_plus[i+1]+y_plus[i])/2 for i in range(len(y_plus)-1)])

if replot_tasks["T6"]:
    if replot_tasks["T5"]:
        plt6_line1 = ax5.plot(u_vD/U_f,
                               y_plus[1:],
                               label="van Driest velocity", 
                               marker = "d", ls="--", c='k', lw=1.5, zorder=2)
        plt5_lgd1 = plt.legend(handles=[plt5_sc1, plt5_line1[0], plt6_line1[0]], 
                               loc='upper left')
        plt5_lgd1 = ax5.add_artist(plt5_lgd1)
        
        fname = "Task_6_plot"
        fig5.savefig(fname=fname+".svg")
        fig5.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
        fig5.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
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
    tt = structs[i].tt
    
    u_var = structs[i].u-u_means[i+1]
    v_var = structs[i].v-v_means[i]
    
    u_var_mean[i] = float(np.sum(np.power(u_var,2)*tt))/np.sum(tt)
    v_var_mean[i] = float(np.sum(np.power(v_var,2)*tt))/np.sum(tt)
    
    turb_quant_u[i] = np.sqrt(u_var_mean[i]) / U_f
    turb_quant_v[i] = np.sqrt(v_var_mean[i]) / U_f
    turb_quant_uv[i] = np.sqrt(-float(np.sum(u_var*v_var*tt))/np.sum(tt))/U_f    
del tt


if replot_tasks["T7"]:
    fig7, ax7 = plt.subplots(figsize=(16, 10))
    
    ax7.scatter(y_plus[1:-1],
                turb_quant_u,
                label=r"$\frac{\sqrt{\overline{u^{\prime 2}}}}{U_f}$", 
                marker = "D", s=100, zorder=2)
    ax7.scatter(y_plus[1:-1],
                turb_quant_v,
                label=r"$\frac{\sqrt{\overline{v^{\prime 2}}}}{U_f}$",
                marker = "h", s=100, facecolor="none", edgecolor="k", 
                linewidth=2, zorder=2)
    ax7.scatter(y_plus[1:-1],
                turb_quant_uv,
                label=r"$\frac{\sqrt{-\overline{u^{\prime}v^{\prime}}}}{U_f}$",
                marker = "v", s=100, zorder=2)
    
# =============================================================================
#     #Empty scatter (needed for the distribution of the labels in the columns 
#     # of the legend)
#     ax7.scatter(1,1, label=' ', c='#ffffff') 
#     
#     #Regions
#     rect_visc_sub = mpl.patches.Rectangle((0,0), 5, 5, 
#                                           hatch="-\\", fc = "1", alpha = .32,
#                                           ec="k", lw=1,
#                                           label= "Viscous sublayer", zorder=0)
#     rect_visc_buffer = mpl.patches.Rectangle((5, 0), 30-5, 5, 
#                                              hatch="/", fc = "1", alpha = .42,
#                                              ec="k", lw=1,
#                                              label= "Buffer layer")
#     rect_log = mpl.patches.Rectangle((30,0), .1*h*U_f/nu-30, 5, 
#                                      hatch="\\/",  fc = "1", alpha = .32,
#                                      ec="k", lw=1,
#                                      label= "Logarithmic layer", zorder=0)
#     rect_out = mpl.patches.Rectangle((.1*h*U_f/nu, 0), 
#                                      .9*h*U_f/nu, 5, 
#                                      hatch="\\", fc = "1", alpha = .42,
#                                      ec="k", lw=1,
#                                      label= "Outer region", zorder=0)
#      
#     ax7.add_patch(rect_visc_sub)
#     ax7.add_patch(rect_visc_buffer)
#     ax7.add_patch(rect_log)
#     ax7.add_patch(rect_out)
# =============================================================================
    
    #Region boundaries
    ax7.axvline(5, ls="--", c="k", lw=1.5)
    ax7.axvline(30, ls="--", c="k", lw=1.5)
    ax7.axvline(.1*h*U_f/nu, ls="--", c="k", lw=1.5)
    
    #Annotations for regions
    arrowstyle = dict(arrowstyle="->", 
                      connectionstyle="angle,angleA=0,angleB=90")
    ax7.text(0,4.9, " ") #To make space for annotations
    ax7.annotate("Viscous sublayer", (2.5,4.5), (5,4.85), arrowprops = arrowstyle)
    ax7.annotate("Buffer layer", (0,4.5), (12,4.6))
    ax7.annotate("Logarithmic layer", (0,4.5), (55,4.6))
    ax7.annotate("Outer region", (.1*h*U_f/nu+3, 4.5), (.1*h*U_f/nu+3-15,4.85), 
                 arrowprops = arrowstyle)
    
    
    #Formatting
    ax7.set_xlim([0,100])
    ax7.set_ylim([0,4.5])
    ax7.set_xlabel('$y^+$')
    ax7.set_ylabel(r'Turbulence quantities')
    ax7.grid(zorder=1)
    ax7.legend(loc="upper right", framealpha=1)
    
    fname = "Task_7_plot"
    fig7.savefig(fname=fname+".svg")
    fig7.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig7.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
    
    plt.close(fig7)
else:
    print("Plot for Task 7 not replotted")

#%% Action Item 8
if replot_tasks["T8"]:
    fig8, ax8 = plt.subplots(figsize=(16, 10))
    ax8.scatter(y[1:-1]/h,
                turb_quant_u,
                label=r"$\frac{\sqrt{\overline{u^{\prime 2}}}}{U_f}$",
                marker = "d", s=80, zorder=2)
    ax8.scatter(y[1:-1]/h,
                turb_quant_v,
                label=r"$\frac{\sqrt{\overline{v^{\prime 2}}}}{U_f}$",
                marker = "h", s=100, facecolor="none", edgecolor="k", 
                linewidth=2, zorder=2)
    ax8.scatter(y[1:-1]/h,
                turb_quant_uv,
                label=r"$\frac{-\overline{u^{\prime}v^{\prime}}}{U_f^2}$",
                marker = "v", s=100, zorder=2)
    
    #Formatting
    ax8.set_xlim([0,1])
    ax8.set_xlabel('$y/h$')
    ax8.set_ylabel(r'Turbulence quantities')
    ax8.grid(zorder=1)
    ax8.legend(loc="upper right")
    
    fname = "Task_8_plot"
    fig8.savefig(fname=fname+".svg")
    fig8.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig8.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
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
            label=r"\[k/U_f^2\]", 
            zorder=2)
    
    #Formatting
    ax9.set_xlim([0,1])
    ax9.set_xlabel('$y/h$')
    ax9.set_ylabel(r'$\frac{k}{U_f^2}$')
    ax9.grid(zorder=1)
    
    fname = "Task_9_plot"
    fig9.savefig(fname=fname+".svg")
    fig9.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig9.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
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
    ax10.scatter(y[1:-1]/h,
            np.power(turb_quant_uv*U_f, 2)*rho/U_f**2,
            label = "Calculation from measurements", 
            marker = "+", s=100, linewidth=1.8, zorder=2)
    ax10.scatter(y/h,
            rs/U_f**2, 
            marker = "d", s=80,
            label = "Approximation", zorder=2)
    
    
    
    #Formatting
    ax10.set_xlim([0,1])
    ax10.set_xlabel('$y/h$')
    ax10.set_ylabel(r"$\frac{-\rho\overline{u^{\prime}v^{\prime}}}{U_f^2}$")
    ax10.grid(zorder=1)
    ax10.legend(loc="upper right")
    
    fname = "Task_10_plot"
    fig10.savefig(fname=fname+".svg")
    fig10.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig10.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
    plt.close(fig10)
else:
    print("Plot for Task 10 not replotted")

#%% Action Item 11
if replot_tasks["T11"]:
    fig11, ax11 = plt.subplots(figsize=(16, 10))
    ax11.scatter(y[1:-1]/h,
                 np.power(turb_quant_uv*U_f, 2)*rho*du_dy[1:-1], zorder=2)
    
    #Formatting
    ax11.set_xlim([0,1])
    ax11.set_xlabel('$y/h$')
    ax11.set_ylabel(r"$-\rho\overline{u^{\prime}v^{\prime}}"
                    r"\cdot \frac{\partial\overline{u}}{\partial y}$")
    ax11.grid(zorder=1)
    
    fname = "Task_11_plot"
    fig11.savefig(fname=fname+".svg")
    fig11.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig11.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
    plt.close(fig11)
else:
    print("Plot for Task 11 not replotted")