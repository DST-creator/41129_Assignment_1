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

#Figure size:
mpl.rcParams['figure.figsize'] = (16, 8)  

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
mpl.rcParams['font.size'] = 25
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
mpl.rcParams.update({"pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])})


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
    fig1, ax1 = plt.subplots()
    ax1.scatter(u_means, y, s=150, linewidths=1.5, zorder=2)
    
    #Formatting
    ax1.set_title('Mean velocity')
    ax1.set_ylabel(r'$y\:\unit{[\m]}$')
    ax1.set_xlabel(r'$\overline{u}\:\unit{[\m/\s]}$')
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
    fig4, ax4 = plt.subplots()
    ax4.scatter(u_means, y, label = "Measurements", s=100, zorder=2)
    ax4.plot(func_log_layer(y[i_log]),
            y[i_log],
            label="Approximation function",
            ls="-", c='k', zorder=2)
    
    #Formatting
    ax4.set_ylabel(r'$y\:\unit{[\m]}$')
    ax4.set_xlabel(r'$\overline{u}\:\unit{[\m/\s]}$')
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
    fig5, ax5 = plt.subplots()
    plt5_sc1 = ax5.scatter(u_means/U_f, y_plus, label = "Measurements", 
                           s=170, linewidths=1.8, zorder=2)
    plt5_line1 = ax5.plot(func_log_layer(y[i_log])/U_f,
                          y_plus[i_log],
                          label="Approximation function",
                          ls="--", c='k', lw=1.5, zorder=2)
    
    #Region boundaries
    ax5.axhline(5, ls="--", c="k", lw=1.4)
    ax5.axhline(30, ls="--", c="k", lw=1.4)
    ax5.axhline(.1*h*U_f/nu, ls="--", c="k", lw=1.4)
    
    #Annotations for regions
    arrowstyle = dict(arrowstyle="<->", 
                      connectionstyle="angle,angleA=90,angleB=0")
    ax5.text(30,1, " ") #To make space for annotations
    # # Drawing arrows for the regions
    ax5.annotate("", (1.02,0), (0,85), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle)
    ax5.text(25, 3, "Viscous sublayer", ha='left', va='center')
    
    ax5.annotate("", (1.02,.14), (0,170), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle)
    ax5.text(25, 13, "Buffer layer", ha='left', va='center')
    
    ax5.annotate("", (1.02,.42), (0,105), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle)
    ax5.text(25, 50, "Logarithmic layer", ha='left', va='center')
    
    ax5.annotate("", (1.02,.59), (0,223), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle)
    ax5.text(25, 300, "Outer region", ha='left', va='center')

    #Formatting
    dx_ticks = 2
    ax5.set_ylabel('$y^+$')
    ax5.set_xlabel(r'$\frac{\overline{u}}{U_f}$')
    ax5.set_yscale("log")
    ax5.set_xlim([0, max(u_means/U_f)*1.05])
    ax5.set_xticks(np.arange(0, int(np.ceil(u_means[-1]/U_f/dx_ticks))*dx_ticks, 
                            dx_ticks))
    ax5.grid(zorder=1)
    plt.legend(loc='upper left')
    
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

if replot_tasks["T6"]:
    if replot_tasks["T5"]:
        #Clear Approximation function for Log region from plot
        list(ax5.lines)[0].remove()
 
        #Plot van Driest
        plt6_line1 = ax5.plot(u_vD/U_f,
                               y_plus[1:],
                               label="van Driest velocity", 
                               marker = "v", ms=8, ls="-.", c='k', 
                               zorder=2)
        plt.legend(loc='upper left')
        
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
    fig7, ax7 = plt.subplots()
    
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
    
    #Region boundaries
    ax7.axvline(5, ls="--", c="k", lw=1.4)
    ax7.axvline(30, ls="--", c="k", lw=1.4)
    ax7.axvline(.1*h*U_f/nu, ls="--", c="k", lw=1.4)
    
    #Annotations for regions
    arrowstyle = dict(arrowstyle="<->", 
                      connectionstyle="angle,angleA=90,angleB=0")
    ax7.text(0,4.9, " ") #To make space for annotations
    # Drawing arrows for the regions
    ax7.annotate("", (0,1.02), (45, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle)
    ax7.annotate("Viscous\nsublayer", (0,0), (5,4.7), 
                 xycoords='data', ha='right', va='bottom')
    
    ax7.annotate("", (.05,1.02), (225, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle)
    ax7.annotate("Buffer layer", (0,0), (18,4.7), 
                 xycoords='data', ha='center', va='bottom')
    
    ax7.annotate("", (.3,1.02), (560, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle)
    ax7.annotate("Logarithmic layer", (0,0), (60,4.7), 
                 xycoords='data', ha='center', va='bottom')
    
    ax7.annotate("", (.1*h*U_f/nu/100,1.02), (70, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', 
                 arrowprops = dict(arrowstyle="->", 
                                   connectionstyle="angle,angleA=90,angleB=0"))
    ax7.annotate("Outer\nregion", (0,0), (.1*h*U_f/nu*1.01,4.7), 
                 xycoords='data', ha='left', va='bottom')
    
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
    fig8, ax8 = plt.subplots()
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
                label=r"$\frac{\sqrt{-\overline{u^{\prime}v^{\prime}}}}{U_f}$",
                marker = "v", s=100, zorder=2)
    
    #Region boundaries
    ax8.axvline(5/U_f*nu/h, ls="--", c="k", lw=1.4)
    ax8.axvline(30/U_f*nu/h, ls="--", c="k", lw=1.4)
    ax8.axvline(.1, ls="--", c="k", lw=1.4)
    
    #Annotations for regions
    arrowstyle_lin = dict(arrowstyle="-", 
                      connectionstyle="angle,angleA=0,angleB=90")
    arrowstyle_point = dict(arrowstyle="<->", 
                      connectionstyle="angle,angleA=90,angleB=0")
    ax8.text(0,3.2, " ") #To make space for annotations
    # Drawing arrows for the regions
    ax8.annotate("Viscous sublayer", (0.002,1.03), (40, 90), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax8.annotate("", (.004,1.02), (28, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax8.annotate("Buffer layer", (0.018,1.03), (40, 55), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax8.annotate("", (.03,1.02), (65, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax8.annotate("Logarithmic layer", (0.065,1.03), (40, 20), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax8.annotate("", (.1,1.02), (805, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax8.annotate("Outer region", (0.55,1.03), (0, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='center', va='bottom')
    
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
    fig9, ax9 = plt.subplots()
    ax9.scatter(y[1:-1]/h,
            tke,
            label=r"\[k/U_f^2\]", 
            marker = "d", s=70, zorder=2)
    
    #Region boundaries
    ax9.axvline(5/U_f*nu/h, ls="--", c="k", lw=1.4)
    ax9.axvline(30/U_f*nu/h, ls="--", c="k", lw=1.4)
    ax9.axvline(.1, ls="--", c="k", lw=1.4)
    
    #Annotations for regions
    arrowstyle_lin = dict(arrowstyle="-", 
                      connectionstyle="angle,angleA=0,angleB=90")
    arrowstyle_point = dict(arrowstyle="<->", 
                      connectionstyle="angle,angleA=90,angleB=0")
    ax9.text(0,3.2, " ") #To make space for annotations
    # Drawing arrows for the regions
    ax9.annotate("Viscous sublayer", (0.002,1.03), (40, 90), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax9.annotate("", (.004,1.02), (28, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax9.annotate("Buffer layer", (0.018,1.03), (40, 55), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax9.annotate("", (.03,1.02), (65, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax9.annotate("Logarithmic layer", (0.065,1.03), (40, 20), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax9.annotate("", (.1,1.02), (805, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax9.annotate("Outer region", (0.55,1.03), (0, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='center', va='bottom')
    
    #Formatting
    ax9.set_xlim([0,1])
    ax9.set_ylim([0,np.ceil(np.max(tke))])
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
    fig10, ax10 = plt.subplots()
    ax10.scatter(y[1:-1]/h,
            np.power(turb_quant_uv*U_f, 2)*rho/U_f**2,
            label = "Calculations from measurements", 
            marker = "+", s=100, linewidth=1.8, zorder=2)
    ax10.plot(y/h,
            rs/U_f**2, 
            marker = "v", ms=8, ls="--", c='k', lw=1.5, zorder=2,
            label = "Approximation")
    
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
    fig11, ax11 = plt.subplots()
    ax11.scatter(y[1:-1]/h,
                 np.power(turb_quant_uv*U_f, 2)*rho*du_dy[1:-1],
                 marker = "d", s=70, zorder=2)
    
    #Region boundaries
    ax11.axvline(5/U_f*nu/h, ls="--", c="k", lw=1.4)
    ax11.axvline(30/U_f*nu/h, ls="--", c="k", lw=1.4)
    ax11.axvline(.1, ls="--", c="k", lw=1.4)
    
    #Annotations for regions
    arrowstyle_lin = dict(arrowstyle="-", 
                      connectionstyle="angle,angleA=0,angleB=90")
    arrowstyle_point = dict(arrowstyle="<->", 
                      connectionstyle="angle,angleA=90,angleB=0")
    ax11.text(0,3.2, " ") #To make space for annotations
    # Drawing arrows for the regions
    ax11.annotate("Viscous sublayer", (0.002,1.03), (40, 90), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax11.annotate("", (.004,1.02), (28, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax11.annotate("Buffer layer", (0.018,1.03), (40, 55), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax11.annotate("", (.03,1.02), (65, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax11.annotate("Logarithmic layer", (0.065,1.03), (40, 20), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='left', va='bottom', arrowprops = arrowstyle_lin)
    
    ax11.annotate("", (.1,1.02), (805, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 va='top', arrowprops = arrowstyle_point)
    ax11.annotate("Outer region", (0.55,1.03), (0, 0), 
                 xycoords='axes fraction', textcoords='offset points', 
                 ha='center', va='bottom')
    
    
    #Formatting
    ax11.set_xlim([0,1])
    ax11.set_xlabel('$y/h$')
    ax11.set_ylabel(r"$-\rho\overline{u^{\prime}v^{\prime}}"
                    r"\cdot \frac{\partial\overline{u}}{\partial y}"
                    + r"\:\unit{\left[\frac{\kg}{\m\cdot\s}\right]}$")
    ax11.grid(zorder=1)
    
    fname = "Task_11_plot"
    fig11.savefig(fname=fname+".svg")
    fig11.savefig(fname+".pdf", format="pdf")       # Save PDF for inclusion
    fig11.savefig(fname+".pgf")                     # Save PGF file for text inclusion in LaTeX
    plt.close(fig11)
else:
    print("Plot for Task 11 not replotted")