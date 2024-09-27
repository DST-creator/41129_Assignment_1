import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure Matplotlib for LaTeX rendering
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",   # or "xelatex" or "lualatex" if you have these installed
    "font.family": "serif",        # Use serif font
    "text.usetex": True,           # Enable LaTeX rendering for all text
    "pgf.rcfonts": False,          # Don't override fonts with matplotlib defaults
    'text.latex.preamble':"\n".join([r'\usepackage{amsmath}',  # Optional, for math symbols
                                                     r'\usepackage{siunitx}'])
})

# Create a figure
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 9, 16], label=r'$y$')  # Ensure proper escaping in labels
ax.set_xlabel(r"$\frac{\sqrt{\overline{u^{\prime 2}}}}{U_f}$")  # Latex styled label
ax.set_ylabel(r'$Y$')  # Proper escaping for brackets and special chars
ax.legend()

# Export the figure
fig.savefig("figure.pdf")  # Save as PDF
fig.savefig("figure.pgf")  # Save PGF file
