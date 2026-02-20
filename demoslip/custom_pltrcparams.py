import matplotlib.pyplot as plt


def update_pltstuff(scale_factor) :



    scale_default_rc_params_dict = {
    "axes.labelpad": 4.0,
    "axes.labelsize": "medium",
    "axes.linewidth": 0.8,
    "axes.titlepad": 6.0,
    "axes.titlesize": "large",

    "contour.linewidth": None,
    "figure.constrained_layout.h_pad": 0.04167,
    "figure.constrained_layout.hspace": 0.02,
    "figure.constrained_layout.w_pad": 0.04167,
    "figure.constrained_layout.wspace": 0.02,
    "figure.dpi": 100.0,
    "figure.figsize": [6.4, 4.8],
    "figure.subplot.bottom": 0.11,
    "figure.subplot.hspace": 0.2,
    "figure.subplot.left": 0.125,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.88,
    "figure.subplot.wspace": 0.2,
    "font.size": 10.0,
    "grid.linewidth": 0.8,
    "legend.borderaxespad": 0.5,
    "legend.borderpad": 0.4,
    "legend.columnspacing": 2.0,
    "legend.edgecolor": 0.8,
    "legend.fontsize": "medium",
    "legend.handleheight": 0.7,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.8,
    "legend.labelspacing": 0.5,
    "legend.markerscale": 1.0,
    "lines.dashdot_pattern": [6.4, 1.6, 1.0, 1.6],
    "lines.dashed_pattern": [3.7, 1.6],
    "lines.dotted_pattern": [1.0, 1.65],
    "lines.linewidth": 1.5,
    "lines.markersize": 6.0,
    "path.simplify_threshold": 0.111111111111,
    "savefig.pad_inches": 0.1,
    "text.hinting_factor": 8,
    "xtick.major.pad": 3.5,
    "xtick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "xtick.minor.pad": 3.4,
    "xtick.minor.size": 2.0,
    "xtick.minor.width": 0.6,
    "ytick.major.pad": 3.5,
    "ytick.major.size": 3.5,
    "ytick.major.width": 0.8,
    "ytick.minor.pad": 3.4,
    "ytick.minor.size": 2.0,
    "ytick.minor.width": 0.6
    }
    


    plt.rcParams['xtick.major.size']= 3.5 *scale_factor
    plt.rcParams['xtick.major.width'] = 0.8 *scale_factor
    plt.rcParams['xtick.minor.size'] = 2.0 *scale_factor
    plt.rcParams['xtick.minor.width'] = 0.6 *scale_factor

    plt.rcParams['ytick.major.size']= 3.5 *scale_factor
    plt.rcParams['ytick.major.width'] = 0.8 *scale_factor
    plt.rcParams['ytick.minor.size'] = 2.0 *scale_factor
    plt.rcParams['ytick.minor.width'] = 0.6 *scale_factor
    plt.rcParams['axes.linewidth'] = 0.6 *scale_factor
    

    #     'xtick.labelsize': 6 * scale_factor,        # Scale x-tick labels
    #     'ytick.labelsize': 6 * scale_factor,        # Scale y-tick labels
    #     'legend.fontsize': 6 * scale_factor,        # Scale legends
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 8*scale_factor}) # go 6
    #     'axes.labelsize': 5 * scale_factor,         # Scale axis labels
    #     'axes.titlesize': 5 * scale_factor,        # Scale titles
    #     'lines.linewidth': 1.2 * scale_factor,      # Scale line widths 
    
  
    # # Update rcParams globally
    # plt.rcParams.update({  

    
    # }) 
