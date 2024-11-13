import matplotlib

matplotlib.rcdefaults()
matplotlib.rcParams['text.usetex'] = False            # Use LaTeX for text rendering

# Update font settings
matplotlib.rcParams.update({
    'font.family': 'serif',                          # Use serif font family
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{mathpazo}',  # Use the amsmath and mathpazo package for LaTeX
})

# Customize the figure size
matplotlib.rcParams['figure.figsize'] = (8, 6)   # Set the default figure size

# Customize axes
matplotlib.rcParams['axes.labelsize'] = 22       # Axis label font size
matplotlib.rcParams['axes.titlesize'] = 36       # Axis title font size
matplotlib.rcParams['axes.linewidth'] = 2        # Axis line width

# Customize ticks
matplotlib.rcParams['xtick.labelsize'] = 18      # X-axis tick label size
matplotlib.rcParams['ytick.labelsize'] = 18      # Y-axis tick label size
matplotlib.rcParams['xtick.major.width'] = 1.2   # X-axis major tick width
matplotlib.rcParams['ytick.major.width'] = 1.2   # Y-axis major tick width
matplotlib.rcParams['xtick.minor.size'] = 4      # X-axis minor tick size
matplotlib.rcParams['ytick.minor.size'] = 4      # Y-axis minor tick size
matplotlib.rcParams['xtick.major.size'] = 8      # X-axis major tick size
matplotlib.rcParams['ytick.major.size'] = 8      # Y-axis major tick size

# Customize legend
matplotlib.rcParams['legend.title_fontsize'] = 22 # Legend title font size
matplotlib.rcParams['legend.fontsize'] = 20      # Legend font size
matplotlib.rcParams['legend.frameon'] = True     # Enable/Disable the frame around the legend

# Customize grid
matplotlib.rcParams['grid.color'] = 'gray'       # Grid color
matplotlib.rcParams['grid.linestyle'] = '-'      # Grid line style
matplotlib.rcParams['grid.linewidth'] = 0.5      # Grid line width

# Customize lines
matplotlib.rcParams['lines.linewidth'] = 2.5       # Line width
matplotlib.rcParams['lines.markersize'] = 6        # Marker size

# Change figure and axes background colors
matplotlib.rcParams['figure.facecolor'] = 'white'    # Figure background color
matplotlib.rcParams['axes.facecolor'] = 'white'      # Axes background color