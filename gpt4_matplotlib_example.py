import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Generate random data for the heatmaps
data = [np.random.randn(10, 10) for _ in range(4)]

# Create the figure and a 2x2 grid specification
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1], wspace=0.5, hspace=0.5)

# For each data (heatmap) in the 2x2 grid, create an axes for the heatmap and another for the colorbar
for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs[i, 2*j])
        cax = fig.add_subplot(gs[i, 2*j + 1])
        
        im = ax.imshow(data[2*i + j], cmap='viridis', origin='lower', aspect='auto')
        fig.colorbar(im, cax=cax, orientation='vertical')

plt.savefig('gpt4_matplotlib_example.png')
