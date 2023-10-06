import matplotlib.pyplot as plt
import numpy as np

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Generate random 10x10 standard Gaussian noise for each heatmap
heatmaps = [np.random.randn(10, 10) for _ in range(4)]

# Create heatmaps and colorbars
for i, ax in enumerate(axes.flat):
    heatmap = ax.imshow(heatmaps[i], cmap='viridis', aspect='auto')
    cbar = fig.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('Colorbar Label')
    ax.set_title(f'Heatmap {i+1}')

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.savefig('gpt35_matplotlib_example.png')
