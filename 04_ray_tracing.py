import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel
from obspy.taup.tau import plot_ray_paths

# Initialize the TauP model (IASP91 is standard)
model = TauPyModel(model="iasp91")

print("Generating Ray Paths for Multiple Distances...")

# Define distances in degrees (e.g., every 15 degrees from 0 to 180)
# We want a nice spread to show mantle and core phases
distances = np.arange(10, 180, 20) 

# Phases to plot (Standard seismological phases)
# P, S: Crust/Mantle
# PKP: Core
# PcP: Core-Mantle Boundary reflection
phases = ["P", "S", "PKP", "PcP", "ScS", "PKiKP"]

plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# Use ObsPy's built-in plotting helper, but accessible directly via model
# plot_ray_paths(model, phase_list=phases, min_degrees=0, max_degrees=180, plot_type="spherical", ax=ax)
# The internal function is a bit tricky to style customizedly, let's use the object method if possible
# or just the function wrapping.

# Standard way:
paths = model.get_ray_paths(
    source_depth_in_km=0,
    distance_in_degree=0, # DUMMY, will plot for list below
    phase_list=phases
)
# Wait, get_ray_paths is for single path. plot_ray_paths handles multiples nicely?
# Actually, iterating is safer for control.

from obspy.taup import plot_ray_paths

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Plot for a range of distances
for dist in distances:
    rays = model.get_ray_paths(
        source_depth_in_km=0,
        distance_in_degree=dist,
        phase_list=phases
    )
    rays.plot_rays(plot_type="spherical", ax=ax, legend=False, show=False)

# Add Legend manually or Title
ax.set_title(f"Caminhos de Raios Sísmicos (Modelo IASP91)\nFases: {', '.join(phases)}", fontsize=14, pad=20)

# Save
output_file = 'ray_paths_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Ray Paths plot saved to '{output_file}'")
