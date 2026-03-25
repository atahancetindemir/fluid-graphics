import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

smoke_files = sorted(glob.glob("frames/smoke_*.txt"))
u_files = sorted(glob.glob("frames/u_*.txt"))
v_files = sorted(glob.glob("frames/v_*.txt"))

num_frames = len(smoke_files)
if num_frames == 0:
    print("File Not Found")
    exit()

smoke = np.loadtxt(smoke_files[0])
u_raw = np.loadtxt(u_files[0])
v_raw = np.loadtxt(v_files[0])

u = u_raw[:smoke.shape[0], :smoke.shape[1]]
v = v_raw[:smoke.shape[0], :smoke.shape[1]]

fig, ax = plt.subplots(figsize=(10, 10))

cax = ax.imshow(smoke, cmap='magma')
fig.colorbar(cax, label='Smoke')

Y_grid, X_grid = np.mgrid[0:smoke.shape[0], 0:smoke.shape[1]]

quiv = ax.quiver(X_grid, Y_grid, u, v, color='cyan', scale=200, alpha=0.8)
quiv.set_visible(False)

ax.set_title("2D Fluid Simulation - Frame: 0")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")

def update(frame_idx):

    smoke_new = np.loadtxt(smoke_files[frame_idx])
    u_raw_new = np.loadtxt(u_files[frame_idx])
    v_raw_new = np.loadtxt(v_files[frame_idx])
    
    u_new = u_raw_new[:smoke_new.shape[0], :smoke_new.shape[1]]
    v_new = v_raw_new[:smoke_new.shape[0], :smoke_new.shape[1]]
    
    cax.set_data(smoke_new)
    
    max_val = np.max(smoke_new)
    cax.set_clim(vmin=0, vmax=max_val if max_val > 0 else 1)
    
    quiv.set_UVC(u_new, v_new)
    
    ax.set_title(f"2D Fluid Simulation - Frame: {frame_idx}")
    
    return cax, quiv

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=5, blit=False)

plt.show()

# ani.save("fluid_sim.gif", writer="pillow", fps=20)