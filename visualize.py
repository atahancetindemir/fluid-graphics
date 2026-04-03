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

cax = ax.imshow(smoke, cmap='turbo', interpolation='bilinear')
fig.colorbar(cax, label='Smoke Concentration', fraction=0.046, pad=0.04)

step = 8
def get_averaged_vectors(u_field, v_field, step):

    trim_y = u_field.shape[0] - (u_field.shape[0] % step)
    trim_x = u_field.shape[1] - (u_field.shape[1] % step)
    
    u_trim = u_field[:trim_y, :trim_x]
    v_trim = v_field[:trim_y, :trim_x]
    
    u_avg = u_trim.reshape(trim_y // step, step, trim_x // step, step).mean(axis=(1, 3))
    v_avg = v_trim.reshape(trim_y // step, step, trim_x // step, step).mean(axis=(1, 3))
    
    return u_avg, v_avg

u_avg, v_avg = get_averaged_vectors(u, v, step)

y_coords = np.arange(1, smoke.shape[0], step)
x_coords = np.arange(1, smoke.shape[1], step)
X_grid_avg, Y_grid_avg = np.meshgrid(x_coords, y_coords)

quiv = ax.quiver(X_grid_avg, Y_grid_avg, u_avg, v_avg, color='white', scale=5, alpha=0.8, pivot='mid', width=0.003)
quiv.set_visible(True)
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
    
    u_avg_new, v_avg_new = get_averaged_vectors(u_new, v_new, step)
    quiv.set_UVC(u_avg_new, v_avg_new)
    
    ax.set_title(f"2D Fluid Simulation - Frame: {frame_idx}")
    
    return cax, quiv

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

plt.show()