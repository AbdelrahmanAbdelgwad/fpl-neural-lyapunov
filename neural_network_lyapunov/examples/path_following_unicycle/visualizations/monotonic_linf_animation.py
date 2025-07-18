#%%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import matplotlib
import os
import numpy as np

saveDir = '/home/zw2445/Documents/neural-network-lyapunov/plots/pf_linf_contour_animation2/' 
image_folder =  saveDir
video_name = 'pf_contour_animation2.mp4'

fig = plt.figure()
plt.axis('off')
images_len = len([img for img in os.listdir(image_folder) if img.endswith(".png")])
frames = [] 
for i in range(1,images_len+1):
    frames.append([plt.imshow(plt.imread(saveDir+'pf_linf_bound'+str(i)+'_contour.png'),animated=True)])
# for i in range(1,images_len+1):
#     # if i%5 != 0:
#     frames.append([plt.imshow(plt.imread(saveDir+str(i)+'.png'),animated=True)])
    

anim = animation.ArtistAnimation(fig, frames, interval=1, blit=True)
# saving to m4 using ffmpeg writer
writervideo = animation.FFMpegWriter(fps=1)#,bitrate=400)
anim.save(saveDir+video_name, writer=writervideo, dpi=600)
plt.close()

# plt.rc('grid', color='#397939', linewidth=1, linestyle='-')
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=5)

# width, height = matplotlib.rcParams['figure.figsize']
# size = min(width, height)
# fig = plt.figure(figsize=(size, size))
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, facecolor='#cfd98c')


# ax.set_rmax(20.0)
# plt.grid(True)

# def data_gen(t=0):
#     tw = 10
#     phase = 0
#     while True:
#         if phase < 2*180:
#             phase += 2
#         else:
#             phase=0
#         yield tw, phase

# arr1 = [None]
# def update(data):
#     tw, phase = data
#     ax.set_title(u"|TW| = {}, Angle: {}°".format(tw, phase))
#     if arr1[0]: arr1[0].remove()
#     arr1[0] = ax.arrow(np.deg2rad(phase), 0, 0, tw, alpha = 0.5, width = 0.080,
#              edgecolor = 'red', facecolor = 'red', lw = 2, zorder = 5)

# ani = animation.FuncAnimation(fig, update, data_gen, interval=100, blit=False, repeat=False)

# plt.show()
# %%
