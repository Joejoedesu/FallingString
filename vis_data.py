import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import PIL
import imageio.v2 as imageio
import os
import subprocess

def draw_data(index=0, env='floor', sour="sim",gran=100, h_size=32, layer_num=1):
    dir = f"data/{sour}/"
    file = f"location_{env}_{index}_{h_size}_l{layer_num}"
    data = np.load(dir + file + ".npy")
    for j in range(len(data)):
        if j % gran == 0:
            # print(data[i])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-1, 4)
            ax.set_ylim(-1, 4)
            ax.set_zlim(-1, 10)
            ax.plot(data[j][:, 0], data[j][:, 1], data[j][:, 2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            #save to frame folder
            plt.savefig('frames/frame' + str(j//gran) + '.png')
            plt.close(fig)
    
    image_list = []
    for j in range(40):
        image_list.append(imageio.imread('frames/frame' + str(j) + '.png'))
    # imageio.mimsave(f'movie_{file}.gif', image_list)
    imageio.mimsave(f'clips/movie_{sour}_{index}_early_{h_size}_l{layer_num}.gif', image_list)

def confirm_start():
    sim_dir = "data/sim/"
    sta_dir = "data/start/"
    for i in range(40):
        inst = f"floor_{i}"
        sim = np.load(sim_dir + f"location_{inst}.npy")
        sta = np.load(sta_dir + f"spline_{inst}.npy")

        start_location = sim[0]

        #plot start_location and the spline
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(start_location[:, 0], start_location[:, 1], start_location[:, 2], 'r')
        ax.plot(sta[:, 0], sta[:, 1], sta[:, 2], 'b')
        plt.show()


if __name__ == "__main__":
    id = 182
    # draw_data(index=id)
    draw_data(index=id,sour="gen",gran=1, h_size=32, layer_num=1)
    # confirm_start()