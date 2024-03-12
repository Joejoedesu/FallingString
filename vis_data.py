import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import PIL
import imageio.v2 as imageio
import os
import subprocess

def draw_data_comp(index=0, env='floor', gran_sim=100, gran_gen=1, h_size=32, layer_num=1, loss_method="3p", ext=False):
    file_sim = "data/sim/" + f"location_{env}_{index}"
    file_gen = "data/gen/" + f"location_{env}_{index}_{h_size}_l{layer_num}_{loss_method}"
    if ext:
        file_sim += "_ext"
        file_gen += "_ext"
    data_sim = np.load(file_sim + ".npy")
    data_gen = np.load(file_gen + ".npy")

    data_sim_t = []
    data_gen_t = []
    for j in range(len(data_sim)):
        if j % gran_sim == 0:
            data_sim_t.append(data_sim[j])
    for j in range(len(data_gen)):
        if j % gran_gen == 0:
            data_gen_t.append(data_gen[j])
    
    assert len(data_sim_t) == len(data_gen_t)
    for j in range(len(data_sim_t)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_zlim(-1, 10)
        ax.plot(data_sim_t[j][:, 0], data_sim_t[j][:, 1], data_sim_t[j][:, 2], 'r')
        ax.plot(data_gen_t[j][:, 0], data_gen_t[j][:, 1], data_gen_t[j][:, 2], 'b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #save to frame folder
        plt.savefig('frames/frame' + str(j) + '.png')
        plt.close(fig)
    
    image_list = []
    J_range = 40
    if ext:
        J_range = 80
    for j in range(J_range):
        image_list.append(imageio.imread('frames/frame' + str(j) + '.png'))
    # imageio.mimsave(f'movie_{file}.gif', image_list)
    if ext:
        imageio.mimsave(f'clips/movie_compare_{env}_{index}_early_{h_size}_l{layer_num}_{loss_method}_ext.gif', image_list)
    else:
        imageio.mimsave(f'clips/movie_compare_{env}_{index}_early_{h_size}_l{layer_num}_{loss_method}.gif', image_list)

def draw_data(index=0, env='floor', sour="sim",gran=100, h_size=32, layer_num=1, loss_method="3p"):
    dir = f"data/{sour}/"
    if sour == "sim":
        file = f"location_{env}_{index}"
    else:
        file = f"location_{env}_{index}_{h_size}_l{layer_num}_{loss_method}"
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
    if sour == "sim":
        imageio.mimsave(f'clips/movie_{sour}_{env}_{index}.gif', image_list)
    else:
        imageio.mimsave(f'clips/movie_{sour}_{env}_{index}_early_{h_size}_l{layer_num}_{loss_method}.gif', image_list)

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

def get_force(loc, env="floor"):
    assert len(loc) == 3

    if env == "floor":
        if loc[2] > 3:
            return [0, 0, -9.8]
        else:
            return [0, 0, 0]

    elif env == "spring":
        if loc[2] > 3:
            return [0, 0, -9.8]
        else:
            return [0, 0, 15 * (3-loc[2])**2]
        
    elif env == "wind":
        center = np.array([1, 1])
        angle = np.arctan2(loc[1]-center[1], loc[0]-center[0])
        dir = np.array([np.cos(angle), np.sin(angle)])
        dis = np.linalg.norm(loc[:2]-center)
        spin = 0.5 * min(0, 2-dis)
        attr = -0.5 * dis
        f_x = np.sin(angle) * spin + dir[0] * attr
        f_y = np.cos(angle) * spin + dir[1] * attr
        lim = 3
        if loc[2] > lim:
            return [f_x, f_y, -9.8]
        else:
            return [f_x, f_y, (lim-loc[2])**2 * 8]

def draw_field(env="floor"):
    save_dir = "report/fig"
    envs = ["floor", "spring", "wind"]
    x, y, z = np.meshgrid(np.linspace(-1, 4, 5), np.linspace(-1, 4, 5), np.linspace(-1, 10, 8))
    u, v, w = np.zeros(x.shape), np.zeros(y.shape), np.zeros(z.shape)
    colormap = mpl.colormaps["cool"]
    for env in envs:
        M = np.zeros(x.shape)
        for i in range(5):
            for j in range(5):
                for k in range(8):
                    loc = [x[i, j, k], y[i, j, k], z[i, j, k]]
                    force = get_force(loc, env)
                    if np.linalg.norm(force) == 0:
                        u[i, j, k], v[i, j, k], w[i, j, k] = 0, 0, 0
                    else:
                        u[i, j, k], v[i, j, k], w[i, j, k] = force / np.linalg.norm(force)

                    M[i, j, k] = np.linalg.norm(force)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        min_M = np.min(M)
        max_M = np.max(M)
        M = colormap((M - np.min(M)) / np.max(M))
        M = M.reshape(-1, 4)
        ax.quiver(x, y, z, u, v, w, color='black', length=0.8, arrow_length_ratio=0.6, pivot='tail')
        ax.scatter(x, y, z, c=M)
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_zlim(-1, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(env)
        # color bar
        ran = mpl.colors.Normalize(vmin=min_M, vmax=max_M)
        fig.colorbar(plt.cm.ScalarMappable(norm=ran, cmap=colormap), ax=ax, orientation='vertical')
        plt.savefig(f'{save_dir}/{env}_force_field.png')
        plt.show()
        plt.close(fig)

def draw_error(env="floor", id=0, h_size=32, layer_num=1, loss_method="3p"):
    pos_e = np.load(f"log/detailed/pos_error_{env}_{id}_{h_size}_l{layer_num}_{loss_method}.npy")
    pos_e = np.mean(pos_e, axis=1)
    vel_e = np.load(f"log/detailed/vel_error_{env}_{id}_{h_size}_l{layer_num}_{loss_method}.npy")
    vel_e = np.mean(vel_e, axis=1)
    com_e = np.load(f"log/detailed/com_error_{env}_{id}_{h_size}_l{layer_num}_{loss_method}.npy")
    len_e = np.load(f"log/detailed/len_error_{env}_{id}_{h_size}_l{layer_num}_{loss_method}.npy")
    print(pos_e.shape)
    print(vel_e.shape)
    print(com_e.shape)
    print(len_e.shape)
    plt.plot(pos_e, label="position")
    plt.plot(vel_e, label="velocity")
    plt.plot(com_e, label="center of mass")
    plt.plot(len_e, label="length")
    plt.xlabel("time step")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"{env} error")
    plt.show()

if __name__ == "__main__":
    # draw_data(index=id)
    # draw_data(index=id,sour="gen",gran=1, h_size=32, layer_num=1)
    # confirm_start()
    # draw_data_comp(index=0, env='spring', gran_sim=100, gran_gen=1, h_size=128, layer_num=3, loss_method="3p", ext=True)
    # draw_data(index=198, env='wind', sour="gen", gran=1, h_size=128, layer_num=3, loss_method="3p")
    # draw_data(index=168, env='spring', sour="sim", gran=100, h_size=256, layer_num=3, loss_method="3p")
    # draw_data(index=194, env='wind', sour="sim", gran=100, h_size=256, layer_num=3, loss_method="3p")

    # envs = ["floor", "spring", "wind"]
    # #create 3 subplots
    # plt.subplots(1, 3, figsize=(5, 1))
    # for i in range(3):
    #     env = envs[i]
    #     train_e = np.load(f"log/error_{env}_128_l3_3p.npy")
    #     test_e = np.load(f"log/test_error_{env}_128_l3_3p.npy")
    #     plt.subplot(1, 3, i+1)
    #     plt.plot(train_e, label="train")
    #     plt.plot(test_e, label="test")
    #     # y_max = max(max(train_e), max(test_e)) * 1.1
    #     # plt.ylim(0, y_max)
    #     plt.xlabel("epoch")
    #     plt.ylabel("error")
    #     plt.legend()
    #     plt.title(f"{env} error")

    draw_error(env="floor", id=180, h_size=128, layer_num=3, loss_method="3p")
    draw_error(env="spring", id=193, h_size=128, layer_num=3, loss_method="3p")
    draw_error(env="wind", id=186, h_size=128, layer_num=3, loss_method="3p")
