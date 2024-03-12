import MLP_model
import vis_data
import numpy as np
import matplotlib.pyplot as plt

def sweep_model():
    # MLP_model.create_train_data_3p("wind", [0, 140], 256, 4096, "train")
    # MLP_model.create_train_data_3p("wind", [140, 200], 256, 1, "test")
    # h_sizes = [16, 32, 64, 128, 256]
    h_sizes = [64, 128, 256]
    # h_sizes = [256]
    # layer_nums = [1, 2, 3]
    layer_nums = [2, 3]
    ids = [186, 189, 194, 195]
    iteration = 4096
    env = "wind"
    for h_size in h_sizes:
        for layer_num in layer_nums:
            print(f"layer_num: {layer_num}, h_size: {h_size}")
            MLP_model.train(layer_num, env, h_size, iteration, loss_method="3p")
            for id in ids:
                MLP_model.generate_sample(env, id, h_size, layer_num, loss_method="3p")
                vis_data.draw_data(index=id, env=env, sour="gen", gran=1, h_size=h_size, layer_num=layer_num, loss_method="3p")

def quality_sweep():
    h_sizes = [64, 128, 256]
    # h_sizes = [64]
    layer_nums = [2, 3]
    ids = [i for i in range(140, 200)]
    # ids = [i for i in range(140, 150)]
    envs = ["floor", "spring", "wind"]
    for env in envs:
        print(f"env: {env}")
        log_dir = f"log/summary/{env}.txt"
        with open(log_dir, "w") as f:
            f.write(f"env: {env}\n")
            f.close()
        for h_size in h_sizes:
            for layer_num in layer_nums:
                print(f"layer_num: {layer_num}, h_size: {h_size}")
                worst_abs_samples_tmp = [(0,-1) for i in range(4)]
                worst_ave_samples_tmp = [(0,-1) for i in range(4)]
                best_abs_samples_tmp = [(0,100) for i in range(4)]
                best_ave_samples_tmp = [(0,100) for i in range(4)]
                for id in ids:
                    # print(f"id: {id}")
                    pos, vel, com, len = MLP_model.quality_measure(env, id, h_size, layer_num=layer_num, loss_method="3p")
                    pos_abs = np.max(pos)
                    vel_abs = np.max(vel)
                    com_abs = np.max(com)
                    len_abs = np.max(len)

                    if pos_abs > worst_abs_samples_tmp[0][1]:
                        worst_abs_samples_tmp[0] = (id, pos_abs)
                    if vel_abs > worst_abs_samples_tmp[1][1]:
                        worst_abs_samples_tmp[1] = (id, vel_abs)
                    if com_abs > worst_abs_samples_tmp[2][1]:
                        worst_abs_samples_tmp[2] = (id, com_abs)
                    if len_abs > worst_abs_samples_tmp[3][1]:
                        worst_abs_samples_tmp[3] = (id, len_abs)
                    if pos_abs < best_abs_samples_tmp[0][1]:
                        best_abs_samples_tmp[0] = (id, pos_abs)
                    if vel_abs < best_abs_samples_tmp[1][1]:
                        best_abs_samples_tmp[1] = (id, vel_abs)
                    if com_abs < best_abs_samples_tmp[2][1]:
                        best_abs_samples_tmp[2] = (id, com_abs)
                    if len_abs < best_abs_samples_tmp[3][1]:
                        best_abs_samples_tmp[3] = (id, len_abs)

                    pos_ave = np.mean(pos)
                    vel_ave = np.mean(vel)
                    com_ave = np.mean(com)
                    len_ave = np.mean(len)

                    if pos_ave > worst_ave_samples_tmp[0][1]:
                        worst_ave_samples_tmp[0] = (id, pos_ave)
                    if vel_ave > worst_ave_samples_tmp[1][1]:
                        worst_ave_samples_tmp[1] = (id, vel_ave)
                    if com_ave > worst_ave_samples_tmp[2][1]:
                        worst_ave_samples_tmp[2] = (id, com_ave)
                    if len_ave > worst_ave_samples_tmp[3][1]:
                        worst_ave_samples_tmp[3] = (id, len_ave)
                    if pos_ave < best_ave_samples_tmp[0][1]:
                        best_ave_samples_tmp[0] = (id, pos_ave)
                    if vel_ave < best_ave_samples_tmp[1][1]:
                        best_ave_samples_tmp[1] = (id, vel_ave)
                    if com_ave < best_ave_samples_tmp[2][1]:
                        best_ave_samples_tmp[2] = (id, com_ave)
                    if len_ave < best_ave_samples_tmp[3][1]:
                        best_ave_samples_tmp[3] = (id, len_ave)

                with open(log_dir, "a") as f:
                    f.write(f"h_size: {h_size}, layer_num: {layer_num}\n")
                    f.write(f"worst_abs_pos_e: id {worst_abs_samples_tmp[0][0]} val {worst_abs_samples_tmp[0][1]}\n")
                    f.write(f"worst_abs_vel_e: id {worst_abs_samples_tmp[1][0]} val {worst_abs_samples_tmp[1][1]}\n")
                    f.write(f"worst_abs_com_e: id {worst_abs_samples_tmp[2][0]} val {worst_abs_samples_tmp[2][1]}\n")
                    f.write(f"worst_abs_len_e: id {worst_abs_samples_tmp[3][0]} val {worst_abs_samples_tmp[3][1]}\n")
                    f.write(f"worst_ave_pos_e: id {worst_ave_samples_tmp[0][0]} val {worst_ave_samples_tmp[0][1]}\n")
                    f.write(f"worst_ave_vel_e: id {worst_ave_samples_tmp[1][0]} val {worst_ave_samples_tmp[1][1]}\n")
                    f.write(f"worst_ave_com_e: id {worst_ave_samples_tmp[2][0]} val {worst_ave_samples_tmp[2][1]}\n")
                    f.write(f"worst_ave_len_e: id {worst_ave_samples_tmp[3][0]} val {worst_ave_samples_tmp[3][1]}\n")
                    f.write(f"best_abs_pos_e: id {best_abs_samples_tmp[0][0]} val {best_abs_samples_tmp[0][1]}\n")
                    f.write(f"best_abs_vel_e: id {best_abs_samples_tmp[1][0]} val {best_abs_samples_tmp[1][1]}\n")
                    f.write(f"best_abs_com_e: id {best_abs_samples_tmp[2][0]} val {best_abs_samples_tmp[2][1]}\n")
                    f.write(f"best_abs_len_e: id {best_abs_samples_tmp[3][0]} val {best_abs_samples_tmp[3][1]}\n")
                    f.write(f"best_ave_pos_e: id {best_ave_samples_tmp[0][0]} val {best_ave_samples_tmp[0][1]}\n")
                    f.write(f"best_ave_vel_e: id {best_ave_samples_tmp[1][0]} val {best_ave_samples_tmp[1][1]}\n")
                    f.write(f"best_ave_com_e: id {best_ave_samples_tmp[2][0]} val {best_ave_samples_tmp[2][1]}\n")
                    f.write(f"best_ave_len_e: id {best_ave_samples_tmp[3][0]} val {best_ave_samples_tmp[3][1]}\n")
                    f.close()

def quality_analysis():
    h_sizes = [64, 128, 256]
    layer_nums = [2, 3]
    ids = [i for i in range(140, 200)]
    envs = ["floor", "spring", "wind"]
    for env in envs:
        config_t = dict()
        for h_size in h_sizes:
            for layer_num in layer_nums:
                pos_e = []
                vel_e = []
                com_e = []
                len_e = []
                start_e = []
                end_e = []
                for id in ids:
                    pos_dir = f"log/detailed/pos_error_{env}_{id}_{h_size}_l{layer_num}_3p.npy"
                    vel_dir = f"log/detailed/vel_error_{env}_{id}_{h_size}_l{layer_num}_3p.npy"
                    com_dir = f"log/detailed/com_error_{env}_{id}_{h_size}_l{layer_num}_3p.npy"
                    len_dir = f"log/detailed/len_error_{env}_{id}_{h_size}_l{layer_num}_3p.npy"
                    pos_load = np.load(pos_dir)
                    vel_load = np.load(vel_dir)
                    com_load = np.load(com_dir)
                    len_load = np.load(len_dir)

                    pos_ave = np.mean(np.mean(pos_load, axis=1))
                    vel_ave = np.mean(np.mean(vel_load, axis=1))
                    com_ave =np.mean(com_load)
                    len_ave =np.mean(len_load)
                    start_ave = np.mean(pos_load[0])
                    end_ave = np.mean(pos_load[-1])

                    pos_e.append(pos_ave)
                    vel_e.append(vel_ave)
                    com_e.append(com_ave)
                    len_e.append(len_ave)
                    start_e.append(start_ave)
                    end_e.append(end_ave)

                pos_e = np.mean(np.array(pos_e))
                vel_e = np.mean(np.array(vel_e))
                com_e = np.mean(np.array(com_e))
                len_e = np.mean(np.array(len_e))
                start_e = np.mean(np.array(start_e))
                end_e = np.mean(np.array(end_e))

                config_t[(h_size, layer_num)] = (pos_e, vel_e, com_e, len_e, start_e, end_e)
        
        with open(f"log/summary/{env}_general.txt", "w") as f:
            f.write(f"env: {env}\n")
            for key in config_t.keys():
                f.write(f"h_size: {key[0]}, layer_num: {key[1]}\n")
                f.write(f"pos_e: {config_t[key][0]}\n")
                f.write(f"vel_e: {config_t[key][1]}\n")
                f.write(f"com_e: {config_t[key][2]}\n")
                f.write(f"len_e: {config_t[key][3]}\n")
                f.write(f"start_e: {config_t[key][4]}\n")
                f.write(f"end_e: {config_t[key][5]}\n")
            f.close()

        # plot histogram
        c = list(config_t.keys())
        pos_e = [config_t[config][0] for config in c]
        vel_e = [config_t[config][1] for config in c]
        com_e = [config_t[config][2] for config in c]
        len_e = [config_t[config][3] for config in c]
        start_e = [config_t[config][4] for config in c]
        end_e = [config_t[config][5] for config in c]
        x = np.arange(0, len(c)*6, 6)
        width = 0.7
        x1 = x
        x2 = x + 0.7
        x3 = x + 1.4
        x4 = x + 2.1
        x5 = x + 2.8
        x6 = x + 3.5

        colormap = plt.cm.get_cmap("coolwarm")
        colors = colormap(np.linspace(0, 1, 6))

        plt.bar(x1, pos_e, width, label="pos_e", color=colors[0])
        plt.bar(x2, vel_e, width, label="vel_e", color=colors[1])
        plt.bar(x3, com_e, width, label="com_e", color=colors[2])
        plt.bar(x4, len_e, width, label="len_e", color=colors[3])
        plt.bar(x5, start_e, width, label="start_e", color=colors[4])
        plt.bar(x6, end_e, width, label="end_e", color=colors[5])
        plt.xticks(x + 0.3, [f"{conf}" for conf in c], rotation=45)
        plt.xlabel("config")
        plt.ylabel("error")
        plt.title(f"{env} error")
        plt.legend()
        plt.savefig(f"report/fig/{env}_config_comp.png")
        plt.close()





if __name__ == "__main__":
    # sweep_model()
    # quality_sweep()
    # quality_analysis()
    MLP_model.generate_sample("floor", 180, 128, 3, loss_method="3p")
    MLP_model.generate_sample("floor", 182, 128, 3, loss_method="3p")
