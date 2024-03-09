import MLP_model
import vis_data

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
    h_sizes = [16, 64, 128, 256]
    layer_nums = [1, 2, 3]
    ids = [180, 182, 193, 199]
    env = "floor"
    for h_size in h_sizes:
        for layer_num in layer_nums:
            print(f"layer_num: {layer_num}, h_size: {h_size}")
            for id in ids:
                print(f"id: {id}")
                MLP_model.quality_measure(env, id, h_size, layer_num)

if __name__ == "__main__":
    sweep_model()
    # quality_sweep()