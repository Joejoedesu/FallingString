import MLP_model
import vis_data

def sweep_model():
    # h_sizes = [16, 32, 64, 128, 256]
    h_sizes = [64, 128, 256]
    layer_nums = [1, 2, 3]
    ids = [180, 182]
    env = "floor"
    for h_size in h_sizes:
        for layer_num in layer_nums:
            print(f"layer_num: {layer_num}, h_size: {h_size}")
            MLP_model.train(layer_num, h_size)
            for id in ids:
                MLP_model.generate_sample(env, id, h_size, layer_num)
                vis_data.draw_data(index=id, sour="gen", gran=1, h_size=h_size, layer_num=layer_num)

if __name__ == "__main__":
    sweep_model()