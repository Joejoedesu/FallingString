import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    use_cuda = True


data_dir = "data/"

class MLP_1L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_1L, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x

class MLP_2L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_2L, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
    
class MLP_3L(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_3L, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

def create_train_data_3p(env, train_range, batch=256, size=100, purpose="train"):
    start = train_range[0]
    end = train_range[1] - 1
    train_set_input = []
    train_set_output = []
    for i in range(size):
        print(f"creating {purpose} data {i}/{size}")
        S_input = []
        S_output = []
        for j in range(batch):
            id = random.randint(start, end) #include
            #load from data directory
            sim_dir = f"sim/location_{env}_{id}"
            sta_dir = f"start/spline_{env}_{id}"
            sim = np.load(data_dir + sim_dir + ".npy")
            sta = np.load(data_dir + sta_dir + ".npy")
            L = len(sim[0])
            T = len(sim)
            # print(L, T)
            l = random.uniform(0, 1)
            t = random.uniform(0, 1)
            l_index = int((L - 1) * l)
            t_index = int((T - 1) * t)
            if l_index == 0:
                l_index = 1
            if l_index == L - 1:
                l_index = L - 2
            l = l_index / (L - 1)
            t = t_index / (T - 1)

            loc = sim[t_index][l_index]
            loc_l = sim[t_index][l_index - 1]
            loc_r = sim[t_index][l_index + 1]
            l_l = (l_index - 1) / (L - 1)
            l_r = (l_index + 1) / (L - 1)
            d1 = np.linalg.norm(loc_r - loc)
            d2 = np.linalg.norm(loc_l - loc)
            angle = np.dot(loc_r - loc, loc_l - loc) / (d1 * d2)
            loc = np.append(loc, [d1, d2, angle])
            S_output.append(loc)
            sta = sta.flatten()
            extra = [l, t, l_l, l_r]
            sta = np.append(sta, extra)
            S_input.append(sta)
        S_input = np.array(S_input)
        S_output = np.array(S_output)
        train_set_input.append(S_input)
        train_set_output.append(S_output)
    np.save(f"data/{purpose}_input_{env}_{start}_{end}_3p", train_set_input)
    np.save(f"data/{purpose}_output_{env}_{start}_{end}_3p", train_set_output)
    return train_set_input, train_set_output

def create_train_data(env, train_range, batch=256, size=100, purpose="train"):
    start = train_range[0]
    end = train_range[1] - 1
    train_set_input = []
    train_set_output = []
    for i in range(size):
        print(f"creating {purpose} data {i}/{size}")
        S_input = []
        S_output = []
        for j in range(batch):
            id = random.randint(start, end) #include
            #load from data directory
            sim_dir = f"sim/location_{env}_{id}"
            sta_dir = f"start/spline_{env}_{id}"
            sim = np.load(data_dir + sim_dir + ".npy")
            sta = np.load(data_dir + sta_dir + ".npy")
            L = len(sim[0])
            T = len(sim)
            # print(L, T)
            l = random.uniform(0, 1)
            t = random.uniform(0, 1)
            l_index = int((L - 1) * l)
            t_index = int((T - 1) * t)
            loc = sim[t_index][l_index]
            S_output.append(loc)
            sta = sta.flatten()
            sta = np.append(sta, [l, t])
            S_input.append(sta)
        S_input = np.array(S_input)
        S_output = np.array(S_output)
        train_set_input.append(S_input)
        train_set_output.append(S_output)
    np.save(f"data/{purpose}_input_{env}_{start}_{end}", train_set_input)
    np.save(f"data/{purpose}_output_{env}_{start}_{end}", train_set_output)
    return train_set_input, train_set_output
            
def test_error(model, test_input, test_output, loss_fn, loss_method="1p"):
    l = -1
    if loss_method == "1p":
        for i in range(1):
            input = torch.tensor(test_input[i], dtype=torch.float32)
            output = torch.tensor(test_output[i], dtype=torch.float32)

            if use_cuda:
                input = input.to(device)
                output = output.to(device)

            pred = model(input)
            loss = loss_fn(pred, output)
            print(f"train loss: {loss.item()}")
            # print(f"train output: {output[0]}")
            # print(f"train prediction: {pred[0]}")
            l = loss.item()
    elif loss_method == "3p":
        for i in range(1):
            input = torch.tensor(test_input[i], dtype=torch.float32)
            output = torch.tensor(test_output[i], dtype=torch.float32)
            indice_start = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            
            start = torch.index_select(input, 1, indice_start)
            l = torch.select(input, 1, 15).reshape(-1, 1)
            t = torch.select(input, 1, 16).reshape(-1, 1)
            l_l = torch.select(input, 1, 17).reshape(-1, 1)
            l_r = torch.select(input, 1, 18).reshape(-1, 1)

            # print(start.shape, l.shape, t.shape, l_l.shape, l_r.shape)

            input = torch.cat((start, l, t), 1)
            input_l = torch.cat((start, l_l, t), 1)
            input_r = torch.cat((start, l_r, t), 1)

            if use_cuda:
                input = input.to(device)
                input_l = input_l.to(device)
                input_r = input_r.to(device)
                output = output.to(device)
            
            pred = model(input)
            pred_l = model(input_l)
            pred_r = model(input_r)

            loss, loc_e, hk_e, angle_e = loss_fn(pred, pred_l, pred_r, output)
            print(f"train loss: {loss.item()}")
            l = loss.item()
    return l

def loss_3p(pred, pred_l, pred_r, output):
    select_i = torch.tensor([0, 1, 2])
    if use_cuda:
        select_i = select_i.to(device)
    loc = torch.index_select(output, 1, select_i)
    d1 = torch.select(output, 1, 3).reshape(-1, 1)
    d2 = torch.select(output, 1, 4).reshape(-1, 1)
    angle = torch.select(output, 1, 5).reshape(-1, 1)

    loc_e = torch.norm(torch.sub(pred, loc), dim=1)

    D1 = torch.sub(pred_r, pred)
    D2 = torch.sub(pred_l, pred)
    D1_l = torch.norm(D1, dim=1).reshape(-1, 1)
    D2_l = torch.norm(D2, dim=1).reshape(-1, 1)
    hk_e = torch.abs(torch.sub(D1_l, d1)) + torch.abs(torch.sub(D2_l, d2))

    dot_12 = torch.sum(torch.mul(D1, D2), dim=1).reshape(-1, 1)
    angle_e = torch.abs(torch.sub(dot_12, angle))

    # loc_e = torch.mean(loc_e)
    # hk_e = torch.mean(hk_e)
    # angle_e = torch.mean(angle_e)

    # total_loss = loc_e + hk_e + angle_e
    total_loss = loc_e + angle_e
    total_loss = torch.mean(total_loss)
    return total_loss, loc_e, hk_e, angle_e

def error_measure(output, ref):
    assert len(output) == len(ref)
    pos_e = 0
    hk_e = 0
    bd_e = 0

    for i in range(len(output)):
        pos_e += np.linalg.norm(output[i] - ref[i])

    dist_ref_1 = np.linalg.norm(ref[2] - ref[0])
    dist_ref_2 = np.linalg.norm(ref[0] - ref[1])

    dist_out_1 = np.linalg.norm(output[2] - output[0])
    dist_out_2 = np.linalg.norm(output[0] - output[1])

    hk_e = abs(dist_out_1 - dist_ref_1) + abs(dist_out_2 - dist_ref_2)

    n_ref_1 = (ref[2] - ref[0]) / dist_ref_1
    n_ref_2 = (ref[0] - ref[1]) / dist_ref_2

    n_out_1 = (output[2] - output[0]) / dist_out_1
    n_out_2 = (output[0] - output[1]) / dist_out_2

    bd_e = abs(np.dot(n_ref_1, n_ref_2) - np.dot(n_out_1, n_out_2))

    return pos_e, hk_e, bd_e

def quality_measure(env="floor", id=182, h_size=32, layer_num=1, gran = 80, loss_method="1p"):
    sim_dir = f"sim/location_{env}_{id}"
    sim = np.load(data_dir + sim_dir + ".npy")
    sta_dir = f"start/spline_{env}_{id}"
    sta = np.load(data_dir + sta_dir + ".npy")
    L = len(sim[0])
    T = len(sim)
    sp_gran = 32
    T_step = int(T/gran)
    L_step = int(L/sp_gran)

    if loss_method == "1p":
        model = torch.load(f"model/{env}/model_{h_size}_l{layer_num}.pth")
    elif loss_method == "3p":
        model = torch.load(f"model/{env}/model_{h_size}_l{layer_num}_{loss_method}.pth")
    model.eval()
    if use_cuda:
        model.to(device)

    pos_e = 0
    hk_e = 0
    bd_e = 0
    i_c = 0
    for i in range(0, T, T_step):
        t = i * 1/T
        j_c = 0
        for j in range(1, L-1, L_step):
            l = j * 1/L
            l_l = (j-1) * 1/L
            l_r = (j+1) * 1/L
            sim_loc =sim[i][j]
            sim_loc_l = sim[i][j-1]
            sim_loc_r = sim[i][j+1]

            gen_input = np.append(sta, [l, t])
            gen_l_input = np.append(sta, [l_l, t])
            gen_r_input = np.append(sta, [l_r, t])
            gen_input = torch.tensor(gen_input, dtype=torch.float32)
            gen_l_input = torch.tensor(gen_l_input, dtype=torch.float32)
            gen_r_input = torch.tensor(gen_r_input, dtype=torch.float32)

            if use_cuda:
                gen_input = gen_input.to(device)
                gen_l_input = gen_l_input.to(device)
                gen_r_input = gen_r_input.to(device)
            
            gen_loc = model(gen_input)
            gen_loc_l = model(gen_l_input)
            gen_loc_r = model(gen_r_input)

            gen_loc = gen_loc.cpu().detach().numpy()
            gen_loc_l = gen_loc_l.cpu().detach().numpy()
            gen_loc_r = gen_loc_r.cpu().detach().numpy()

            pos_e_, hk_e_, bd_e_ = error_measure([gen_loc, gen_loc_l, gen_loc_r], [sim_loc, sim_loc_l, sim_loc_r])
            pos_e += pos_e_
            hk_e += hk_e_
            bd_e += bd_e_

            j_c += 1
        i_c += 1

    pos_e = pos_e / ((T//T_step) * ((L-2)//L_step))
    hk_e = hk_e / ((T//T_step) * ((L-2)//L_step))
    bd_e = bd_e / ((T//T_step) * ((L-2)//L_step))

    print(f"position error: {pos_e}")
    print(f"hook error: {hk_e}")
    print(f"bend error: {bd_e}")
    return pos_e, hk_e, bd_e

def train(layer_num, env='floor', h_size=64, iteration=100, loss_method="1p"):
    
    i_size = 5*3 + 2 # 5 points with 3 coordinates + time + interpolated points
    o_size = 3 # single coordinate

    if layer_num == 1:
        model = MLP_1L(i_size, h_size, o_size)
    elif layer_num == 2:
        model = MLP_2L(i_size, h_size, o_size)
    elif layer_num == 3:
        model = MLP_3L(i_size, h_size, o_size)
    else:
        print("invalid layer number")
        return

    if use_cuda:
        model.to(device)

    # batch = 256
    # iteration = 4096
    # epoch = 500
    epoch = 200

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # loss_fn = nn.MSELoss()
    if loss_method == "1p":
        loss_fn = nn.L1Loss()
    elif loss_method == "3p":
        loss_fn = loss_3p
    # train_input, train_output = create_train_data(env, [0, 140], batch, iteration, "train")
    # test_input, test_output = create_train_data(env, [140, 200], batch, 1, "test")
    
    if loss_method == "1p":
        train_input = np.load(f"data/train_input_{env}_0_139.npy")
        train_output = np.load(f"data/train_output_{env}_0_139.npy")
        test_input = np.load(f"data/test_input_{env}_140_199.npy")
        test_output = np.load(f"data/test_output_{env}_140_199.npy")
    elif loss_method == "3p":
        train_input = np.load(f"data/train_input_{env}_0_139_3p.npy")
        train_output = np.load(f"data/train_output_{env}_0_139_3p.npy")
        test_input = np.load(f"data/test_input_{env}_140_199_3p.npy")
        test_output = np.load(f"data/test_output_{env}_140_199_3p.npy")

    error_list = []
    test_error_list = []
    ave_test_error = []
    over_f_c = 0

    #training
    for e in range(epoch):
        sum_loss = 0
        for i in range(iteration):
            input = torch.tensor(train_input[i], dtype=torch.float32)
            output = torch.tensor(train_output[i], dtype=torch.float32)


            if loss_method == "3p":
                # print(input.shape, output.shape)
                # print(input)
                indice_start = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
                
                start = torch.index_select(input, 1, indice_start)
                l = torch.select(input, 1, 15).reshape(-1, 1)
                t = torch.select(input, 1, 16).reshape(-1, 1)
                l_l = torch.select(input, 1, 17).reshape(-1, 1)
                l_r = torch.select(input, 1, 18).reshape(-1, 1)

                # print(start.shape, l.shape, t.shape, l_l.shape, l_r.shape)

                input = torch.cat((start, l, t), 1)
                input_l = torch.cat((start, l_l, t), 1)
                input_r = torch.cat((start, l_r, t), 1)

                # print(input)
                # print(input_l)
                # print(input_r)
                # print(output.shape)

                if use_cuda:
                    input = input.to(device)
                    input_l = input_l.to(device)
                    input_r = input_r.to(device)
                    output = output.to(device)
                
                optimizer.zero_grad()
                pred = model(input)
                pred_l = model(input_l)
                pred_r = model(input_r)

                loss, loc_e, hk_e, angle_e = loss_fn(pred, pred_l, pred_r, output)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            
            else:
                if use_cuda:
                    input = input.to(device)
                    output = output.to(device)

                optimizer.zero_grad()
                pred = model(input)
                loss = loss_fn(pred, output)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
        train_e = sum_loss/iteration
        print(f"epoch {e} loss: {train_e}")
        error_list.append(train_e)
        test_e = test_error(model, test_input, test_output, loss_fn, loss_method)
        test_error_list.append(test_e)
        if e > 11:
            if sum(test_error_list[-10:]) > 1.05 * sum(test_error_list[-11:-1]):
                break
        if test_e > 1.2 * train_e:
            over_f_c += 1
        else:
            over_f_c = 0
        if over_f_c > 5:
            break
    
    #plot error
    plt.plot(error_list)
    plt.plot(test_error_list)
    plt.legend(["train", "test"])
    plt.close()
    # plt.show()
    np.save(f"log/error_{env}_{h_size}_l{layer_num}_{loss_method}", np.array(error_list))
    np.save(f"log/test_error_{env}_{h_size}_l{layer_num}_{loss_method}", np.array(test_error_list))

    #testing
    print(f"final test error: {test_error(model, test_input, test_output, loss_fn, loss_method)}")
    
    # save model
    torch.save(model, f"model/{env}/model_{h_size}_l{layer_num}_{loss_method}.pth")

    
def generate_sample(env="floor", id=182, h_size=32, layer_num=1, loss_method="1p"):
    # model = torch.load(f"model/{env}/model_{h_size}.pth")
    if loss_method == "1p":
        model = torch.load(f"model/{env}/model_{h_size}_l{layer_num}.pth")
    elif loss_method == "3p":
        model = torch.load(f"model/{env}/model_{h_size}_l{layer_num}_{loss_method}.pth")
    model.eval()
    if use_cuda:
        model.to(device)
    sta_dir = f"start/spline_{env}_{id}"
    sta = np.load(data_dir + sta_dir + ".npy")
    sta = sta.flatten()
    sta = np.append(sta, [0, 0])
    # gran = 100
    # sim = np.load(data_dir + f"sim/location_{env}_{id}.npy")
    # L = len(sim[0])
    gran = 40
    t_step = 1/gran
    L = 15
    l_step = 1/L
    gen = []
    for i in range(gran+1):
        t_pos = []
        for j in range(L+1):
            t = i * t_step
            l = j * l_step
            sta[-2] = l
            sta[-1] = t
            input = torch.tensor(sta, dtype=torch.float32)
            if use_cuda:
                input = input.to(device)
            pred = model(input)
            t_pos.append(pred.cpu().detach().numpy())
        gen.append(t_pos)
    np.save(f"data/gen/location_{env}_{id}_{h_size}_l{layer_num}_{loss_method}", np.array(gen))
    
# train()
# generate_sample()
# create_train_data_3p("spring", [0, 140], 256, 128, "train")
# create_train_data_3p("spring", [140, 200], 256, 1, "test")
# train(layer_num=2, h_size=64, iteration=100, loss_method="3p")
# train(2, 64, "1p")