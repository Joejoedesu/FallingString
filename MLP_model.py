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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
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

def create_train_data(env, train_range, batch=256, size=100, purpose="train"):
    start = train_range[0]
    end = train_range[1] - 1
    train_set_input = []
    train_set_output = []
    for i in range(size):
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
        train_set_input.append(S_input)
        train_set_output.append(S_output)
    np.save(f"data/{purpose}_input_{env}_{start}_{end}", train_set_input)
    np.save(f"data/{purpose}_output_{env}_{start}_{end}", train_set_output)
    return train_set_input, train_set_output
            
def test_error(model, test_input, test_output, loss_fn):
    l = -1
    for i in range(1):
        input = torch.tensor(test_input[i], dtype=torch.float32)
        output = torch.tensor(test_output[i], dtype=torch.float32)

        if use_cuda:
            input = input.to(device)
            output = output.to(device)

        pred = model(input)
        loss = loss_fn(pred, output)
        print(f"train loss: {loss.item()}")
        print(f"train output: {output[0]}")
        print(f"train prediction: {pred[0]}")
        l = loss.item()
    return l

def train():
    
    i_size = 5*3 + 2 # 5 points with 3 coordinates + time + interpolated points
    h_size = 128
    o_size = 3 # single coordinate
    model = MLP(i_size, h_size, o_size)

    if use_cuda:
        model.to(device)

    batch = 64
    iteration = 256
    epoch = 1000
    env = "floor"

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()

    train_input, train_output = create_train_data(env, [0, 140], batch, iteration, "train")
    test_input, test_output = create_train_data(env, [140, 200], batch, 1, "test")

    error_list = []
    test_error_list = []
    over_fit_cnt = 0

    #training
    for e in range(epoch):
        sum_loss = 0
        for i in range(iteration):
            input = torch.tensor(train_input[i], dtype=torch.float32)
            output = torch.tensor(train_output[i], dtype=torch.float32)

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
        test_e = test_error(model, test_input, test_output, loss_fn)
        test_error_list.append(test_e)
        if test_e > train_e * 1.3:
            over_fit_cnt += 1
        else:
            over_fit_cnt = 0
        if over_fit_cnt > 20:
            break
    
    #plot error
    plt.plot(error_list)
    plt.show()
    np.save(f"log/error_{env}_{h_size}", np.array(error_list))

    #testing
    print(f"final test error: {test_error(model, test_input, test_output, loss_fn)}")
    
    # save model
    torch.save(model, f"model/{env}/model_{h_size}.pth")

    
def generate_sample():
    env = "floor"
    h_size = 128
    model = torch.load(f"model/{env}/model_{h_size}.pth")
    model.eval()
    if use_cuda:
        model.to(device)
    id = 182
    sta_dir = f"start/spline_{env}_{id}"
    sta = np.load(data_dir + sta_dir + ".npy")
    sta = sta.flatten()
    sta = np.append(sta, [0, 0])
    # gran = 100
    # sim = np.load(data_dir + f"sim/location_{env}_{id}.npy")
    # L = len(sim[0])
    gran = 80
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
    np.save(f"data/gen/location_{env}_{id}", np.array(gen))
    
# train()
generate_sample()
