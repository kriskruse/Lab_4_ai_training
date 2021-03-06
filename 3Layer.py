import torch
import numpy as np
# import matplotlib.pyplot as plt
# import csv


# print("""This is the bruteforcing argument test by
# Marcus
# Kris
# Casper
#
# Starting script
# """)

def weird_fun(x):
    return np.sin(1 / x)


np.random.seed(1)

N = 50  # Number of observations
s = 0.02  # Noise standard deviation

x_train = np.sort(np.random.rand(N) * 2 - 1)
y_train = weird_fun(x_train) + s * np.random.randn(N)

x_test = np.sort(np.random.rand(N) * 2 - 1)
y_test = weird_fun(x_test) + s * np.random.randn(N)

device = torch.device('cpu')

x = torch.tensor(np.expand_dims(x_train, 1), dtype=torch.float32, device=device)
y = torch.tensor(np.expand_dims(y_train, 1), dtype=torch.float32, device=device)

loss_fn = torch.nn.MSELoss(reduction='sum')

# print("Random training data set generated")

sH = 1 # start at H = x

H = 1000  # Max Hidden dimension
L = 3  # Lag
T = 10000  # Number of iterations


def train_network(T, model):
    model.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    Loss = np.zeros(T)
    for t in range(T):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        Loss[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_error = loss_fn(y_pred, y).item()
    x_t = torch.tensor(np.expand_dims(x_test, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(np.expand_dims(y_test, 1), dtype=torch.float32, device=device)
    y_t_pred = model(x_t)
    x_all = np.linspace(-1, 1, 1000)
    x_all_t = torch.tensor(np.expand_dims(x_all, 1), dtype=torch.float32, device=device)
    y_all_t = model(x_all_t)
    test_error = loss_fn(y_t_pred, y_t).item()

    return train_error, test_error


def define_model(L, H):
    if L == 1:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 2:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 3:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 4:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 5:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 6:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 7:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 8:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 9:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )
    elif L == 10:
        model = torch.nn.Sequential(
            torch.nn.Linear(1, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 1)
        )

    return model


# with open('DataSettings.csv', 'a', newline="") as csvfile:
#   writer = csv.writer(csvfile, delimiter=',')

print("H,L,T,A[0],A[1]")
# Layers

# print(f"Starting learning with Layer {l}")

# Iterations


# Hidden dimension


for h in range(sH, H):
    model = define_model(L, h)

    a = train_network(T, model)
    print(f"{h},{L},{T},{a[0]},{a[1]}")
    # writer.writerow([h, l, t, a[0], a[1]])
#     print(f"Done with {t} Iterations")
# print(f"Done with Layer {l}")

# network(1, 100)
