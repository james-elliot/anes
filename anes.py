import torch
import math
dtype = torch.float
#device_name="cpu"
device_name="cuda:0"
device = torch.device(device_name)

f = open("source_learn.txt", "r")
l_in=[[float(x) for x in line.split()] for line in f]
f.close()
print("finished loading source learn")
f = open("obj_learn.txt", "r")
l_out=[[float(x) for x in line.split()] for line in f]
f.close()
print("finished loading obj learn")
x = torch.tensor(l_in,device=device,dtype=dtype)
y = torch.tensor(l_out,device=device,dtype=dtype)

f = open("source_test.txt", "r")
l_in_test=[[float(x) for x in line.split()] for line in f]
f.close()
print("finished loading source test")
f = open("obj_test.txt", "r")
l_out_test=[[float(x) for x in line.split()] for line in f]
f.close()
print("finished loading obj test")
x_test = torch.tensor(l_in_test,device=device,dtype=dtype)
y_test = torch.tensor(l_out_test,device=device,dtype=dtype)

D_in, H1, H2,D_out = x[0].size()[0], 20, 20,y[0].size()[0]
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.Sigmoid(),
    torch.nn.Linear( H1,H2),
    torch.nn.Sigmoid(),
#    torch.nn.ReLU(),
    torch.nn.Linear(H2, D_out)
)
if device_name=="cuda:0": model.cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for t in range(200000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 100 == 99:
        print(t, loss.item())
        y_pred_test = model(x_test)
        loss2 = loss_fn(y_pred_test, y_test)
        v=loss2.item()
        print(t, " MSE:",v," RMSE:",math.sqrt(v))
        if loss.item()<1e-6:break
y_pred=model(x)
print(x)
print(y)
print(y_pred)

