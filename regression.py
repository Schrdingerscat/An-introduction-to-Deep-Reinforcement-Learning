#torch is the module pytorch
import torch
#F means import excitation function
import torch.nn.functional as F
#plt Data Visualization
import matplotlib.pyplot as plt

# torch.manual_seed(1) # reproducible

#torch.squeeze() This function mainly compresses the dimension of the data
#torch.unsqueeze() This function is mainly to expand the data dimension
#generate data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) # noisy y data (tensor), shape=(100, 1)


# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module): #Inheritance
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #The constructor of the parent class is called
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output) # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x)) # activation function for hidden layer
        x = self.predict(x) # linear output
        return x

#The underlying __call__ defines: if the class name is called as a function, the forward function of the class is automatically called
net = Net(n_feature=1, n_hidden=10, n_output=1) # define the network
print(net) # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2) #SGD: stochastic gradient descent
loss_func = torch.nn.MSELoss() # this is for regression mean squared loss mean squared loss

plt.ion() # something about plotting

for t in range(200):
    #equivalent to prediction = net.predict(x)
    prediction = net(x) # input x and predict based on x
    loss = loss_func(prediction, y) # must be (1. nn output, 2. target)
    # clear gradient vector
    optimizer.zero_grad() # clear gradients for next train
    loss.backward() # backpropagation, compute gradients backpropagation
    optimizer.step() # apply gradients update parameters

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()