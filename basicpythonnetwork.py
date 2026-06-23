import numpy as np

# sigmoid func
def nonlin(x, deriv = False):
    if (deriv == True):
        return x * (1 - x)
    return 1/(1+np.exp(-x))

# input data
X = np.array([  [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1]  ])

# output data
Y = np.array([[0, 0, 1, 1]]).T

# random number seeding to make calculation deterministic
np.random.seed(1)

# init weights randomly with mean = 0
syn0 = 2*np.random.random((3, 1)) - 1

for elem in range(10000):

    # forward propogation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # error calcs
    l1error = Y - l1

    # multiply how much we missed by the slope of the sigmoid at the values in l1
    l1delta = l1error * nonlin(l1, True)

    # update weightings
    syn0 += np.dot(l0.T, l1delta)

print("Output after training:")
print(l1)