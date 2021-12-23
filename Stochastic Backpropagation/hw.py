import numpy as np

def sig(n):
    return 1/(1+np.exp(-n))
def main():
    train = np.loadtxt("train.csv",skiprows=1,dtype=np.int,delimiter=',')
    train_y=train[:,0]
    train=np.delete(train,0,axis=1)
    train_size = int(np.square(np.size(train,1))/2)
    i1 = int(np.size(train,1)/2)
    j1 = np.size(train,1)
    w1 = np.random.rand(i1,j1)
    w1 = w1/10000
    b1 = np.random.rand(i1,1)
    b1 = b1/10000
    i2 = 4
    j2 = i1
    w2 = np.random.rand(i2,j2)/100
    b2 = np.random.rand(i2,1)/100
    a1 = sig(np.dot(w1,train[0].reshape(784,1))+b1)
    a2 = sig(np.dot(w2,a1)+b2)
    
main()
