import numpy as np
a = [10,20,30,60]
n1 = np.array(a)
print(n1)
print(type(n1))


a = [10,20,30,60]
b = [1,2,3,4,]
n2 = np.array(b)
print('dimentiom of array n1',n1.ndim)
print('dimentiom of array n2',n2.ndim)

print('size of array n2',n2.size)
print('shape of array n2',n2.shape)
print('type of array n2',n2.dtype)


n3=np.arange(1,24,2)
print(n3)

n3=np.arange(24)
print(n3)

#ones()

n4=np.ones(4)
print(n4)
print(n4.dtype)




n6=np.zeros(4)
print(n6)
print(n6.dtype)

n8=np.linspace(10,20)
print(n8)
print(n8.dtype)


n9=np.linspace(10,20,num=5)
print(n9)
print(n9.dtype)

n10=np.linspace(10,20,num=5,endpoint=False)
print(n10)

n10=np.arange(1,13)
print(n10)
print('dimention is',n10.ndim)
n11=np.reshape(n10,(3,4))
print(n11)
print('dimention is',n11.ndim)


n12=np.ravel(n11)
print('dimention of n12', n12.ndim)  
print(n12)
         