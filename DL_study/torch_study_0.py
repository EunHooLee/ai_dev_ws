import torch as th

s1 = th.tensor([1.])
s2 = th.tensor([2.])

print(th.add(s1,s2))

v1 = th.tensor([1., 2., 3.])
v2 = th.tensor([3., 2., 1.])

print(th.mul(v1,v2), th.dot(v1,v2))

m1 = th.tensor([[1, 2],[3, 4]])
m2 = th.tensor([[3, 4],[4, 2]])

print(th.matmul(m1, m2))

t1 = th.tensor([[[1, 2], [3, 4]],[[5, 6],[7, 8]]])
t2 = th.tensor([[[9, 10], [11, 12]],[[13, 14],[15, 16]]])

print(th.matmul(t1,t2).size())

print(th.mm(m1,m2))
print(m1.mm(m2))

print(m1)
print(m1.view(2*2,-1))
print(m1.view(-1,2*2))