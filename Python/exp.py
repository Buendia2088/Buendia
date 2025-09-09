import numpy as np
import math
### 计算油滴带电量函数
def Qi(ti,U):
    pai=3.1415926
    g=9.8
    d=0.005
    l=0.002
    rou=981
    n=1.83*10e-5
    b=8.23*10e-3
    p=0.9633*10e5
    k1=18*pai*d/np.sqrt(2*rou*g)
    k2=math.pow(n*l,3/2)
    k3=b/p
    f=np.sqrt(2*rou*g*ti/(9*n*l))
    
    qi=k1*k2*math.pow(1+k3*f,-3/2)*(1/U)*math.pow(ti,-3/2)
    return qi

ti = eval(input("t = "))
U = eval(input("U = "))

print("Qi is", Qi(ti, U))

''' 
###倒过来验证法（举个例子）
t1=19.5
U=213
q1=Qi(t1,U)
e=1.6*10e-19
print(q1/e) #这里算出来n为14.771775148635285，取整，n=15。
e' = q1/15
'''
