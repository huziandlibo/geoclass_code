import numpy as np
from lib.Pnm import Pnm,CalPnm
from lib.Openfile import Get_CnmSnm
from math import sqrt,sin,cos,pi,acos,atan2
"""static params"""
#GRS80正常椭球参数
a=6378137
GM=3986005*1e8
J2=108263*1e-8
w=7292115*1e-11
b=6356752.3141
E=521854.0097
c=6399593.6259 
e2=0.00669438002290
e2_=0.00673949677548
f=0.00335281068118
U0 = 62636860.850
J4 =-0.00000237091222
J6 =0.00000000608347
J8 =-0.00000000001427
m = 0.00344978600308
ra =9.7803267715
rb =9.8321863685
"""endline"""

def GetC2n(J2n:float,n:int)->float:
    return -J2n/sqrt(4*n+1)

Cnm,Snm=Get_CnmSnm()
shape=Cnm.shape
Cnm_T,Snm_T=Cnm,Snm
Cnm_T[2,0]-=GetC2n(J2,1)
Cnm_T[4,0]-=GetC2n(J4,2)
Cnm_T[6,0]-=GetC2n(J6,3)
Cnm_T[8,0]-=GetC2n(J8,4)


def Calculate(B:float,L:float,H:float,N:int=360)->tuple[float,float,float,float]:
    B,L=Sec_to_Rad(B),Sec_to_Rad(L)
    gammaQ=Cal_gamma(B)
    X,Y,Z=BLHtoXYZ(B,L,H)
    r,theta,lamda=XYZtortl(X,Y,Z)
    P0=Pnm(N,N,theta/pi*180)
    P=CalPnm(P0,"标准向前列递推")
    cost = np.cos(np.arange(N+1) * lamda)
    sint = np.sin(np.arange(N+1) * lamda)
    res_T=0;res_delg=0;res_vardelg=0
    tmp_T = np.zeros(N-1)
    for n in range(2, N+1):
        tmp_T[n-2]= np.sum((Cnm_T[n, :n+1] * cost[:n+1] + Snm_T[n, :n+1] * sint[:n+1]) * P[n, :n+1], axis=0)
    k1 = (a/r)**np.arange(2, N+1)
    res_T = np.sum(k1 * tmp_T)
    res_delg = np.sum(k1 * (np.arange(2, N+1) - 1) * tmp_T)
    res_vardelg = np.sum(k1 * (np.arange(2, N+1) + 1) * tmp_T)
    T=(GM)/r*res_T
    k2=(GM)/r**2
    del_g=k2*res_delg
    vardel_g=k2*res_vardelg
    return T,T/gammaQ,del_g*100000,vardel_g*100000


def Cal_gamma(B:float)->float:
    return (a*ra*cos(B)**2+b*rb*sin(B)**2)/sqrt(a**2*cos(B)**2+b**2*sin(B)**2)

def Cal_V(B:float,L:float,H:float,N:int=360)->float:
    B,L=Sec_to_Rad(B),Sec_to_Rad(L)
    X,Y,Z=BLHtoXYZ(B,L,H)
    r,theta,lamda=XYZtortl(X,Y,Z)
    P0=Pnm(N,N,theta/pi*180)
    P=CalPnm(P0,"标准向前列递推")
    cost = np.cos(np.arange(N+1) * lamda)
    sint = np.sin(np.arange(N+1) * lamda)
    res=0
    tmp_T = np.zeros(N+1)
    for n in range(0, N+1):
        tmp_T[n]= np.sum((Cnm[n, :n+1] * cost[:n+1] + Snm[n, :n+1] * sint[:n+1]) * P[n, :n+1], axis=0)
    k1 = (a/r)**np.arange(0, N+1)
    res= np.sum(k1 * tmp_T)
    return (GM/r)*res


def Cal_U(B:float,L:float,H:float,N:int=360)->float:
    B,L=Sec_to_Rad(B),Sec_to_Rad(L)
    X,Y,Z=BLHtoXYZ(B,L,H)
    r,theta,_=XYZtortl(X,Y,Z)
    P0=Pnm(N,N,theta/pi*180)
    P=CalPnm(P0,"标准向前列递推")
    res=1
    res+=(a/r)**(2)*GetC2n(J2,1)*P[2,0]
    res+=(a/r)**(4)*GetC2n(J2,2)*P[4,0]
    res+=(a/r)**(6)*GetC2n(J2,3)*P[6,0]
    res+=(a/r)**(8)*GetC2n(J2,4)*P[8,0]
    return res*GM/r

def BLHtoXYZ(B:float,L:float,H:float)->tuple[float,float,float]:
    N=a/sqrt(1-e2*sin(B)**2)
    X=(N+H)*cos(B)*cos(L)
    Y=(N+H)*cos(B)*sin(L)
    Z=(N*(1-e2) + H)*sin(B)
    return X,Y,Z

def XYZtortl(X:float,Y:float,Z:float)->tuple[float,float,float]:
    r=sqrt(X**2+Y**2+Z**2)
    theta=atan2(sqrt(X**2+Y**2),Z)
    lamda=atan2(Y,X)
    if lamda<0:
        lamda+=2*pi
    return r,theta,lamda

def Sec_to_Rad(sec:float)->float:
    return sec/180*pi
