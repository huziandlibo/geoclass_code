import numpy as np
from math import sqrt,sin,cos,pi


# 0<=l<=180,0<=m<=l,theat=30

class Pnm:
    
    def __init__(self,n: int,m: int,theta) -> None:
        self.__n=n
        self.__m=m
        self.__theta=theta/180*pi
        self.__P00=1
        self.__P10=sqrt(3)*cos(self.__theta)
        self.__P11=sqrt(3)*sin(self.__theta)
        self.__Nnm=Cal_Nnm(self.n)
    @property
    def n(self)-> int:
        return self.__n
    @property
    def m(self)->int:
        return self.__m
    @property
    def theta(self):
        return self.__theta
    @property
    def P00(self):
        return self.__P00
    @property
    def P10(self):
        return self.__P10
    @property
    def P11(self):
        return self.__P11
    def returnPmn(self) ->None:
        print(self.__theta,self.__P00,self.__P10,self.__P11)    


#region 跨阶数递推法

#跨阶数递推公式各系数
def A_nm(n,m):
    a_nm_up=(2*n-1)*(2*n+1)
    a_nm_down=(n-m)*(n+m)
    return (sqrt(a_nm_up/a_nm_down))

def B_nm(n,m):
    b_nm_up=(2*n+1)*(n+m-1)*(n-m-1)
    b_nm_down=(n-m)*(n+m)*(2*n-3)
    return (sqrt(b_nm_up/b_nm_down))

def C_nm(n,m):
    sigma=0
    if m==2:
        sigma=1
    c_nm_up=(2*n+1)*(n+m-2)*(n+m-3)
    c_nm_down=(2*n-3)*(n+m)*(n+m-1)
    c_nm_left=(1+sigma)
    return (sqrt(c_nm_left*c_nm_up/c_nm_down))

def D_nm(n,m):
    sigma=0
    if m==2:
        sigma=1
    d_nm_up=(n-m+1)*(n-m+2)
    d_nm_down=(n+m)*(n+m-1)
    d_nm_left=1+sigma
    return (sqrt(d_nm_left*d_nm_up/d_nm_down))    

def H_nm(n,m):
    h_nm_up=(2*n+1)*(n-m)*(n-m-1)
    h_nm_down=(2*n-3)*(n+m)*(n+m-1)
    
    return (sqrt(h_nm_up/h_nm_down))


#endregion

#region Belikov递推法

#计算系数N
def Cal_Nnm(n:int):
    """
    计算Nnm系数矩阵
    
    Args:
    -n: 阶数
    -m: 次数
    
    Return:
    -Nnm: Nnm系数组成的矩阵
    """
    Nnm=np.zeros([n+1,n+1],dtype=object)
    for i in range(n+1):
        for j in range(n+1):
            Nnm[i,j]=(Nnm[i,j])
    Nnm[0,0]=Nnm[1,0]=Nnm[1,1]=(1)
    
    for i in range(2,n+1):
        Nnm[i,i]=(sqrt((2*i-1)/(2*i)))*Nnm[i-1,i-1]
    
    
    for i in range(0,n):
        for j in range(i+1,n+1):
            Nnm[j,i]=(sqrt((j**2-i**2)/(j**2)))*Nnm[j-1,i]    
    
    return Nnm

#将完全规划转为非正常
def Trans_Plm_To_Unnormal(Pnm,n:int,m:int,Nnm):
    Nnm_=Nnm[n,m]
    k=(sqrt(2*n+1))
    return Pnm/(k*Nnm_)

#将非正常转为完全规划
def Trans_Unnormal_To_Plm(Unnormal,n:int,m:int,Nnm):
    Nnm_=Nnm[n,m]
    k=(sqrt(2*n+1))
    return Unnormal*k*Nnm_
#endregion

def CalPnm(Pnm:Pnm,Method:str):
    """
    计算规格化缔合勒让得函数系数
    
    Args:
    -Pnm: 自定义的Pnm类实例化
    -Method: 递推计算的方法,支持“跨阶数递推”,“Belikov递推”和“标准向前列递推”三种方法,字符串形式。
    
    Return:
    P: size为(n×m)的Pnm矩阵
    """
    #跨阶数递推公式计算函数
    if Method=="跨阶数递推":
        theta=Pnm.theta
        n=Pnm.n;m=Pnm.m
        P=np.zeros([n+1,m+1],dtype=object)
        t=(cos(theta))
        P[0,0]=(1); P[1,0]=(sqrt(3)*cos(theta)); P[1,1]=(sqrt(3)*sin(theta))
        for j in range(0,2):
            for i in range(2,n+1):
                a_nm=A_nm(i,j);b_nm=B_nm(i,j)
                P[i,j]=a_nm*t*P[i-1,j]-b_nm*P[i-2,j]
                
        for j in range(2,m+1):
            for i in range(j,n+1):
                c_nm=C_nm(i,j);d_nm=D_nm(i,j);h_nm=H_nm(i,j)
                P[i,j]=c_nm*P[i-2,j-2]+h_nm*P[i-2,j]-d_nm*P[i,j-2]                
        return P
    
    #Belikov递推法
    if Method=="Belikov递推":
        theta=Pnm.theta
        n=Pnm.n;m=Pnm.m
        P=np.zeros([n+1,m+1],dtype=object)
        Pnormal00=(1);Pnormal10=(sqrt(3)*cos(theta));Pnormal11=(sqrt(3)*sin(theta))
        Nnm=Cal_Nnm(n)
            
        P[0,0]=Trans_Plm_To_Unnormal(Pnormal00,0,0,Nnm);P[1,0]=Trans_Plm_To_Unnormal(Pnormal10,1,0,Nnm)
        P[1,1]=Trans_Plm_To_Unnormal(Pnormal11,1,1,Nnm)

        t=(cos(theta))
        u=(sin(theta))     
        
        for i in range(2,n):
            for j in range(0,i+1):
                if j==0:
                    P[i,0]=t*P[i-1,0]-u*P[i-1,1]/(2)
                else:
                    P[i,j]=t*P[i-1,j]-u*(P[i-1,j+1]/(4)-P[i-1,j-1])
        
        #当i==n时,特判
        for j in range(0,n+1):
            if j==0:
                P[n,0]=t*P[n-1,0]-u*P[n-1,1]/(2)
            elif(j<n):
                P[n,j]=t*P[n-1,j]-u*(P[n-1,j+1]/(4)-P[n-1,j-1])
            else:
                P[n,j]=t*P[n-1,j]+u*P[n-1,j-1]

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                P[i,j]=Trans_Unnormal_To_Plm(P[i,j],i,j,Nnm)
        return P
    
    if Method=="标准向前列递推":
        theta=Pnm.theta
        n=Pnm.n;m=Pnm.m
        P=np.zeros((n+1,m+1))
        P[0,0] = 1
        P[1,0] = sqrt(3) * cos(theta)
        P[1,1] = sqrt(3) * sin(theta)
        for l in range(2,n+1):
            f1=sqrt(1 + 1.0 / 2 / l)
            f2=sqrt(2 * l + 1)
            for m in range(0,l-1):
                f3 = sqrt((2.0 * l + 1.0) / (l - m) / (l + m))
                f4 = sqrt(2.0 * l - 1.0)
                f5 = sqrt((l - m - 1) * (l + m - 1) * 1.0 / (2.0 * l - 3.0))
                P[l,m] = f3 * (f4 * cos(theta) * P[l - 1,m] - f5 * P[l - 2,m])
            P[l,l - 1] = f2 * cos(theta) * P[l - 1,l - 1]
            P[l,l] = f1 * sin(theta) * P[l - 1,l - 1]
        return P
    
#格式化输出函数
def Standard_Output(result)->str:
    """
    格式化输出Plm函数
    
    Args:
    -result:CalPnm(Pnm)得到的Pnm矩阵
    
    Return:
    -res: 格式化输出结果
    
    """
    res=" {:<{width}}".format("l",width=4),"{:<{width}}".format("m",width=4),"Plm"+'\n'
    print(result)
    for i in range(result.shape[0]):

        for j in range(0,i+1):
            tempstr="{:<{width}}".format(i,width=4),"{:<{width}}".format(j,width=4),"{:.4e}".format(result[i,j])+'\n'
            res+=tempstr
            print(tempstr)
    return res

