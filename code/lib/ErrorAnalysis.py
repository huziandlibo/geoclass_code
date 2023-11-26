import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .Pnm import Pnm,CalPnm
plt.rcParams['font.family']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

#计算某一阶数的相对精度
def Cal_Error(result,n)->float:
    temp_Tn=0
    for i in range(0,n+1):
        temp_Tn+=result[n,i]**2
    Tn=abs(temp_Tn-(2*n+1))/(2*n+1)
    return Tn

#默认使用n=2160来评估计算方法的精度,角度可以自定义
def PaintingErrorCurve(theta,Method:str,n=3000)->Line2D:
    """
    绘制计算缔合勒让得函数的相对精度曲线,默认阶数为2160
    
    Args:
    -theta: 归化余纬
    -n: 默认n=2160
    
    Return:
    -Line2D: 相对精度曲线对象
    """
    P=Pnm(n,n,theta)
    result=CalPnm(P,Method)
    Tns=[Cal_Error(result,i) for i in range(result.shape[0])]
    N=range(0,result.shape[0])
    plt.figure(dpi=200)
    pic=plt.scatter(N,Tns,s=0.5,color='grey')
    plt.ylabel('相对精度')
    plt.yscale('log')
    plt.xlabel('阶数')
    plt.ylim([0,1.1*max(Tns)])
    plt.show()
    return pic 
  