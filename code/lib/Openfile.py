import numpy as np

def Get_CnmSnm(N:int=360)->tuple[np.ndarray,np.ndarray]:
    Cnm=np.zeros((N+1,N+1))
    Snm=np.zeros((N+1,N+1))
    with open('../src/SGG-UGM-2.gfc',"r",encoding='utf-8') as file:
        line=file.readline()
        count=0
        while line:
            if line[0:3]=="gfc" and count <(N+1)*(N+2)/2:
                data_info=line.strip().split()
                n=int(data_info[1])
                m=int(data_info[2])
                Cnm[n,m]=float(data_info[3])
                Snm[n,m]=float(data_info[4])
                count+=1
                line=file.readline() 
            elif count==(N+1)*(N+2)/2:
                break
            else:
                line=file.readline()
    return Cnm,Snm


if __name__ =="__main__":
    t1,t2=Get_CnmSnm()
    print(t1,t2)
    
    