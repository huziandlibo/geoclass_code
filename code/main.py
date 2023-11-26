import numpy as np
from multiprocessing import Pool
from lib.Calculate import Calculate
import time
import matplotlib.pyplot as plt
from lib.Draw import DrawMap
import pandas as pd
# 其余部分保持不变

world_Bs=np.linspace(-89.5,89.5,int(179/1)+1)
world_Ls=np.linspace(0,360,int(360/1)+1)
area_Bs=np.linspace(0,40,int(40/(5/60))+1)
area_Ls=np.linspace(100,140,int(40/(5/60))+1) 

#计算子矩阵
def Parallel_Calculate(row_start,col_start,row_size,col_size,Bs:np.ndarray,Ls:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    sub_result_T=np.zeros((row_size,col_size))
    sub_result_N=np.zeros((row_size,col_size))
    sub_result_delg=np.zeros((row_size,col_size))
    sub_result_vardelg=np.zeros((row_size,col_size))
    for i in np.arange(row_size):
        for j in np.arange(col_size):
            sub_result_T[i,j],sub_result_N[i,j],sub_result_delg[i,j],sub_result_vardelg[i,j]=Calculate(Bs[row_start+i],Ls[col_start+j],0)
    return sub_result_T,sub_result_N,sub_result_delg,sub_result_vardelg

#并行计算打包函数
def Parallel_Calculate_wrapper(args):
    result_T,result_N,result_delg,result_vardelg=Parallel_Calculate(*args)
    print(f"Submatrix ({args[0]}, {args[1]}) calculation completed")
    return result_T,result_N,result_delg,result_vardelg,args[0],args[1],args[2],args[3]

#划分矩阵并返回参数
def Get_args(row:int,col:int,sub_size:int,Bs:np.ndarray,Ls:np.ndarray)->tuple[list[tuple],str]:
    if sub_size > min(row,col):
        raise ValueError("子矩阵大小>母矩阵大小,请重新选择合适的子矩阵大小!")
    args = [(i,j,sub_size,sub_size,Bs,Ls) for i in range(0, (row if(row%sub_size)==0 else (row//sub_size)*sub_size), sub_size)
                for j in range(0, (col if(col%sub_size)==0 else (col//sub_size)*sub_size), sub_size)]
    
    #剩余<子矩阵大小的部分
    if row%sub_size==0 and not col%sub_size==0:
        args.append((0,col//sub_size*sub_size,row,col%sub_size,Bs,Ls))
        return args,"col"
    elif col%sub_size==0 and not row%sub_size==0:
        args.append((row//sub_size*sub_size,0,row%sub_size,col,Bs,Ls))
        return args,"row"
    elif not row%sub_size==0 and not col%sub_size==0:
        args.append((row//sub_size*sub_size,0,row%sub_size,col,Bs,Ls))
        args.append((0,col//sub_size*sub_size,row//sub_size*sub_size,col%sub_size,Bs,Ls))
        return args,"row and col"
    else:
        return args,"none"

if __name__ == '__main__':
    
    num_workers = 10  # 并行计算进程数
    
    #region 计算全球区域的
    row = len(world_Bs)  
    col = len(world_Ls) 
    sub_size=10
    print(row,col)
    start_time=time.time()
    with Pool(num_workers) as pool:
        args,info=Get_args(row,col,sub_size,world_Bs,world_Ls)
        results = []
        for arg in args:
            result = pool.apply_async(Parallel_Calculate_wrapper, (arg,))
            results.append(result)
            print(f"Submatrix ({arg[0]}, {arg[1]})<size:({arg[2]},{arg[3]})> calculation started")
        
        result_T_list=[]
        result_N_list=[]
        result_delg_list=[]
        result_vardelg_list=[]
        row_index_list=[]
        col_index_list=[]
        row_size_list=[]
        col_size_list=[]
        # 获取并处理计算结果
        for result in results:
            result_T,result_N,result_delg,result_vardelg,r_index,c_index,r_size,c_size=result.get()
            result_T_list.append(result_T)
            result_N_list.append(result_N)
            result_delg_list.append(result_delg)
            result_vardelg_list.append(result_vardelg)
            row_index_list.append(r_index)
            col_index_list.append(c_index)
            row_size_list.append(r_size)
            col_size_list.append(c_size)

        #输出计算时间
        print("All submatrix calculations completed")
        end_time=time.time()
        print(f"运行时间:{end_time-start_time:.2f}秒")
        
        #region 合并子矩阵
        col_count=col//sub_size
        row_count=row//sub_size
        rest_count=len(result_T_list)-row_count*col_count
        count=0
        final_T=np.zeros((row,col))
        final_N=np.zeros((row,col))
        final_delg=np.zeros((row,col))
        final_vardelg=np.zeros((row,col))
        for index,_ in enumerate(result_T_list):
            start_row_index=row_index_list[index]
            start_col_index=col_index_list[index]
            row_size=row_size_list[index]
            col_size=col_size_list[index]
            end_row_index=start_row_index+row_size
            end_col_index=start_col_index+col_size
            final_T[start_row_index:end_row_index,start_col_index:end_col_index]=result_T_list[index]
            final_N[start_row_index:end_row_index,start_col_index:end_col_index]=result_N_list[index]
            final_delg[start_row_index:end_row_index,start_col_index:end_col_index]=result_delg_list[index]
            final_vardelg[start_row_index:end_row_index,start_col_index:end_col_index]=result_vardelg_list[index]
        #endregion
        print(final_T)

        # 在不同的子图上调用 DrawMap 函数
        fig_T=DrawMap(final_T,world_Bs,world_Ls)
        fig_N=DrawMap(final_N,world_Bs,world_Ls)
        fig_delg=DrawMap(final_delg,world_Bs,world_Ls)
        fig_vardelg=DrawMap(final_vardelg,world_Bs,world_Ls)
        fig_T.savefig("../figures/T.png") #扰动位
        fig_N.savefig("../figures/N.png") #大地水准面高
        fig_delg.savefig("../figures/delg.png") #重力异常
        fig_vardelg.savefig("../figures/vardelg.png") #重力扰动
        
        
        #保存计算结果
        col=world_Ls
        row=world_Bs
        
        df_T = pd.DataFrame(final_T, columns=[str(x)+"°" for x in col])
        df_T.index = [str(x)+"°" for x in row]
        df_T.to_excel('../results/world_T.xlsx')
        
        df_N = pd.DataFrame(final_N, columns=[str(x)+"°" for x in col])
        df_N.index = [str(x)+"°" for x in row]
        df_N.to_excel('../results/world_N.xlsx')
        
        df_delg = pd.DataFrame(final_delg, columns=[str(x)+"°" for x in col])
        df_delg.index = [str(x)+"°" for x in row]
        df_delg.to_excel('../results/world_delg.xlsx')
        
        df_vardelg = pd.DataFrame(final_vardelg, columns=[str(x)+"°" for x in col])
        df_vardelg.index = [str(x)+"°" for x in row]
        df_vardelg.to_excel('../results/world_vardelg.xlsx')
        plt.show()
    #endregion
    
    
    #region 计算局部区域的
    # row = len(area_Bs)  
    # col = len(area_Ls)
    # print(row,col)
    # sub_size=10
    # start_time=time.time()
    # with Pool(num_workers) as pool:
    #     args,info=Get_args(row,col,sub_size,area_Bs,area_Ls)
    #     results = []
    #     for arg in args:
    #         result = pool.apply_async(Parallel_Calculate_wrapper, (arg,))
    #         results.append(result)
    #         print(f"Submatrix ({arg[0]}, {arg[1]})<size:({arg[2]},{arg[3]})> calculation started")
        
    #     result_T_list=[]
    #     result_N_list=[]
    #     result_delg_list=[]
    #     result_vardelg_list=[]
    #     row_index_list=[]
    #     col_index_list=[]
    #     row_size_list=[]
    #     col_size_list=[]
    #     # 获取并处理计算结果
    #     for result in results:
    #         result_T,result_N,result_delg,result_vardelg,r_index,c_index,r_size,c_size=result.get()
    #         result_T_list.append(result_T)
    #         result_N_list.append(result_N)
    #         result_delg_list.append(result_delg)
    #         result_vardelg_list.append(result_vardelg)
    #         row_index_list.append(r_index)
    #         col_index_list.append(c_index)
    #         row_size_list.append(r_size)
    #         col_size_list.append(c_size)

    #     # 打印语句
    #     print("All submatrix calculations completed")
    #     end_time=time.time()
    #     print(f"运行时间:{end_time-start_time:.2f}秒")
        
    #     #region 合并子矩阵
    #     col_count=col//sub_size
    #     row_count=row//sub_size
    #     rest_count=len(result_T_list)-row_count*col_count
    #     count=0
    #     final_T=np.zeros((row,col))
    #     final_N=np.zeros((row,col))
    #     final_delg=np.zeros((row,col))
    #     final_vardelg=np.zeros((row,col))
    #     for index,_ in enumerate(result_T_list):
    #         start_row_index=row_index_list[index]
    #         start_col_index=col_index_list[index]
    #         row_size=row_size_list[index]
    #         col_size=col_size_list[index]
    #         end_row_index=start_row_index+row_size
    #         end_col_index=start_col_index+col_size
    #         final_T[start_row_index:end_row_index,start_col_index:end_col_index]=result_T_list[index]
    #         final_N[start_row_index:end_row_index,start_col_index:end_col_index]=result_N_list[index]
    #         final_delg[start_row_index:end_row_index,start_col_index:end_col_index]=result_delg_list[index]
    #         final_vardelg[start_row_index:end_row_index,start_col_index:end_col_index]=result_vardelg_list[index]
    #     #endregion
    #     print(final_N)
        

    #     # 在不同的子图上调用 DrawMap 函数
    #     fig_T=DrawMap(final_T,area_Bs,area_Ls)
    #     fig_N=DrawMap(final_N,area_Bs,area_Ls)
    #     fig_delg=DrawMap(final_delg,area_Bs,area_Ls)
    #     fig_vardelg=DrawMap(final_vardelg,area_Bs,area_Ls)
    #     fig_T.savefig("../figures/area_T.png") #扰动位
    #     fig_N.savefig("../figures/area_N.png") #大地水准面高
    #     fig_delg.savefig("../figures/area_delg.png") #重力异常
    #     fig_vardelg.savefig("../figures/area_vardelg.png") #重力扰动
        
    #     col=area_Ls
    #     row=area_Bs
    #     # 保存计算结果
    #     df_T = pd.DataFrame(final_T, columns=[str(x)+"°" for x in col])
    #     df_T.index = [str(x)+"°" for x in row]
    #     df_T.to_excel('../results/area_T.xlsx')
        
    #     df_N = pd.DataFrame(final_N, columns=[str(x)+"°" for x in col])
    #     df_N.index = [str(x)+"°" for x in row]
    #     df_N.to_excel('../results/area_N.xlsx')
        
    #     df_delg = pd.DataFrame(final_delg, columns=[str(x)+"°" for x in col])
    #     df_delg.index = [str(x)+"°" for x in row]
    #     df_delg.to_excel('../results/area_delg.xlsx')
        
    #     df_vardelg = pd.DataFrame(final_vardelg, columns=[str(x)+"°" for x in col])
    #     df_vardelg.index = [str(x)+"°" for x in row]
    #     df_vardelg.to_excel('../results/area_vardelg.xlsx')
        
    #     plt.show()
    #endregion
    