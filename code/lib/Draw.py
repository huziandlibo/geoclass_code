import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def DrawMap(data:np.ndarray,ylim:np.ndarray,xlim:np.ndarray):
    fig=plt.figure(figsize=(10,6))
    ymin=np.min(ylim)
    ymax=np.max(ylim)
    xmin=np.min(xlim)
    xmax=np.max(xlim)
    ax=fig.add_subplot(111)
    map = Basemap(projection='cyl', llcrnrlat=ymin, urcrnrlat=ymax, llcrnrlon=xmin, urcrnrlon=xmax, ax=ax)

    # 绘制海岸线和国家边界
    map.drawcoastlines(linewidth=0.5)
    map.drawcountries(linewidth=0.5)

    # 设置 x 轴和 y 轴的刻度标签可见，线宽为 0，而不绘制经纬线
    ax.xaxis.set_tick_params(labeltop=True, bottom=False, top=False, labelbottom=False, width=0)
    ax.yaxis.set_tick_params(labelright=True, left=False, right=False, labelleft=False, width=0)

    # 设置 x 轴和 y 轴的刻度标签文字和刻度标签位置
    xticks = np.linspace(xmin,xmax,7)
    yticks = np.linspace(ymin,ymax,7)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{:.2f}°'.format(xtick) for xtick in xticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels(['{:.2f}°'.format(ytick) for ytick in yticks])
    
    

    ax.xaxis.tick_top() # 设置 x 轴刻度在坐标上方显示
    ax.xaxis.set_label_position('top')
    ax.yaxis.tick_left() # 设置 y 轴刻度在坐标左边显示
    ax.yaxis.set_label_position('left')
    

    # 设置 x 轴和 y 轴的刻度线可见
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')

    # 设置 x 轴和 y 轴的刻度线的长度,宽度
    ax.tick_params(which='both', length=5)
    ax.tick_params(which='both', width=1)

    # 创建边界值列表和范围
    cmap = plt.cm.jet

    # 使用 pcolor 绘制热力图
    x, y = np.meshgrid(xlim, ylim)
    pcm = ax.pcolormesh(x, y, data, cmap=cmap)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.09)
    cbar = plt.colorbar(pcm, cax=cax)

    # 设置 colorbar 的纵横比与热力图相同
    ax.set_aspect('equal')

    return fig
    