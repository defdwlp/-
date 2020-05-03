# -*- coding: utf-8 -*-

#如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

import numpy as np

import matplotlib.pyplot as plt
import operator
import random
'''
1.初始化：随机生成种群大小为n的种群

2.计算适应度值：当重量大于限定值时:适应度为0；
              当重量小于或等于限定值时:计算价值
3.采用轮盘赌选择父母
4.进行交叉操作
5.进行变异操作
6.选出q个精英个体与子代共同组成新种群
 '''
while(1):
    answer = input("是否自己输入参数（y/n）：")
    if answer == 'y' or answer == 'Y':
        weight_2 = input("请输入重量：")
        weight_2 = weight_2.split(" ")
        weight = []
        for i in weight_2:
            weight.append(int(i))
        value_2 = input("请输入价值：")
        value_2 =value_2.split(" ")
        value = []
        for i in value_2:
            value.append(int(i))
        weight_limit = int(input("请输入背包限制重量："))
        pop_size = int(input("请输入种群规模："))
        mu = float(input("请输入变异率："))
        n_generations =int(input("请输入迭代次数："))
        break
    elif answer == 'n' or answer == 'N':
        weight = [71,34 ,82, 23, 1, 88 ,12 ,57, 10, 68, 5 ,33, 37, 69 ,98 ,24 ,26 ,83 ,16 ,26] #物品重量
        value = [26, 59, 30, 19 ,66, 85, 94 ,8 ,3, 44 ,5, 1, 41 ,82 ,76, 1 ,12 ,81, 73, 32] #物品价值
        weight_limit = 420
        pop_size = 200
        n_generations = 200# 迭代次数
        mu = 0.2#变异率

        break
    else:
        print("请重新输入正确的选项")


value_best = []
#初始化种群
def init():
    global pop,pop_size,q
    global pop_child,weight_pop,value_pop
    pop =  np.random.randint(0,2,(pop_size,len(weight)))
    pop_child = []
    weight_pop = [0] * len(pop)
    value_pop = [0] * len(pop)
    q = int(0.1*len(pop))


#排序
def sort():
    for i in range(len(pop)):
        for j in range(len(weight)):
            weight_pop[i] += pop[i][j]*weight[j]

    # print("weight_pop:",weight_pop)
    for i in range(len(pop)):
        for j in range(len(weight)):
            if weight_pop[i]<=weight_limit:
                value_pop[i] += pop[i][j]*value[j]
            else:
                value_pop[i] = 0

    global c
    a = np.arange(0, len(pop) , 1).tolist()
    b = dict(zip(a, value_pop))
    c = sorted(b.items(), key=operator.itemgetter(1), reverse=True)
    # print("c:",c)

#计算适应度值
def getfitness():
    weight_pop = [0] * len(pop)
    value_pop = [0] * len(pop)
    for i in range(len(pop)):
        for j in range(len(weight)):
            weight_pop[i] += pop[i][j]*weight[j]

    # print("weight_pop:",weight_pop)
    for i in range(len(pop)):
        for j in range(len(weight)):
            if weight_pop[i]<=weight_limit:
                value_pop[i] += pop[i][j]*value[j]
            else:
                value_pop[i] = 0
    # print("value_pop:",value_pop)
    global c
    a = np.arange(0, len(pop) , 1).tolist()
    b = dict(zip(a, value_pop))
    c = sorted(b.items(), key=operator.itemgetter(1), reverse=True)
    # print("c:",c)

#轮盘赌选择父母
def chooseparents():


    global index_1, index_2
    d = [0]*len(pop)
    for i in range(len(pop)):
        d[i] = c[i][1]
    for i in range(1,len(pop)):
        d[i] += d[i-1]
    for i in range(len(pop)):
        if d[i] == 0:
            d[i] = 0
        else:
            d[i] = d[i]/d[-1]
    # print(d)
    ra_1 = random.uniform(0, 1)  # 生成随机数1
    ra_2 = random.uniform(0, 1)  # 生成随机数2
    index_1 = 0
    index_2 = 0
    for i in range(len(pop)):
        if ra_1 < d[i]:
            index_1 = i
            break
    for i in range(len(pop)):
        if ra_2 < d[i]:
            index_2 = i
            break
    # print("ra_1:",ra_1,"ra_2:",ra_2)
    # print("index_1:",index_1,"index_2:",index_2)
    # print(c[index_1][0],c[index_2][0])
    # print(pop[c[index_1][0]],pop[c[index_2][0]])

#交叉操作生成子代
def crossover():
    site = random.randint(1, len(weight)-1)
    # print("site:",site)
    parent_1_1 = pop[c[index_1][0]][:site].tolist()
    parent_1_2 = pop[c[index_1][0]][site:].tolist()
    parent_2_1 = pop[c[index_2][0]][:site].tolist()
    parent_2_2 = pop[c[index_2][0]][site:].tolist()
    child_1 = parent_1_1+parent_2_2
    child_2 = parent_2_1+parent_1_2
    pop_child.append(child_1)
    pop_child.append(child_2)

#变异操作
def muta():
    weight_pop = [0] * len(pop)
    value_pop = [0] * len(pop)
    for i in range(len(pop)):
        for j in range(len(weight)):
            weight_pop[i] += pop[i][j]*weight[j]

    # print("weight_pop:",weight_pop)
    for i in range(len(pop)):
        for j in range(len(weight)):
            if weight_pop[i]<=weight_limit:
                value_pop[i] += pop[i][j]*value[j]
            else:
                value_pop[i] = 0
    # print("value_pop:",value_pop)
    global c
    a = np.arange(0, len(pop) , 1).tolist()
    b = dict(zip(a, value_pop))
    c = sorted(b.items(), key=operator.itemgetter(1), reverse=True)


    for i in range(2,len(pop)):
        if i != c[0][0]:
            ra_3 = random.uniform(0, 1)  # 变异生成随机数3
            site_2 = random.randint(0,len(pop[0])-1)
            if ra_3 <= mu:
                if pop[i][site_2] == 0:
                    pop[i][site_2] =1
                else:
                    pop[i][site_2] = 0
        else:
            continue

#选择q个最优值加入子代中
def select_q():
    weight_pop = [0] * len(pop)
    value_pop = [0] * len(pop)
    for i in range(len(pop)):
        for j in range(len(weight)):
            weight_pop[i] += pop[i][j]*weight[j]

    # print("weight_pop:",weight_pop)
    for i in range(len(pop)):
        for j in range(len(weight)):
            if weight_pop[i]<=weight_limit:
                value_pop[i] += pop[i][j]*value[j]
            else:
                value_pop[i] = 0
    # print("value_pop:",value_pop)
    global c
    a = np.arange(0, len(pop) , 1).tolist()
    b = dict(zip(a, value_pop))
    c = sorted(b.items(), key=operator.itemgetter(1), reverse=True)
    # print(c)
    for i in range(q):
        # print(c[i][0])
        pop_child.append(pop[c[i][0]])


init()
for i in range(n_generations):
    getfitness()

    for j in range(pop_size//2):
        chooseparents()
        crossover()

    select_q()
    pop = np.array(pop_child)
    muta()
    pop_child = []
    value_best.append(c[0][1])
    print("当前为第%d代种群"%(i+1),"当前最优值为%d"%c[0][1])
print("物品重量为：",value)
print("物品价值为：",weight)
print("背包限制重量为：",weight_limit)
print("计算完毕最优值为：%d"%c[0][1])
value_best2 = c[0][1]
getfitness()
a = np.arange(0, len(pop), 1).tolist()
b = dict(zip(a, value_pop))
c = sorted(b.items(), key=operator.itemgetter(1), reverse=True)
plt.plot(range(1,len(value_best)+1),
         value_best, c='b')
plt.xlabel('代数')
plt.ylabel('最优值')
plt.title("01背包问题的最优值随种群迭代的变化(最优值为%d)"%value_best2)
plt.show()