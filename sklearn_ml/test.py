# coding=utf-8
import pandas as pd
from  sklearn.preprocessing import OneHotEncoder
from  sklearn.preprocessing import LabelEncoder
from  sklearn.preprocessing import LabelBinarizer
# data={'color':['red','green','yellow'],'number':[23,34,52]}
# df=pd.DataFrame(data)
# print df
# print pd.get_dummies(df)
# encoder=LabelEncoder()
# shape1=encoder.fit_transform(df['color'].reshape(-1,1))
# print shape1
# encoder1=LabelBinarizer()
# print encoder1.fit_transform(shape1.reshape(-1,1))

# encoder=OneHotEncoder()
# print encoder.fit_transform(df['number'].reshape(-1,1)).toarray()
# print df['number'].reshape(-1,1)
#
import numpy as np
import pandas as pd
from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import Normalizer
#
# nd1=np.random.rand(3,1)*10
# nd2=np.random.rand(3,1)*100
# n3=np.concatenate((nd1,nd2),axis=1)
# df=pd.DataFrame(n3)
# print df
# transformer=StandardScaler()
# result_std=transformer.fit_transform(df)
# print result_std,'StandardScaler'
# transformer1=Normalizer()
# result_norm=transformer1.fit_transform(df)
# print result_norm,'Normalizer'
# L = [8,2,50,3]
# print L[::2]
#
# a={1:1,2:2,3:3}
# list1=[str(key) for key in a.keys()]
# list1.sort()
# print list1
#
# list=list()
# list=[i for i in range(2,100)]
# for x in list:
#   for item in range(x*2,100,x):
#     try:
#       list.remove(item)
#     except:
#         continue
# print ' '.join([str(i) for i in list])

# L = [0,2, 3, 4]
# L.sort()
# a=L[len(L)/2] if len(L)%2!=0 else (L[len(L)/2]+L[len(L)/2-1])*0.5
# print a

# L = [2, 8, 3, 50]
# n2=0
# n5=0
# for l in L:
#     while l%2==0:
#         l=l/2
#         n2+=1
#     while l%5==0:
#         l=l/5
#         n5+=1
# result=min(n2,n5)
# print result
#
# astr= "OurWorldIsFullOfLoVE"
#
# astr=astr.upper()
# if 'LOVE'in astr:
#     print 'LOVE'
# else:
#     print 'SINGLE'
# a="cagy"
# b=3
# new_a=''
# for i in a:
#     loc=ord(i)
#     new_loc=loc+b
#     if new_loc>122:
#         result=chr(new_loc-26)
#     else:
#         result=chr(new_loc)
#     new_a=new_a+result
# print new_a

# a=7
# a_bi=bin(a)
# print str(a_bi).count('1')
# import this
# print(this.s)
# s='PPALLL'
# print s.count('A')
# print s.count('LLL')
# print s.find('LLL')
# s = 'abcdefgf'
# k = 3
# s=list(s)
# for i in range(0,len(s),2*k):
#     s[i:i+k]=reversed(s[i:i+k])
#     print i+k,s[i+k]
#
# print ''.join(s)

# nums=[]
# ans = anchor = 0
# for i in range(len(nums)):
#     if  nums[i - 1] >= nums[i]: anchor = i
#     ans = max(ans, i - anchor + 1)
#
# print ans
# year='0012'
# result=366 if int(year)%4==0 and int(year)%100!=0 or int(year)%100==0 else 365
# print result
# A = [12, 28, 46, 32, 50]

# B = [50, 12, 32, 46, 28]
# result=[B.index(i) for i in A]
# print result
# moves='UD'
# print moves.count('U')==moves.count('D') and moves.count('L')==moves.count('R')

# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# t1=TreeNode()
# t1.val=1
# t1.left=TreeNode().left=1
# t2=TreeNode()
# t2.val=2

# left = 1
# right = 22
# result=range(left,right+1)
# print result
# for i in range(left,right+1):
#     print i
#     l=list(str(i))
#     print l
#     for j in l:
#         if j=='0'or i%int(j)!=0:
#             # result.remove(i)
#             print i
#             result.remove(i)
# print result
# nums=[1,4,3,2]
# nums=sorted(nums)
# result=sum(nums[::2])
# print result
# s="Let's take LeetCode contest"
# s=s.split(' ')
# s=[i[::-1] for i in s]
# print ' '.join(s)
# candies = [1,1,2,2,3,3]
# candies=sorted(candies)
# kinds=len(set(candies))
# n=len(candies)/2
# result=n if kinds>n else kinds
# print result
# nums1 = [4,1,2]
# nums2 = [1,3,4,2]
# num2_sort=sorted(nums2)
# result=0
# for num in nums1:
#     if nums2.index(num2_sort[num2_sort.index(num)+1])>=len(nums2) or nums1.index(num)==len(nums1)-1:
#         result=-1
#     else:
#         result=num2_sort[num2_sort.index(num)+1]
# n=7
# sn=bin(n).replace('0b','')
# result=True
# for i in range(1,len(sn)):
#     if sn[i]==sn[i-1]:
#         result=False
#         break
# print result
# nums=[1,1,2,2,5,5,7]
# print sum(set(nums))*2-sum(nums)
# nums=[1,1,0,1,1,1]
# l=''.join(map(str,nums)).split('0')
# result=max([i.count('1') for i in l])
# print result
# nums_matrix=np.mat(nums)
# print nums_matrix.T
# print nums_matrix.transpose()
# print nums_matrix
# slice=nums_matrix[0,1:3].copy()
# print slice
# slice[0,1]=2
# print nums_matrix
# import math
# vector1=np.mat([1,2,3])
# vector2=np.mat([4,5,6])
# print math.sqrt((vector1-vector2)*(vector1-vector2).T)
# print sum(abs(vector1-vector2))
# print np.dot(vector1,vector2.transpose())/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
# from sklearn.cluster import KMeans
# c=[4,8,15,21,21,24,25,28,34,465]
# c1=c
# k=KMeans(n_clusters=4)
# import numpy as np
# c=np.array(c)
# c=c.reshape(len(c),-1)
# k.fit_transform(c)
# center=k.cluster_centers_
# print center
# c2=k.predict(c)
# print c2
# import matplotlib.pyplot as plt
# c2=list(c2)
# class_dict={}
# for i in range(len(c2)):
#     if c2[i] not in class_dict.keys():
#         class_dict[c2[i]]=[]
#         class_dict[c2[i]].append(c1[i])
#     else:
#         class_dict[c2[i]].append(c1[i])
# print class_dict
# #直接这样用plot连成线了
# # plt.plot(class_dict[0],color='red')
# # plt.plot(class_dict[1],color='blue')
# # plt.plot(class_dict[2],color='yellow')
# # plt.plot(class_dict[3],color='green')
# #第一个和第二个参数是数据，长度相等，marker是形状，s是大小
# plt.scatter(class_dict[0],class_dict[0],color='yellowgreen',marker='o',s=40)
# plt.scatter(class_dict[1],class_dict[1],color='lightskyblue',marker='o',s=40)
# plt.scatter(class_dict[2],class_dict[2],color='gold',marker='o',s=40)
# plt.scatter(class_dict[3],class_dict[3],color='lightcoral',marker='o',s=40)
# # plt.scatter(center,center,color='r',marker='D',s=20)
# # plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation
#
# pause = False
#
#
# def simData():
#     t_max = 10.0
#     dt = 0.05
#     x = 0.0
#     t = 0.0
#     while t < t_max:
#         if not pause:
#             x = np.sin(np.pi * t)
#             t = t + dt
#         yield t, x
#
#
# def onClick(event):
#     global pause
#     pause ^= True
#
#
# def simPoints(simData):
#     t, x = simData[0], simData[1]
#     time_text.set_text(time_template % (t))
#     line.set_data(t, x)
#     return line, time_text
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# line, = ax.plot([], [], 'bo', ms=10)  # I'm still not clear on this stucture...
# ax.set_ylim(-1, 1)
# ax.set_xlim(0, 10)
#
# time_template = 'Time = %.1f s'  # prints running simulation time
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# fig.canvas.mpl_connect('button_press_event', onClick)
# ani = animation.FuncAnimation(fig, simPoints, simData, blit=False, interval=10,
#                               repeat=True)
# plt.show()
# print'\n'.join([''.join([('Iloveyou'[(x-y)%7]if((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3<=0 else' ')for x in range(-30,30)])for y in range(15,-15,-1)])
dic={}
dic[3]=6
print dic
dic[3]=7
print dic




