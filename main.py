from __future__ import division
__author__ = 'admin'


# __author__ = 'admin'

# -----------Import packages--------------
import pandas as pn
import numpy as np
from progressbar import *               # just a simple progress bar
import csv
import pickle

#------------Function Define------------------
# def is_selected (value1 , value2):
#     tempp =np.asarray(selected_item[value1],list,'c')
#     tmp=(len(np.extract(tempp==value2,tempp)))
#     if tmp ==0:
#         return 0
#     else :
#         return 1
# -----------Rred data -------------------
data = pn.read_table('train.data')
# data_test = pn.read_table('test.data')

## -----------Variable Initialization--------------##
# print data.item_id
# print data.rate
user_id =data.user_id
item_id = data.item_id
user_id=np.asarray(user_id,list,'c')
item_id=np.asarray(item_id,list,'c')
N=50
core_size =930
target_user=10
K=5
# ----------Progress bar----------------------
widgets = ['Progress: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),' ', ETA()] #see docs for other options

## ------------Degree-----------
user, user_degree = np.unique(user_id, return_counts=True)
item, item_degree = np.unique(item_id, return_counts=True)
#tmp1= np.loadtxt('user.txt',dtype=float,delimiter=',')
#tmp2= np.loadtxt('item.txt',dtype=float,delimiter=',')
#user, user_degree = tmp1[:,0] ,tmp1[:,1]
#item, item_degree = tmp2[:,0] ,tmp2[:,1]
us=np.array([user ,user_degree]).T
it=np.array([item ,item_degree]).T
#np.savetxt("user.txt",us, delimiter=',')
#np.savetxt("item.txt",it, delimiter=',')
print np.amax(user_id)
print np.amax(item)
#--------------Selected item by each user--------------
selected_item =[]
for i in user:
   temp= item_id[np.where(user_id==i)]
   selected_item.append(temp)
# ----------------is_selected----------------
pbar = ProgressBar(widgets=widgets, maxval=len(user_id))
pbar.start()
print "is selected calculation"
counter=0
# --------------------------------------------
is_selected=np.zeros([np.amax(user)+1,np.amax(item)+1])

for i in range(0,len(user_id)):
        is_selected[user_id[i]][item_id[i]]=1
        counter +=1
        pbar.update(counter)
pbar.finish()
#np.savetxt('is_selected.txt',is_selected, delimiter=',')
#--------------User similarity----------------
pbar = ProgressBar(widgets=widgets, maxval=pow(len(user),2))
counter=0
pbar.start()
print "\n Calculation of User similarity progress:"
# ------------------------------------------------
user_subscription=np.zeros([len(user),len(user)])
user_similarity=np.zeros([len(user),len(user)])
for i in user:
    for j in user:
        if i==j:
            user_subscription[i-1][j-1]=0
            user_similarity[i-1][j-1]=0
            counter +=1
            pbar.update(counter)
        else:
            user_subscription[i-1][j-1]=len(set(selected_item[int(i-1)]) & set(selected_item[int(j-1)]))
            user_similarity[i-1][j-1]=user_subscription[i-1][j-1]/np.sqrt(user_degree[i-1]*user_degree[j-1])
            counter +=1
            pbar.update(counter)
pbar.finish()
# ---------------Top-N ----------------------
pbar = ProgressBar(widgets=widgets, maxval=len(user))
pbar.start()
print "\n Calculation of Top-N progress:"
# -----------------------------------------------
top_n=np.zeros([len(item),N])
for i in user:
    top_n[i-1]=np.argpartition(user_similarity[i-1],-N)[-N:]
    pbar.update(i)
pbar.finish()
#---------------Count of appearence in Top-N --------
users ,count_appear =np.unique(top_n, return_counts=True)
count_appear=count_appear.astype(int)
print len(count_appear)
#
#
# ----------------core selection ---------------
#
core=users[(np.argpartition(count_appear,-core_size)[-core_size:])]
core =core.astype(int)
#
#
# ----------------KNN --------------------
#

# knn=core[(np.argpartition(user_similarity[target_user-1][core],-K)[-K:])]
knn= core
print knn
knn=knn.astype(int)
###-------------- ---------------------
#
#-------KNN item----------
selected_item=np.asarray(selected_item,list,'c')
temp=selected_item[knn]
knn_items=[]
for i in range(0,len(temp)):
    for j in range(0,len(temp[i])):
        knn_items.append(temp[i][j])
knn_items=np.asarray(knn_items,list,'c')
knn_items=knn_items.astype(int)
knn_items= np.unique(knn_items)

# --------W calculation --------------
pbar = ProgressBar(widgets=widgets, maxval=len(knn_items)*len(knn_items))
pbar.start()
print "\n Calculation of W progress:"
# -------------------W------------------------
knn=user[knn]
knn=knn.astype(int)
print knn
counter =0
W={}
print len(knn_items)
for a in knn_items:
    for b in knn_items:
        tmp=0
        beta=np.where(item==b)
        alpha=np.where(item==a)
        for j in knn:
            u=j-1
            tmp=((1/user_degree[u])*is_selected[u,alpha]*is_selected[u,beta])+tmp
        W[a,b]=tmp*(1/item_degree[beta])
        counter=counter+1
        pbar.update(counter)
# --------------Save in file ------------
output = open('w.pkl', 'wb')
pickle.dump(W, output)
output.close()
pbar.finish()
print "Mission complete"
# print core
#
# print knn
# print(user[knn])
# print user_degree[10]
# print len(selected_item)
# print len(core)


# print selected_item
#
# print user_degree[item_id[0]]
# # print user, user_degree
# print item_id[0]



