from __future__ import division
__author__ = 'admin'


# __author__ = 'admin'

# -----------Import packages--------------
import pandas as pn
import numpy as np
from progressbar import *               # just a simple progress bar
import csv
import pickle


data = pn.read_table('train.data')
data_test = pn.read_table('test.data')
#pkl_file = open('w.pkl', 'rb')
#W= pickle.load(pkl_file)
#pkl_file.close()
## -----------Variable Initialization--------------##
# print data.item_id
# print data.rate
test_user_id =data_test.user_id
test_item_id =data_test.item_id
#---------------Load test data---------------------#
user_id =data.user_id
item_id = data.item_id
#-------------------------------------------------
user_id=np.asarray(user_id,list,'c')
item_id=np.asarray(item_id,list,'c')
test_user_id=np.asarray(test_user_id,list,'c')
test_item_id=np.asarray(test_item_id,list,'c')
#--------------------------------------------------
L=10 #Recommend number
N=10
#core_size =200
target_user=0
K=5
# ----------Progress bar----------------------
widgets = ['Progress: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),' ', ETA()] #see docs for other options

## ------------Degree-----------
user, user_degree = np.unique(user_id, return_counts=True)
item, item_degree = np.unique(item_id, return_counts=True)
test_user, test_user_degree = np.unique(test_user_id, return_counts=True)
test_item, test_item_degree = np.unique(test_item_id, return_counts=True)
us=np.array([user ,user_degree]).T
it=np.array([item ,item_degree]).T

print np.amax(user_id)
print np.amax(item)
#--------------Selected item by each user--------------
selected_item =[]
for i in user:
   temp= item_id[np.where(user_id==i)]
   selected_item.append(temp)
#---------------for test-----------
test_selected_item ={}
for i in test_user:
   temp= test_item_id[np.where(test_user_id==i)]
   test_selected_item[i]=temp
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

recall={}
print "\n Recommendation:"
pbar = ProgressBar(widgets=widgets, maxval=len(test_user))
pbar.start()
counter =0
for USER in test_user:
    target_user=USER
#
#
# ----------------KNN --------------------
#
    knn=core[(np.argpartition(user_similarity[target_user-1][core],-K)[-K:])]
    knn= core
    # print knn
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
    #----------f--------------------
    f_knn_items = set(knn_items) - set(test_selected_item[i])
    f_knn_items=np.array(list(f_knn_items))
    f_knn_items.sort()
#------------Weight------------------
    weights=np.zeros([len(f_knn_items),len(f_knn_items)])
    a=b=0
    for i in f_knn_items:
        for j in f_knn_items:
            weights[a][b]=float(W[i,j])
            b+=1
        b=0
        a+=1
    f=np.ones([len(f_knn_items),1])
    rec_list=np.dot(weights,f)
    rec_list=rec_list.T
    ind =(np.argsort(rec_list))[0][-L:]
    recommend=f_knn_items[ind]
    recall[USER]=(np.intersect1d(test_selected_item[USER],recommend)).size/(test_selected_item[USER]).size
    counter +=1
    pbar.update(counter)
pbar.finish()

# --------W calculation --------------

# -------------------W------------------------
# knn=user[knn]
# knn=knn.astype(int)
# print knn

# --------------Save in file ------------
# output = open('w.pkl', 'wb')
# pickle.dump(W, output)
# output.close()
# pbar.finish()
file_name='L_'+str(L)+'core_size_'+str(core_size)+'N_'+str(N)+'K_'+str(K)+'.csv'
f=open(file_name,'wb')
wrtr=csv.DictWriter(f,recall.keys())
wrtr.writerow(recall)
f.close()
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



