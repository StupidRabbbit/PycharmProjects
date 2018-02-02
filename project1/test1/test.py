import linecache
from PERM_API_By_PERM import API_BY_PERM

class_list=[]
filename='E:/JetBrains/PycharmProjects/project1/test1/result.txt'
f=open('E:/JetBrains/PycharmProjects/project1/test1/result.txt','w')
for value in API_BY_PERM.values():
    for api in value:
        api_class=api.split(';')[0].split('$')[0]
        print api_class
        class_list.append(api_class)
api_list=list(set(class_list))
for re in api_list:
    f.write(re+'\n')