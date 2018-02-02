from PERM_API_By_PERM import API_BY_PERM
import linecache
import re
import os
class fileoperation:
 def extract(self):
    lines=linecache.getlines('system-current.txt')
    if(os.path.exists('raw_api_result.txt')):
        os.remove('raw_api_result.txt')
    list=[]
    dict={}
    package=''
    round=1;
    for line in lines:
        if(line.__contains__('{')):
          str = line.split('{')[0]
          if re.search('package',str) and re.search('package',str).span()[0]==0:
                if(round!=1):

                    f=open('raw_api_result.txt','a')
                    for a in list:
                        file_in=package+'/'+a+'\n'
                        f.write(file_in)
                list=[]
                package = str
                round=round+1
                print 'package',str
          else:
                list.append(str)
 def process(self):
     lines=linecache.getlines('raw_api_result.txt')
     f=open('api_result.txt','w')
     class_name=''
     for line in lines:
         package_name=line.split('/')[0].split(' ')[1]
         package_name=package_name.replace('.','/')
         # print package_name,'packagename'
         if line.__contains__('class'):
             class_name=line.split('/')[1].split('class')[1].split(' ')[1]
             class_name=class_name.replace('.','&')
             # print class_name,'class_name'
         elif line.__contains__('interface'):
             class_name=line.split('/')[1].split('interface')[1].split(' ')[1]
             class_name=class_name.replace('.', '&')
             #print class_name, 'interface_name'
         else:
             pass
         str="L"+package_name+'/'+class_name
         if(str.__contains__('<')):
             str=str.split('<')[0]
         print str,'~~~'
         f.write(str+'\n')

 def union_two_api(self):
     class_list = []
     filename = 'E:/JetBrains/PycharmProjects/project1/test1/result.txt'
     f = open('E:/JetBrains/PycharmProjects/project1/test1/result.txt', 'w')
     for value in API_BY_PERM.values():
         for api in value:
             api_class = api.split(';')[0]
             print api_class
             class_list.append(api_class)
     api_list = list(set(class_list))
     for re in api_list:
         f.write(re + '\n')
     api_list_godeye=linecache.getlines('result.txt')
     api_list_result=linecache.getlines('api_result.txt')
     api_list_result.extend(api_list_godeye)
     api_list=list(set(api_list_result))
     api_list.sort(key=api_list_result.index)
     f=open('api_list.txt','w')
     no=4
     for api in  api_list:
         print api,no
         file_in_str=api.strip('\n')+'\t'+str(no)+'\n'
         f.write(file_in_str)
         no=no+1


if __name__ == '__main__':
    operator=fileoperation()
    operator.extract()
    operator.process()
    operator.union_two_api()