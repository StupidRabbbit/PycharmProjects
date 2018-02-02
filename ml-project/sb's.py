# coding=utf-8
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

filename='C:\Users\Administrator\Desktop\\2017digest1.csv'
df_sb=pd.read_csv(filename,encoding='utf-8')
print df_sb.shape

df_meeting=df_sb['meeting']
print df_meeting

sb_dict={}
i=0
for item in df_meeting:
   if item not in sb_dict.keys():
        list=[]
        sb_dict[item]=list
   sb_dict[item].append(i)
   i=i+1
print sb_dict.keys()

A_Journal=['TIFS']
B_Journal=['Computers & Security','Journal of Computer Security']
C_Journal=['EURASIP Journal on Information Security','IET','IJISP','SCN']
A_Meeting=['S&P','USENIX Security']
B_Meeting=['ESORICS','NDSS','CSFW','DSN']
C_Meeting=['WiSec','Symposium On Usable Privacy and Security','secureComm','USENIX Workshop on Hot Topics in Security','Wisec','DIMVA'
           ,'ASIACCS','ACM MM&SEC']

sb_list=[A_Journal,B_Journal,C_Journal,A_Meeting,B_Meeting,C_Meeting]
newfile='C:\Users\Administrator\Desktop\\2017.xlsx'
writer = pd.ExcelWriter(newfile)
ii=0
for sb in sb_list:
    ii=ii+1
    sb_index=[]
    for ssb in sb:
        temp=sb_dict[ssb]
        sb_index=sb_index+temp
    print sb_index
    df=pd.DataFrame(df_sb.loc[sb_index] )
    print df.shape
    df.to_excel(writer,str(ii))

writer.save()


