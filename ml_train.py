import pickle
import pandas as pd
from pathlib import Path
import getopt
import traceback
import sys
import os
import time
import mysql.connector
from mysql.connector import Error
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

import signal
import sys

import logging

max_dir = 3 # 存取最近幾次學習數量
train_num = 1000 #每次訓練數量
path = './model/' #model存放預設位置
new_folder = ''

name = ['subject','plain','html']
# db_setting = {
#     'host':'localhost',
#     'database':'my_db',
#     'user':'sharetech',
#     'password':'27050888',
# }

db_setting = {
    'host':'125.227.221.217',
    'database':'my_db',
    'user':'aigo',
    'password':'ZV4KYH9334d0xvNT',
}



#顯示log
class Log:
    def __init__(self, file_name):
        self.logger = logging.getLogger(file_name)  #
        self.logger.setLevel(logging.DEBUG) 
        mode = 'a' if self.logger.handlers else 'w'
        log_path = os.getcwd() + '/'
        logfile = log_path + file_name + '.log'
        fmt = "%(asctime)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(fmt)
        fh = logging.FileHandler(logfile, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.handlers = []
        self.logger.addHandler(fh)

    def info(self, message):
        self.logger.info(message)
    def warning(self,message):
        self.logger.warning(message)
    def error(self,message):
        self.logger.error(message)
    def debug(self,message):
        self.logger.debug(message)
    def critical(self,message):
        self.logger.critical(message)


def usage():
    print("""
        Usage:sys.args[0] [option]
        -h or --help：顯示幫助信息
        -m or --model：訓練模型路徑   例如：ml_train.py -m <dir_name>
        -n or --num：訓練檔案數量 例如：ml_train.py -n <train_num>
    """)

def signal_handler(signum, frame): #取得訊號
    if signum == signal.SIGINT.value or signum == signal.SIGTERM.value or signum == signal.SIGHUP.value :
        log = Log('logfile') #生log檔案 名稱為logfile.log
        log.warning(signum)  #logfile.log裡面印出 signum
        log.info("被強制關掉了") #logfile.log裡面印出字串
        shutil.rmtree(f'{new_folder}') #發生錯誤未完成的訓練檔全部刪除
        remark() # mark回0
    raise KeyboardInterrupt()



def del_folder():    #當訓練的資料夾大於max_dir 刪除最舊的訓練檔
    folder = []

    allfile = os.listdir(path)
    for i in allfile:
        if os.path.isdir(path+i):
            folder.append(i)
    count = len(folder)
    
    while( count > max_dir):   
        old_folder = min(folder)
        folder.remove(old_folder)   
        shutil.rmtree(f'{path}{str(old_folder)}')  #刪除最舊的訓練檔
        count -= 1  #計算資料夾的數量


def create_folder(): #新建資料夾存取訓練檔，並回傳目前最近存取的訓練檔路徑
    folder = []
    max_folder = 0
    allfile = os.listdir(path)
    for i in allfile:
        if os.path.isdir(path+i):
            folder.append(i)
    count = len(folder)

    if count > 0 : #當訓練檔數量大於0再去找最近存取的訓練檔
        max_folder = max(folder)

    filepath = Path(path,'temp') #未訓練完成先取名為temp
    filepath.mkdir(parents=True, exist_ok=True)
    
    return filepath,max_folder


def word_vec(word,n): #信件內容向量化 分成訓練集  測試集

    target = []
    mail_sentence=word['sentence']
    if (walk_ext_file(path,f'tfidf_model_{n}.pkl') == 0):
        tfidf_model = TfidfVectorizer()
        tfidf_vec  = tfidf_model.fit_transform(mail_sentence) #訓練向量化
        pickle.dump(tfidf_model,open(f"./model/tfidf_model_{n}.pkl",'wb')) 
    else:
        tfidf_model = pickle.load(open(f"./model/tfidf_model_{n}.pkl", "rb")) #向量化訓練檔需與前面都相同,向量轉換格式才是相同的，之後才可正常進行model訓練
        tfidf_vec  = tfidf_model.transform(mail_sentence) #將mail 內容轉成向量
    word['label'] = word['label'].map( {'SPAM':2 ,'EDM': 1, 'HAM': 0}) #將label依據類別分成0 1 2  必須先定義好有幾個label，避免之後多一個label訓練時會出錯
    # 當要更動label數量或是類別都必須重新做訓練
    target = word['label']

    X = pd.DataFrame(tfidf_vec.toarray()) 
    Y = target
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=.1, random_state=12) #將資料集拆分為train data 跟 test data
    return  X_train,X_test,y_train,y_test 


# 取得預訓練信件斷詞以及label
def get_word(): 
    
    connection = mysql.connector.connect(**db_setting)
    result = []
    try:
        if connection.is_connected():
            cursor = connection.cursor()
            for i in name:
                
                sql_u = "update machine_learning set mark_"+ i + "= -1 where mark_" + i + " = 0 limit %s" #將mark轉成-1 代表目前要抓取做訓練的data
                cursor.execute(sql_u,(train_num,))
                connection.commit()

                sql = "select w.md5sum,e.label,w."+ i +\
                            " from word_segment w, email_file e, machine_learning m\
                            where w.md5sum = e.md5sum and w.md5sum = m.md5sum and m.mark_"+i+" = -1 "  #取得預訓練集
                cursor.execute(sql)
                sql_data = pd.DataFrame(cursor.fetchall())

                if len(sql_data) != 0:
            
                    sql_data.columns = ['md5sum','label','sentence']
                    sql_data.replace(to_replace=r'[,\[\]\']+', value='', regex=True,inplace=True) #去除掉標點符號

                result.append(sql_data)

    except Error as e:
        traceback.print_exc()
        print(e)
    finally:
        if(connection.is_connected()):
            connection.close()
            cursor.close()
        return result


# 搜尋檔案是否已存在
def walk_ext_file(dir_path,ext):
    # 尋找檔案
    for root, dirs, files in os.walk(dir_path):
        # 獲取檔名稱及路徑
        for file in files:
            file_path = os.path.join(root, file)
            file_name=os.path.basename(file_path)
            value=file_name.find(ext)
            if  value != -1:
                return 1
    return 0

# 搜尋資料夾是否已存在
def walk_ext_dir(dir_path,ext):
    for root, dirs, files in os.walk(dir_path):
        # 獲取檔名稱及路徑
        for dir in dirs:
            file_path = os.path.join(root, dir)
            file_name=os.path.basename(file_path)
            value=file_name.find(ext)
            if  value != -1:
                return 1
    return 0


def remark():  #當訓練時發生錯誤,需要重新做訓練，必須把訓練檔案 mark 回 0
    connection = mysql.connector.connect(**db_setting)
    try:
        if connection.is_connected():
            cursor = connection.cursor()
            for i in name:
                sql = "update machine_learning set  mark_" + i + " = 0 where mark_"+ i + "= -1" #將目前正在訓練的 data mark轉為 0
                cursor.execute(sql)
            connection.commit()
    except Error as e:
        traceback.print_exc()
        print(e)
    finally:
        if(connection.is_connected()):
            connection.close()
            cursor.close()


def mark(current): #標記 訓練完成 訓練的時間
    connection = mysql.connector.connect(**db_setting)
    try:
        if connection.is_connected():
            cursor = connection.cursor()
            for i in name:

                sql = "update machine_learning set  train_time_"+i+"  = \" "+ current +" \", mark_" + i + " = 1  where mark_"+ i + "= -1"
                #訓練時間為目前時間
                cursor.execute(sql)
            connection.commit()
    except Error as e:
        traceback.print_exc()
        print(e)
    finally:
        if(connection.is_connected()):
            connection.close()
            cursor.close()


def retrain(model_time): #重新訓練舊的model，會把新訓練的資料 mark 回 0 表示 重新做訓練
    connection = mysql.connector.connect(**db_setting)
    try:
        for i in name:
            if connection.is_connected():
                cursor = connection.cursor()
                sql = "update machine_learning set  mark_" + i + " = 0,train_time_"+ i +" = 0 where train_time_"+ i +" > \""+ model_time +"\" and train_time_"+ i +" > 0" 
                #在訓練時間在想訓練的model的時間之後 mark 回 0 train time 回 0
                cursor.execute(sql)
            connection.commit()
    except Error as e:
        traceback.print_exc()
        print(e)
    finally:
        if(connection.is_connected()):
            connection.close()
            cursor.close()


def main(argv):
    global new_folder, train_num

    start = time.time()
    
    #接收訊號 刪除temp
    signal.signal(signal.SIGINT, signal_handler) #signum=2    crtl+c 
    signal.signal(signal.SIGTERM, signal_handler) #signum=15  sudo kill pid 但不包括sudo kill -9 pid
    signal.signal(signal.SIGHUP, signal_handler) #signum=1 關掉terminal

    if not os.path.isdir("./model/"): #檢查是否有model資料夾,訓練檔存放位置
        os.mkdir("./model/")

    # 如果temp存在就關閉程式
    if walk_ext_dir(path,'temp') != 0:
        print('\033[1;34m'+'上一次尚未訓練結束',end='\033[0m\n')
        sys.exit()

    

    new_folder,max_folder = create_folder() #創建訓練存放資料夾，並取得 最新 訓練檔資料夾
    
    # 使用指令
    try:
        opts,args = getopt.getopt(argv,"hm:n:",["help","model=","num="])
    except getopt.GetoptError:
        print ('錯誤指令：test_model.py -h  or test_model.py --help 查詢指令')
        shutil.rmtree(new_folder)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h',"--help"):
            usage()
            shutil.rmtree(new_folder)
            sys.exit()
        elif opt in ("-m","--model"):
            # 要選擇前幾次的model去做訓練，把這中間訓練過得資料mark回 0
            max_folder = arg
            retrain(arg)
        elif opt in ("-n","--num"):
            train_num = int(arg)


    max_path = path + str(max_folder) #預訓練model路徑
    clfs = ['SVM','NB','MLP'] #model種類
    email_type = 0 #計算信件內容類別count
    flag = 0 #計算沒有data的訓練集

    
    try:

        data = get_word()
        end = time.time()
        print('資料抓取 花費時間: ',end-start,' s')

        for i in data:
            if len(i) != 0:
                X_train,X_test,y_train,y_test = word_vec(i,name[email_type]) #將文字向量化
                print(name[email_type])
                for clf in clfs:
                    
                    if (max_folder == 0): #沒有訓練紀錄 需要創建訓練model
                        if clf == 'SVM':    #支持向量機
                            model = SGDClassifier(loss='log')
                        elif clf == 'NB':   #樸素貝葉斯分類器
                            model = MultinomialNB() 
                        elif clf == 'MLP': #多层感知器分类器
                            model = MLPClassifier(max_iter = 800,solver='adam', verbose=0, tol=1e-8, random_state=1,
                                        learning_rate_init=.06)
                    else :
                        model = pickle.load(open(f'{max_path}/spam_ham_{clf}_{name[email_type]}.pkl',"rb")) #取得已經存在的model
                    model.partial_fit(X_train,y_train,[0,1,2]) #終身學習 [0,1,2]為信件的分類代號 會依據前面的label數量做定義，繼續訓練已存在model時 label數量必須相等
                    prediction = model.predict(X_test) #預測結果
                    score = f1_score(prediction, y_test,average='micro')
                    print(f'{clf} F1 score is: {round(score,3)}') #計算預測結果 F-score
                    pickle.dump(model, open(f"./{new_folder}/spam_ham_{clf}_{name[email_type]}.pkl","wb"))#存取訓練好的model
                    # print('pkl存取成功！')

            else: #沒有訓練用data 將上一次訓練結果放到目前訓練資料夾
                print('\033[1;35m'+name[email_type]+' 已沒有data做訓練'+'\033[0m')
                flag += 1
                for clf in clfs:
                    if (walk_ext_file(path,f'spam_ham_{clf}_{name[email_type]}.pkl')!=0):
                        source = f'{max_path}/spam_ham_{clf}_{name[email_type]}.pkl'
                        dst = f"./{new_folder}/spam_ham_{clf}_{name[email_type]}.pkl"
                        dest = shutil.copyfile(source, dst)
                    else:
                        print(f'\033[1;31m'+'error: {name[email_type]} model 不存在 \033[0m')
                        shutil.rmtree(f'{new_folder}')
                        return

            email_type+=1 #subject , plain , html 

        if flag !=3 : #有訓練data 完成訓練

            current = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #目前時間
        # 將訓練完成的temp改名為目前時間
            file_newname = os.path.join(path,current )
            os.rename(new_folder, file_newname)
            mark(current) #標記訓練 完成 時間
        else: #三種類型都未有訓練data
            print('\033[1;31m無效的訓練\033[0m')
            shutil.rmtree(f'{new_folder}')

    except:
        print('\033[1;31m'+'error: 存取失敗\n'+'\033[0m')
        shutil.rmtree(f'{new_folder}') #發生錯誤未完成的訓練檔全部刪除
        remark() #發生錯誤 mark回0
        traceback.print_exc()

    del_folder()
    end = time.time()
    print('總花費時間: ',end-start,' s')
        

if __name__ == '__main__':
    # main(sys.argv[1:])
    data = get_word()
    print(data)