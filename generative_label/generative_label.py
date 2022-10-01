import jieba
import string
from zhon.hanzi import punctuation
from common import save_jsonl
import csv

class generative_label:
    def __init__(self, text) -> None:
        super().__init__()
        self.text = text
        self.token = []
        self.label_list = []

    def word_split(self):
        token = ' '.join(jieba.cut(self.text, cut_all=True, HMM=True))
        token_tmp = ''
        for num, i in enumerate(token):
            if '\u4e00' <= i <= '\u9fff':
                token_tmp += i + ' '
            else:
                token_tmp += i  
        for i in punctuation + string.punctuation:
            token_tmp = token_tmp.replace(i, "")
        # print(token_tmp)
        self.token = [x for x in token_tmp.split(" ") if x != '']
        return self.token

    def display(self):
        step = 10
        list_num = [i + 1 for i in range(10 * (len(self.token) // 10 + 1))]
        for i in list_num:
            print("%5d"%(i - 1), end = ' ')
            if i % (step) == 0:
                print()
                print(self.token[i - 10:i])
                print()

    def assign_number(self):
        self.label_list = []
        while True:
            start_value = input("輸入空白退出: ")
            if(start_value == " "):
                break
            self.label_list.append(start_value)
            print("the value of start word is : %s "%(start_value))
            # star = eval(input("開始的數字:"))
            end_value = input("the value of end word is :")
            # print(end_value)

            self.label_list.append(end_value)

        print('label_list = ', self.label_list) 

    def create_output_list(self, email_id, cls_label):
        global output_list
        output = {}
        output_label = [0] * len(self.text)

        list_tmp = [self.label_list[i:i+2] for i in range(0, len(self.label_list), 2)]
        for i in list_tmp:
            for j in range (int(i[0]), int(i[1])):
                output_label[j] = 1


        output['email_id'] = email_id
        output['text'] = self.token
        output['gen_label'] = output_label
        output['cls_label'] = cls_label


        output_list.append(output)
        print('output_list : ', output_list)
        save_jsonl(output_list, './test_output.jsonl')


        

if __name__ == '__main__':

    global output_list
    output_list = []

    with open("spam_data_test.csv", newline='', encoding = 'utf-8') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
            email_id = row[0]
            cls_label = row[2]
            text = row[3]

            gen_label = generative_label(text)
            print(gen_label.word_split())
            gen_label.display()
            gen_label.assign_number()
            gen_label.create_output_list(email_id = email_id, cls_label = cls_label)

            

    # reader = csv.reader(text)
    # print(reader)
    # email_id = reader.
    # cls_label = reader[2]
    # text = reader[3]
    # print(text)

    # text= "自 然 語 言 處 理  (NLP)  是 人? 工 智 慧 的 分 支 領 域，使用機器學習技術來處理及解讀文字和資料。自然語言辨識和自然語言產生均為 NLP 的類型。"
    # gen_label = generative_label(text)
    # print(gen_label.word_split())
    # gen_label.display()
    # gen_label.assign_number()
    # gen_label.create_output_list(email_id = 1, cls_label = 1)
    # gen_label.output()


    # print(' '.join(jieba.cut(text, cut_all=True, HMM=True)))