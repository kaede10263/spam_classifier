import tkinter as tk
import matplotlib.pyplot as plt
import tkinter.ttk as ttk
import tkinter.font as tkFont
from tkinter import * 
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from utils import function

class Button:
    def __init__(self,page_one,Button_name,Button_text,command,x,y):
        self.page_one = page_one
        self.Button_name = Button_name
        self.Button_text = Button_text
        self.command = command
        self.x = x
        self.y = y
         
    def button_show(self):
        self.button_name = tk.Button(self.page_one,text= self.Button_text
                                    ,width=12
                                    ,height=2   
                                    ,bg='#1cb1e3'
                                    ,activebackground='#38CDFB'
                                    ,relief= 'raised'
                                    ,font=('Arial',20,'bold')
                                    ,fg='#FFFFFF'
                                    ,command=self.command)
        self.button_name.place(x=self.x, y=self.y)

class Label:
    def __init__(self,page,button_name,text_font,grape_detail,x,y):
        self.page = page
        self.button_name = button_name
        self.text_font = text_font
        self.grape_detail = grape_detail
        self.x = x
        self.y = y
         
    def label_show(self):

        self.button_name = tk.Label(self.page, relief="sunken", height=10, width=14, font=self.text_font)#顯示苦腐病、晚腐病、正常 統計
        self.button_name.config(text = self.grape_detail)#代入內容文字
        self.button_name.place(x= self.x,y= self.y)#統計位置

        
class user_interface_app:
    def __init__(self,root,page_one):
        self.root = root
        self.page_one = page_one

        self.list_font = tkFont.Font(family="Times New Roman CE"
                                , size=14
                                , weight="bold"
                                , slant="italic")#字體設定格式
        self.text_font = tkFont.Font(family="Lucida Grande", size=21)#字體設定格式
        self.small_text = tkFont.Font(family="Lucida Grande", size=15)#字體設定格式
    def user_interface_show(self):
        
        """
        前置
        """        
        self.email_heards = tk.Listbox(self.page_one,height=7, width=33, font=self.list_font)
        self.img_viewer = ScrolledText(self.page_one,height=22,width=32,font=self.list_font)#顯示照片
        self.system_info = ScrolledText(self.page_one,height=20,width=42,font=self.list_font)#顯示照片

        self.running = tk.Listbox(self.page_one,height=9, width=54, font=self.small_text)



        self.function  = function (self.page_one,self.root,self.email_heards,self.img_viewer,self.system_info,self.running)
        
        self.grape_detail = """"""#內容文字

        text = ""
        self.single_result = Label(self.page_one,"single_result",self.text_font,text,900,10)
        self.single_result.label_show()
        """
        第一頁物件
        """

        self.email_heards.place(x=10,y=10)#Listbox位置
        
        self.img_viewer.place(x=10,y=200)#顯示照片位置
        self.img_viewer.image_filenames = [] #存資料夾名稱
        self.img_viewer.images = [] #存資料夾照片
        

        
        self.system_info.place(x=400,y=10)


        self.running.place(x=400,y=500)#Listbox位置
        


        self.upload = Button(self.page_one,"upload","上傳",self.function.open_email,900,300)#按鈕
        self.upload.button_show()#顯示按鈕
        
        self.start = Button(self.page_one,"start","開始",self.function.start,900,400)#按鈕
        self.start.button_show()#顯示按鈕

        self.stop = Button(self.page_one,"stop","停止",self.function.stop,900,500)#按鈕
        self.stop.button_show()#顯示按鈕

        self.view_result = Button(self.page_one,"view_result","查看結果",self.function.start,900,600)#按鈕
        self.view_result.button_show()#顯示按鈕
        

