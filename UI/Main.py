import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from UI import user_interface_app


class Main_App:
    def __init__(self,root):
         # 視窗大小
        self.root = root
        self.root.geometry('1200x780+180+10')#介面大小
        self.root.resizable(False,False)#不可調整大小
        # 主頁面分類
        self.tabControl = ttk.Notebook(root)
        self.notebook = ttk.Notebook()#分頁設定
        self.notebook.place(relx=0.02, rely=0.02, relwidth=0.95, relheight=0.95)#分頁設定
        
        self.page_one = tk.Frame(self.notebook, bg='#2082df')#分頁1
        self.page_one.place()
        self.notebook.add(self.page_one,text='釣魚郵件檢測')#分頁1名稱
 

        
        app = user_interface_app(self.root,self.page_one)
        app.user_interface_show()

"""
前置設定
"""         

if __name__ == "__main__":
        
    root = tk.Tk()
    root.title("釣魚郵件檢測")    
    app = Main_App(root)

    root.mainloop()

    
            
            