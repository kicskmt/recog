
import tkinter as tk
import tkinter.ttk as ttk
import recognition_nn as NN
from PIL import Image
import random
import os

class Application(tk.Frame):
    #初期メソッド
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        master.geometry("400x800")
        master.title("キャンバス")
        self.master = master

        # カレントディレクトリを移動する
        print(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        #設定情報
        self.width = tk.Scale(master, from_ = 1, to = 50, orient = tk.HORIZONTAL) 
        self.width.set(30)

        # ＝＝＝キャンバスの作成＝＝＝
        

        # ＝＝＝ラベルの作成＝＝＝
        


        self.mas = master
        self.nn = NN.NN()
        self.nn.init1()
        #＝＝＝ニューラルネットの切り替え＝＝＝
        self.nn.modelload_a()
        # self.nn.modelload_b()

        #＝＝＝以下ボタン作成＝＝＝
        #クリアボタン
        
        #データ見るボタン
        
        #予測ボタン
        
        #予測(複数)ボタン
        
        #コンボボックス（１つ学習）
        
        #学習ボタン
        

        # self.canvas.create_oval(100,100,200,200, fill = "black", width = 0)


    #マウスドラッグ時のイベント
    def on_draw(self, event):
        self.sx = event.x
        self.sy = event.y
        han = self.width.get()/2
        #＝＝＝色設定＝＝＝
        color = ["#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])]
        
        self.canvas.create_oval(event.x-han, event.y-han, event.x+han, event.y+han, outline= color, fill = color)


    #=================================
    #    ここからボタンの処理
    #=================================

    #クリアボタン
    def clear_btn(self):
        print("clear_btn")
        self.canvas.delete("all")
        self.label["text"] = ""
        self.combobox.set('')

    #データを表示するボタン
    def showdata_btn(self):
        print("showdata_btn")
        self.save_img()
        self.nn.show_digital_number_from_img()

    #予測ボタン
    def predict_btn(self):
        print("predict_btn")
        self.save_img()
        #todo 予測メソッド
        ret = self.nn.predict_proc()
        # strmax = '{}かもしれない・・・'.format(ret)
        self.label["text"] = ret

    #１つ学習ボタン コンボボックスが選ばれたときの処理
    def onSelected(self, event):
        print("onSelected")
        value = event.widget.get()
        print(value,"が選択されました")
        self.save_img()#画像として保存
        self.nn.learn_one(value)
        self.clear_btn()#クリア処理
        str = '{}として登録されました。'.format(value)
        self.label["text"] = str

    #予測（複数）ボタン
    def predict_all_btn(self):
        print("predict_all_btn")
        self.nn.predict_all()

    #学習ボタン
    def learn_btn(self):
        print("learn_btn")
        self.nn.learn()

    #ボタンを作成するメソッド
    def create_btn(self, txt, method):
        button = tk.Button(self.mas, text = txt, font=("MSゴシック", "20", "bold")
                           ,command = method, width = 10)
        button.pack()
        return button

    #１つ学習するコンボボックス
    def create_combobox(self):
        option = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"] 
        combobox = ttk.Combobox ( self.mas ,  font = ( "MSゴシック" , 20 , "bold" ), 
                                      values = option, width = 10)
        combobox.bind("<<ComboboxSelected>>", self.onSelected)
        combobox.option_add("*TCombobox*Listbox.Font", ("MSゴシック", 20))
        combobox.pack()
        self.combobox = combobox
        return combobox

    #キャンバス作成
    def create_canvas(self):
        self.canvas = tk.Canvas(self.master, bg = "white", width = 300, height = 300)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_draw)
        self.canvas.bind("<B1-Motion>", self.on_draw)
        return self.canvas

    #ラベル作成
    def create_label(self):
        self.label = tk.Label(self.master, text="ラベル", font=("MSゴシック", "20", "bold")
                        , relief=tk.SUNKEN, width = 20)
        self.label.pack()
        return self.label
    
    #画面をイメージファイルに保存
    def save_img(self):
        self.canvas.postscript(file = 'test1' + '.eps') 
        img = Image.open('test1' + '.eps') 
        img.save('test1' + '.png', 'png') 

#メイン処理
def main():
    win = tk.Tk()
    app = Application(master=win)
    app.mainloop()

#実行時のメソッド指定
if __name__ == "__main__":
    main()