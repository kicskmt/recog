
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
# from google.colab import files
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import webbrowser
import json

#参考HP
##https://qiita.com/Soichiro1223/items/f03e1922f1bc5a9d4920
#
# pip install tensorflow==2.7.0
# pip install opencv-python
# pip install matplotlib
# pip install protobuf==3.20.*
# pip install flask

#普通にVisualStudioCodeなどで実行するときはこちらを有効に
os.chdir(os.path.dirname(os.path.abspath(__file__))) # カレントディレクトリを移動する
print(os.path.dirname(os.path.abspath(__file__)))

resize_image_size = 50
# resize_image_size = 96

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

allpath = ('pic_cat\\', 'pic_dog\\')
classes = ["ねこ","いぬ"]
kerasfilename = "dogcat_model.keras"
dict = dict()#保存

def training():

    paths = []  #ファイル名一覧
    imgs = []   #イメージ読み込み
    for i in range(len(allpath)):
        print(allpath[i])
        paths.append(os.listdir(allpath[i]))
        for j in range(len(paths[i])):
            print(j, paths[i][j], allpath[i]+paths[i][j])
            img = cv2.imread(allpath[i] + paths[i][j], 1)
            img = cv2.resize(img, (resize_image_size, resize_image_size))
            imgs.append(img)

    X = np.array(imgs)
    y = np.empty(0)
    for i in range(len(allpath)):
        y = np.append(y, np.array([i]*len(paths[i])))
    print(y)


    rand_index = np.random.permutation(np.arange(len(X)))
    X = X[rand_index]
    y = y[rand_index]

    X_train = X[:int(len(X)*1.0)]#0.8
    y_train = y[:int(len(y)*1.0)]#0.8
    X_test = X[int(len(X)*0.8):]
    y_test = y[int(len(y)*0.8):]

    print("ytrain",len(y_train), y_train)
    print("ytest",len(y_test), y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # input_tensor = Input(shape=(50, 50, 3))
    input_tensor = Input(shape=(resize_image_size, resize_image_size, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(2048, activation='relu'))#256
    top_model.add(Dropout(0.2))#もと0.5
    top_model.add(Dense(2048, activation="relu"))#128
    top_model.add(Dense(len(allpath), activation='softmax'))

    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    for layer in model.layers[:19]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(learning_rate=1e-4,  momentum=0.9),metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size = 3, epochs = 10)

    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    pred = np.argmax(model.predict(X_test[0:10]), axis=1)
    print(pred)

    model.summary()
    model.save(kerasfilename)
    print(kerasfilename+"ファイルに書き込みました。")


def testing():

    print("=======================testing=====================")
    paths = []  #ファイル名一覧
    imgs = []   #イメージ読み込み

    # model = load_model(os.path.join(result_dir, 'etcmodel.h5'))
    model = load_model(kerasfilename)

    #ファイルオープン
    openfile = open(r'test.html', 'w')
    openfile.write("""
    <h1>結果表示</h1>
    <p>画像認識の結果です！</p>
    """)

    for i in range(len(allpath)):
        print(allpath[i])
        paths.append(os.listdir(allpath[i]))

        dict.clear()

        for j in range(len(paths[i])):
            img = image.load_img(allpath[i]+paths[i][j], target_size=(resize_image_size, resize_image_size,3))
            img = image.img_to_array(img)
            data = np.array([img])
            result = model.predict(data)[0]
            predicted = result.argmax()
            re =  [ f'{s:.2f}' for s in result]
            print(allpath[i]+paths[i][j] ,classes[predicted], re)
            ree = " ".join([str(_) for _ in re])
            writestr = '<p><img src="%s">%s [%s]</p>'%(allpath[i]+paths[i][j], classes[predicted], ree)
            openfile.write(writestr)

            pred = classes[predicted]

            if pred in dict:
                dict[pred] = dict[pred] + 1
            else:
                dict[pred] = 1
        
        print(dict)
        str_dict = json.dumps(dict)
        writestr = '<p>%s</p><br>'%(str_dict)
        openfile.write(writestr)

    #ファイルのクローズ
    openfile.close()
    print("test.htmlファイルに書き込みました。")

# training()
# testing()




import tkinter as tk
import tkinter.ttk as ttk
# import recognition_nn as NN
from PIL import Image
# import os

class Application(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack()
        master.geometry("400x600")
        master.title("雛形")

        # 普通にVisualStudioCodeなどで実行するときはこちらを有効に
        print(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(os.path.dirname(os.path.abspath(__file__))) # カレントディレクトリを移動する

        # ついか
        self.canvas = tk.Canvas(master, bg = "white", width = 300, height = 300)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_draw)
        self.canvas.bind("<B1-Motion>", self.on_draw)

        #ここについか
        self.label = tk.Label(master, text="", font=("MSゴシック", "20", "bold")
                              , relief=tk.SUNKEN, width = 20)
        self.label.pack()


        self.color = tk.StringVar()
        self.color.set("black")
        self.width = tk.Scale(master, from_ = 1, to = 50, orient = tk.HORIZONTAL) 
        self.width.set(10)
        self.mas = master
        
        # 追加
        learn_button = self.create_btn("学習", training)
        predict_all_button = self.create_btn("複数テスト", testing)
        clear_button = self.create_btn("クリア", self.clear_btn)
        predict_button = self.create_btn("予測", self.predict_btn)

    #クリアボタン
    def clear_btn(self):
        print("clear_btn")
        self.canvas.delete("all")
        self.label["text"] = ""

    #予測ボタン
    def predict_btn(self):
        print("predict_btn")
        #todo 予測メソッド
        retstr = self.predict_process()
        self.label["text"] = retstr

    def on_draw(self, event):
        han = self.width.get()/2
        # 追加
        self.canvas.create_oval(event.x-han, event.y-han, event.x+han, event.y+han, fill = "black")
        
    #画面をイメージファイルに保存
    def save_img(self):
        self.canvas.postscript(file = 'test1' + '.eps') 
        img = Image.open('test1' + '.eps') 
        img.save('test1' + '.png', 'png') 

    #ボタンを作成するメソッド
    def create_btn(self, txt, method):
        button = tk.Button(self.mas, text = txt, font=("MSゴシック", "20", "bold")
                           ,command = method, width = 10)
        button.pack()

    def predict_process(self):
        print("predict_process")
        self.save_img()#イメージ保存
        model = load_model(kerasfilename)#モデル読み込み
        img = image.load_img("test1.png", target_size=(resize_image_size, resize_image_size,3))
        img = image.img_to_array(img)
        data = np.array([img])
        result = model.predict(data)[0]
        # print("type", type(result))#numpy.ndarray
        print("max", result.max())#max
        max = result.max()

        maxstr= "でしょう"

        print(result)
        predicted = result.argmax()
        pred_answer = "これは " + classes[predicted]+maxstr
        print(pred_answer)
        return pred_answer


#メイン処理
def main():
    win = tk.Tk()
    app = Application(master=win)
    app.mainloop()

#実行時のメソッド指定
if __name__ == "__main__":
    main()
