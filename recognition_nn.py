

#エラーレベル
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import recognition
import tensorflow.compat.v1 as tf # type: ignore

# リスト 8-2-(1)
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
# リスト 8-2-(6)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import time
# from tf.keras.preprocessing import image
# from tf.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

global modelfile
# modelfile = 'number5model.h5'

from PIL import Image
# from keras.optimizers import SGD


print(tf.__version__)


class NN():
# 学習のための準備
    def init1(self):
        # mnistデータのロード
        global x_train, y_train, x_test, y_test
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # 訓練データの前処理
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_train = x_train.astype("float32")
        x_train = x_train / 255
        # print(y_train)
        y_train = to_categorical(y_train, num_classes=10)
        # print(y_train)
        # テストデータの前処理
        x_test = x_test.reshape(10000, 28, 28, 1)
        x_test = x_test.astype("float32")
        x_test = x_test / 255
        y_test = to_categorical(y_test, num_classes=10)

        # フィルターの準備 ----------
        myfil1 = np.array([[ 1,  1,  1],                # (A)
                        [ 1,  1,  1],
                        [-2, -2, -2]], dtype=float)
        myfil2 = np.array([[-2,  1,  1],                # (B) 
                        [-2,  1,  1],
                        [-2,  1,  1]], dtype=float)

        # 入力画像の準備 ----------
        id_img = 2  # 使用する画像のインデックス
        x_img = x_train[id_img, :, :, 0]
        img_h = 28
        img_w = 28
        x_img = x_img.reshape(img_h, img_w)  # 入力画像
        out_img1 = np.zeros_like(x_img)  # myfil1の出力用の行列を準備
        out_img2 = np.zeros_like(x_img)  # myfil2の出力用の行列を準備

        # フィルター処理 ----------
        for ih in range(img_h - 3 + 1):
            for iw in range(img_w - 3 + 1):
                img_part = x_img[ih : ih + 3, iw : iw + 3]
                out_img1[ih + 1, iw + 1] \
                    = img_part.reshape(-1) @ myfil1.reshape(-1)
                out_img2[ih + 1, iw + 1] \
                    = img_part.reshape(-1) @ myfil2.reshape(-1)


    #aのモデル(入力層、中間層１６、出力層１０)
    def modelload_a(self):
        # self.modelfile = 'number5model_a.h5'
        self.modelfile = 'number5model_a.keras'
        is_file = os.path.isfile(self.modelfile)
        if(is_file):
            model = load_model(self.modelfile)
            print(self.modelfile, "ファイルを読み込みました。")
            self.model = model
            return model
        
        print("--最初のモデル--")
        # モデルの定義 ----------
        model = Sequential()                                             # (B)
        # model.add(Dense(units=16, input_dim=784, activation="sigmoid"))  # (C)
        model.add(Dense(units=16, input_shape=(28, 28, 1), activation="sigmoid"))  # (C)
        # model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu",))
        model.add(Flatten())
        model.add(Dense(units=10, activation="softmax"))                 # (D)
        model.compile(                                                   # (E)
            loss="categorical_crossentropy", optimizer="adam",
            metrics=["accuracy"],
        )
        self.model = model
        return model


    #最終のモデル（入力層→中間層（畳み込み）１６→中間層（畳み込み）３２→プーリング→
    #→中間層６４→プーリング→ドロップアウト→中間層１２８→出力層１０）
    def modelload_b(self):
        # self.modelfile = 'number5model_b.h5'
        self.modelfile = 'number5model_b.keras'
        is_file = os.path.isfile(self.modelfile)
        if(is_file):
            model = load_model(self.modelfile)
            print(self.modelfile, "ファイルを読み込みました。")
            self.model = model
            return model

        # モデルの定義 ----------
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu",))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))         # プーリング層
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))         # プーリング層
        model.add(Dropout(rate=0.25))                     # ドロップアウト層
        model.add(Flatten())
        model.add(Dense(units=128, activation="relu"))
        model.add(Dropout(rate=0.25))                     # (D) ドロップアウト層
        model.add(Dense(units=10, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam", metrics=["accuracy"],
        )
        self.model = model
        return model


    #最終のモデル
    def modelload_c(self):
        self.modelfile = 'number5model_c.h5'
        is_file = os.path.isfile(self.modelfile)
        if(is_file):
            model = load_model(self.modelfile)
            print(self.modelfile, "ファイルを読み込みました。")
            self.model = model
            return model

        # モデルの定義 ----------
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu",))
        model.add(Flatten())
        model.add(Dense(units=10, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam", metrics=["accuracy"],
        )
        self.model = model
        return model




    ###======================================================================


    def predict_proc(self):
        data = self.get_digital_number_from_img()
        print("self.modelfile="+self.modelfile)
        # model = load_model(self.modelfile)

        print("predict_procデータサイズ:", data.size)
        # data = data.reshape(1,28,28)
        # y = self.model.predict(data[:1, :])
        data = data.reshape(1,28,28,1)
        y = self.model.predict(data[:1, :])
        print(y)

        max = np.argmax(y)
        print("max:",max)
        # self.label["text"] = max

        strmax = '{}かもしれない・・・'.format(max)
        print(strmax)
        return strmax
        # self.label["text"] = strmax
        # self.nowpredit = False


    def predict_all(self):
        # model = load_model(self.modelfile)
        # テストデータに対する出力を計算 ----------
        n_show = 96
        # (A) yはn_show x 10の行列
        y = self.model.predict(x_test[:n_show, :])

        # 結果の描画 ----------
        plt.figure(figsize=(12, 8))
        for i in range(n_show):
            x = x_test[i, :]
            x = x.reshape(28, 28)
            # y[i, j]のj=0～9にはそれぞれの数字に対する確率が入っている
            # 最も確率が大きい数字を予測した数値とする
            prediction = np.argmax(y[i, :])
            plt.subplot(8, 12, i + 1)
            plt.gray()
            plt.pcolor(1 - x)  # 入力画像の表示（白黒を反転）
            plt.text(22, 25.5, f"{prediction}", fontsize=12)
            if prediction != np.argmax(y_test[i, :]):
                plt.plot(  # 間違っていた場合の青い線の表示
                    [0, 27], [1, 1], "cornflowerblue", linewidth=5)
            plt.xlim(0, 27)
            plt.ylim(27, 0)
            plt.xticks([], "")  # x軸の目盛りを消す
            plt.yticks([], "")  # y軸の目盛りを消す
        plt.show()



    #イメージファイルから数値のデータを取り出して表示する
    def show_digital_number_from_img(self):
        data = self.get_digital_number_from_img()
        self.show_digital_number2(data)


    def get_digital_number_from_img(self):
        # img = image.load_img('test1.png', grayscale=True, color_mode='grayscale', target_size=(28, 28))
        img = image.load_img('test1.png', color_mode='grayscale', target_size=(28, 28))
        img = image.img_to_array(img)#画像をそのままarray配列に
        data = np.array([img])

        print("データサイズ:", data.size)
        # print(data)
        data = data.reshape(1,784)#配列を1x784の一列に
        # print(data)
        data = 255 - data#0～255のデータなので反転させる。
        # print(data)
        data = data.astype("float32")#小数点で表示できるように型を修正
        # print(data)
        data = data /255#0～255のデータを0～1のデータに修正(圧縮)
        # print(data)
        data = data.reshape(28,28)
        # print("=====data=====")
        # print(data)
        return data

    def predict_proc2(self):
        data = self.get_digital_number_from_img()
        print("self.modelfile="+self.modelfile)
        # model = load_model(self.modelfile)

        print("predict_procデータサイズ:", data.size)
        # data = data.reshape(1,28,28)
        # y = self.model.predict(data[:1, :])
        data = data.reshape(1, 28,28)
        y = self.model.predict(data)

        # y = model.predict(data[: :1,])
        # y = model.predict(data[0])
        print(y)
        # print("y.type=", type(y))

        max = np.argmax(y)
        print("max:",max)
        # self.label["text"] = max

        strmax = '{}かもしれない・・・'.format(max)
        print(strmax)
        return strmax
        # self.label["text"] = strmax
        # self.nowpredit = False
    

    #デジタルナンバーを出力する
    def show_digital_number(self):
        
        n_show = 1#96
        # plt.figure
        for i in range(n_show):
            x = x_test[i, :]
            x = x.reshape(28, 28)
        # plt.plot()
        plt.figure(figsize=(28,16), dpi=72)
        plt.xlim(0,1)
        plt.ylim(0,1)
        for j in range(28):
            for i in range(28):
                # plt.text(1/28*i, 1-1/28*j, "({},{})".format(i,j) )
                number = str(round(x[i,j], 2))
                plt.text(1/28*i, 1-1/28*j, "{}".format(number) )

        plt.axis('off')
        # plt.text(0, 0, 'Hello, World!')
        plt.show()


    def show_digital_number2(self, data):
        
        data = data.reshape(28, 28)

        # plt.plot()
        plt.figure(figsize=(10,10), dpi=70)
        # plt.figure(figsize=(20,15))
        plt.xlim(0,1)
        plt.ylim(0,1)
        for j in range(28):
            for i in range(28):
                # plt.text(1/28*i, 1-1/28*j, "({},{})".format(i,j) )
                number = str(round(data[i,j], 2))
                if number == "0.0":
                    number = "0"
                plt.text(1/28*j, 1-1/28*i, "{}".format(number) )

        plt.axis('off')
        plt.show()



    def learn_one(self, value):
        print(value, "として登録")
        value = np.array([int(value)])
        value = value.reshape(1,1)
        value = to_categorical(value, num_classes=10)
        data = self.get_digital_number_from_img()
        data = data.reshape(1,28,28)
        # print("self.modelfile="+self.modelfile)
        # model = load_model(self.modelfile)
        # 学習 ----------
        start_time = time.time()

        from keras import optimizers
        # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # sgd = optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # self.model.compile(loss='mean_squared_error', optimizer=sgd)

        learning_rate = 0.001
        momentum = 0.9
        nesterov = True
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


        history = self.model.fit(
            data, value,
            batch_size=1, epochs=1, verbose=1,
            # validation_data=(data, value),
        )
        score = self.model.evaluate(x_test, y_test, verbose=0)
        calculation_time = time.time() - start_time

        # 結果表示 ----------
        print(f"Test loss: {score[0]:.4f}")
        print(f"Test accuracy: {score[1]:.4f}")
        print(f"Calculation time:{calculation_time:.2f} sec")


    def learn(self):
        # 学習 ----------
        start_time = time.time()
        history = self.model.fit(
            x_train, y_train,
            batch_size=1000, epochs=10, verbose=1,
            validation_data=(x_test, y_test),
        )
        score = self.model.evaluate(x_test, y_test, verbose=0)
        calculation_time = time.time() - start_time

        # 結果表示 ----------
        print(f"Test loss: {score[0]:.4f}")
        print(f"Test accuracy: {score[1]:.4f}")
        print(f"Calculation time:{calculation_time:.2f} sec")

        self.model.save(self.modelfile)


 
    def show_figure(self):
        # 結果の描画 ----------
        n_show = 96
        plt.figure(figsize=(12, 8))
        for i in range(n_show):
            x = x_test[i, :]
            x = x.reshape(28, 28)
            # y[i, j]のj=0～9にはそれぞれの数字に対する確率が入っている
            # 最も確率が大きい数字を予測した数値とする
            plt.subplot(8, 12, i + 1)
            plt.gray()
            plt.pcolor(1 - x)  # 入力画像の表示（白黒を反転）
            plt.xlim(0, 27)
            plt.ylim(27, 0)
            plt.xticks([], "")  # x軸の目盛りを消す
            plt.yticks([], "")  # y軸の目盛りを消す

    # リスト 8-2-(4)
        plt.show()




    # リスト 8-1-(7)
    def show_prediction(self):
        model = load_model(self.modelfile)
        # テストデータに対する出力を計算 ----------
        n_show = 96
        # (A) yはn_show x 10の行列
        y = model.predict(x_test[:n_show, :])

        # 結果の描画 ----------
        plt.figure(figsize=(12, 8))
        for i in range(n_show):
            x = x_test[i, :]
            x = x.reshape(28, 28)
            # y[i, j]のj=0～9にはそれぞれの数字に対する確率が入っている
            # 最も確率が大きい数字を予測した数値とする
            prediction = np.argmax(y[i, :])
            plt.subplot(8, 12, i + 1)
            plt.gray()
            plt.pcolor(1 - x)  # 入力画像の表示（白黒を反転）
            plt.text(22, 25.5, f"{prediction}", fontsize=12)
            if prediction != np.argmax(y_test[i, :]):
                plt.plot(  # 間違っていた場合の青い線の表示
                    [0, 27], [1, 1], "cornflowerblue", linewidth=5)
            plt.xlim(0, 27)
            plt.ylim(27, 0)
            plt.xticks([], "")  # x軸の目盛りを消す
            plt.yticks([], "")  # y軸の目盛りを消す

#実行時のメソッド指定
if __name__ == "__main__":
    test.main()