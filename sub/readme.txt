

ghostscriptのインストール
設定ー＞path->システム環境変数の
PathにC:\Program Files\gs\gs10.03.0\bin
を追加

python -m venv env
Set-ExecutionPolicy RemoteSigned -Scope Process
env\Scripts\Activate.ps1
pip install -r requirements.txt





python -m pip install flask  
python -m pip uninstall flask  
pip install tensorflow
pip install matplotlib 
pip install opencv-python　　（cv2）

python -m pip uninstall tensorflow
python -m pip install tensorflow==2.10.0
python -m pip uninstall tensorflow
python -m pip install tensorflow==2.15.0

python -m pip install tensorflow==2.5.3 だめnumpyのバージョン合わず

pip install --upgrade tensorflow-probability



PS C:\Users\ko\python\testing_test_20240322\testing_test> python --version
Python 3.9.13
PS C:\Users\ko\python\testing_test_20240322\testing_test> python -m pip list
Package                      Version
---------------------------- -------------------
absl-py                      2.1.0
astunparse                   1.6.3
blinker                      1.8.0
cachetools                   5.3.3
certifi                      2024.2.2
charset-normalizer           3.3.2
click                        8.1.7
cloudpickle                  3.0.0
colorama                     0.4.6
contourpy                    1.2.1
cycler                       0.12.1
decorator                    5.1.1
dm-tree                      0.1.8
Flask                        3.0.3
flatbuffers                  24.3.25
fonttools                    4.51.0
gast                         0.4.0
google-auth                  2.29.0
google-auth-oauthlib         1.2.0
google-pasta                 0.2.0
grpcio                       1.62.2
h5py                         3.11.0
idna                         3.7
importlib_metadata           7.1.0
importlib_resources          6.4.0
itsdangerous                 2.2.0
Jinja2                       3.1.3
keras                        2.15.0
keras-nightly                2.5.0.dev2021032900
Keras-Preprocessing          1.1.2
kiwisolver                   1.4.5
libclang                     18.1.1
Markdown                     3.6
markdown-it-py               3.0.0
MarkupSafe                   2.1.5
matplotlib                   3.8.4
mdurl                        0.1.2
ml-dtypes                    0.3.2
namex                        0.0.8
numpy                        1.26.4
oauthlib                     3.2.2
opencv-python                4.9.0.80
opt-einsum                   3.3.0
optree                       0.11.0
packaging                    24.0
pillow                       10.3.0
pip                          22.0.4
protobuf                     4.25.3
pyasn1                       0.6.0
pyasn1_modules               0.4.0
Pygments                     2.17.2
pyparsing                    3.1.2
python-dateutil              2.9.0.post0
requests                     2.31.0
requests-oauthlib            2.0.0
rich                         13.7.1
rsa                          4.9
setuptools                   58.1.0
six                          1.15.0
tensorboard                  2.15.2
tensorboard-data-server      0.7.2
tensorboard-plugin-wit       1.8.1
tensorflow                   2.15.1
tensorflow-estimator         2.15.0
tensorflow-intel             2.15.1
tensorflow-io-gcs-filesystem 0.31.0
tensorflow-probability       0.24.0
termcolor                    1.1.0
typing_extensions            4.11.0
urllib3                      2.2.1
Werkzeug                     3.0.2
wheel                        0.43.0
wrapt                        1.12.1
zipp                         3.18.1



pip install tensorflow
pip install Pillow
pip install matplotlib


pip freeze > requirements.txt
pip install -r requirements.txt
