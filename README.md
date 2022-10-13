# spam_classifier

Email是企業網路最易受病毒感染的途徑。而現今市面上，大部分防毒軟體只能攔截已經成功辨識的病毒，這種方式無法在日新月異的攻擊之下順利抵擋第一次的攻擊。本次的解題是將郵件分為SPAM、EDM( Electronic  Direct  Mail，電子報)、HAM( good,  non-spam  email )三類，而分類重點為SPAM的郵件。
我們整理出本題三大難點進行解題
(1)攻擊的信件以不同的語言撰寫
(2) 攻擊的信件以正常內容包裝
(3)攻擊的信件的關鍵詞語

以transformer作為BERT的雙向編碼模型Bert(Bidirectional  Encoder  Representations  from Transformers)，BERT模型讀取整個文本一次就可以利用Self-Attention 的機制如圖二，關注前詞，以及上下文對當前詞的影響，建立Q、K、V三個權重紀錄結果。
![](https://imgur.com/Dfus32m.jpg)
本案結合使用Multilingual classifier 來架構模型，如圖三，此模型是以BERT作為基底架構分類器，將接收到的mail依其主旨及本文的內容進行理解並分類，目標類別為以下三類，分別為0: SPAM、1: EDM、2: HAM。
![](https://imgur.com/He6PAYS.jpg)
模型訓練使用的硬體規格
![](https://imgur.com/u59MYpD.jpg)

本案結合使用Multilingual classifier 來架構模型，如圖三，此模型是以BERT作為基底架構分類器。下圖為模型訓練的evaluation 準確率，訓過程大約花費6.5個小時，訓練後的eval資料集準確率達到99.4%，F1 score為98.9%。
![](https://imgur.com/kn0AhYX.jpg)
## envirment
*   python = 3.8.13
*	cudatoolkit = 11.3.1
*	datasets =2.5.1
*	eml-parser	=	1.17.0
*	jieba	=	0.42.1
*	opencv-python	=	4.6.0.66
*	pandas	=	1.5.0
*	pytorch	=	1.12.0
*	pytorch-mutex	=	1
*	tk	=	8.6.12
*	transformers	=	4.22.2



## install
``` 
pip install datasets
```
``` 
pip install datasets
```




## command use
進入虛擬環境
``` 
conda activate Grape
```
進入桌面
``` 
cd Desktop
```
here is the folders

``` 
C:\Users\user\Desktop>cd 釣魚郵件介面
```
do this command

```
python Main.py
```
GUI介面呈現
![](https://imgur.com/OyhxCsP.jpg)
![](https://imgur.com/6z6zhvY.jpg)
