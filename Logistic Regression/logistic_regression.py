import numpy as np

def error(n):                                                           #返回activation function後的值
    return 1 / (1 + np.exp(-n))
def test(w,b):                                                          #test data
    ans = []                                                            #答案
    ans.append("ans")                                                   #加入column name
    test=np.loadtxt("test.csv",skiprows=1,dtype=np.int,delimiter=',')   #讀取data
    for x in test:
        x = np.array(x)                                                 # x = 一筆測資
        n = np.dot(w,x)+b                                               #將data 代入
        ym = error(n)                                                   #利用error()返回值計算y帽
        if ym >= 0.5:                                                   #如果y帽>=0.5 資料預測為5
            ans.append("5")
        if ym < 0.5:                                                    #如果y帽<0.5 資料預測為2
            ans.append("2")
    ans = np.asarray(ans)
    np.savetxt("test_ans.csv", ans,fmt='%s', delimiter=",")             #產生答案的csv

def main():                                                             #主程式
    train=np.loadtxt("train.csv",skiprows=1,dtype=np.int,delimiter=',') #讀入訓練資料
    train_y=train[:,0]                                                  #將第一行的y值取出
    train=np.delete(train,0,axis=1)                                     #刪除第一行
    train_size = np.size(train,1)                                       #取得x的數量n
    w = np.random.rand(train_size)                                      #產生n個隨機的w
    lrate = 0.5                                                         #learning rate初始值為0.5
    b = np.random.rand()                                                #隨機的bias
    mse = 0.01                                                          #設定容忍誤差
    max_epoch = 1500                                                    #最大世代數
    epoch = 0                                                           #最終世代數
    data = []                                                           #空資料

    #tidy data
    for i, x in enumerate(train):                                       #i為index ,x為資料y
        if train_y[i] == 5:                                             
            temp_label = 1                                              #如果為5 ,y=1
        else:
            temp_label = 0                                              #如果為2 ,y=0
        data.append((x,temp_label))                                     #加入data陣列中
    data = np.array(data)

    for i in range(max_epoch):                                              #當在最大世代範圍內
        total_mse = 0                                                   #計算mse
        for x,y in data:                                                #Generate the estimated value for a single example                                   
            x = np.array(x)
            n = np.dot(w,x)+b
            ym = error(n)
            w = w - lrate * ( ym - y) * x
            b = b - lrate * ( ym - y)
            total_mse = total_mse + (y-ym)**2                       
        lrate *= 0.95                                                   #每一世代後,learning rate減少
        total_mse /= train_size                                         #每一世代最終的mse值
        if(total_mse<=mse):                                             #如果低於容忍值則停止迴圈
            epoch = i
            break
    correct = 0                                                         #計算準確次數
    for x,y in data:
        x = np.array(x)
        n = np.dot(w,x)+b
        ym = error(n)
        if ym >= 0.5 and y==1:                                          #y帽>=0.5 又是1的類別代表為正確結果
            correct += 1
        if ym < 0.5 and y==0:                                           #y帽<0.5 又是0的類別代表為正確結果
            correct +=1
    rate = correct/ np.size(train,0)                                    #計算準確率
    test(w,b)                                                           #測試資料並猜測
    print(f"weight:{w} epoch: {epoch}, learning rate:{lrate},correct rate:{rate}")

main()                                                                  #執行主程式