import matplotlib.pyplot as plt
import numpy as np
def read_data():#讀取資料
    fp = open('Iris_training.txt', "r")#開啟文件
    lines = fp.readlines()#讀取資料
    data = []
    for i in range(len(lines)):
            temp = np.fromstring(lines[i], sep=',') 
            data = np.append(data,temp)
    fp.close()#關閉文件
    data = np.reshape(data,(int(data.size/3),3))#改為二維
    data = [((x[0],x[1]),x[2]) for x in data]#再分割
    return data
def sign_check(w,b,data):#進行pla
    c = 0 #確定還有沒有錯誤 0//無須再修正 >0//繼續修正
    terminal = 1 #外迴圈判斷值 ==0時結束pla修正
    for x,y in data:
        x = np.array(x)
        if np.sign(np.dot(w,x)+b) != y: #如果 w*x+b 的 sign 與 y不同進行修正
            w = w + (x * y)#修正w
            b = b + y#修正b
            c += 1
    if c == 0:#如果沒錯誤
        terminal = 0 #停止修正
        return w, b, terminal   
    return w, b, terminal
def seperate(data): #分離資料
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for x,y in data:
        x = np.array(x)        
        if y==1: #如果原資料 y值是1的 x1丟入x1 x2丟入y1
            x1 = np.append(x1,x[0])
            y1 = np.append(y1,x[1])
        elif y==-1: #如果原資料 y值是-1的 x1丟入x0 x2丟入y0
            x0 = np.append(x0,x[0])
            y0 = np.append(y0,x[1])
    return x0,x1,y0,y1
def check(w,b): #確認測試準確
    w1 = w[0]
    w2 = w[1]
    fp = open('Iris_test.txt', "r")
    lines = fp.readlines()
    data = []
    for i in range(len(lines)):
            temp = np.fromstring(lines[i], sep=',') 
            data = np.append(data,temp)
    fp.close()
    data = np.reshape(data,(int(data.size/3),3))
    correct = 0
    cx = []
    cy = []
    wx = []
    wy = []#以上都是資料整理
    for x in data:
        if np.sign(w1 * x[0] + w2 * x[1] + b) == x[2]: # 如果 w1 * x1 + w2 * x2 + b == 資料給的y及代表正確
            correct += 1 #計算正確筆數
            cx = np.append(cx,x[0])
            cy = np.append(cy,x[1])
        else:
            wx = np.append(wx,x[0])
            wy = np.append(wy,x[1])
    return cx,cy,wx,wy,correct

def main():
    data = read_data()#讀取資料
    w = np.array((0.1,0.2))#臆測剛開始初始值為w1 = 0.1 w2 = 0.2 b =0.3
    b = 0.3
    terminal = 1
    while(terminal==1):#進行pla修正w和b
        w, b, terminal=sign_check(w,b,data)
    x0,x1,y0,y1 =seperate(data) #將+1和-1的資料分離
    cx,cy,wx,wy,correct = check(w,b) #將測試正確和錯誤分離
    correct = int(correct * 10)
    print('w1 = '+str(round(w[0],1)))
    print('w2 = '+str(round(w[1],1)))
    print('b = '+str(round(b,1)))
    print('準確率 = '+str(correct)+' %')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l = np.linspace(4,8)#x軸從4-8
    ax1.plot(l,(l*w[0]+b)/w[1]*-1)#繪製線性
    plt.scatter(cx,cy,c="blue",marker ='*',label="Testing Correct ("+str(correct)+"%)") #正確測試資料
    plt.scatter(wx,wy,c="red",marker ='x',label='Testing Wrong') #錯誤測試資料
    plt.scatter(x0,y0,c="black",marker ='^',label='Non_Setosa') #Non_Setosa蘭花
    plt.scatter(x1,y1,c="green",marker ='.',label='Setosa') #Setosa蘭花
    plt.legend(loc='lower right')#圖例顯示右下角
    plt.show()
main()