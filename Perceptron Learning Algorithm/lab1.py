import matplotlib.pyplot as plt
import numpy as np

data = np.array([ #初始10筆資料
    ((1,0),1),
    ((1,3),-1),
    ((2,-6),1),
    ((-1,-3),1),
    ((-5,5),-1),
    ((5,2),1),
    ((-2,2),-1),
    ((-7,2),-1),
    ((4,-4),1),
    ((-5,-1),-1), 
],dtype=object)
test_data = np.array([ #測試資料
    (2,-4),
    (-5,1),
    (-2,-2),
],dtype=int)
def main():
    w = np.array((0.1,0.2))#臆測剛開始初始值為w1 = 0.1 w2 = 0.2 b =0.3
    b = 0.3
    terminal = 1
    while(terminal==1):#進行pla修正w和b
        w, b, terminal=sign_check(w,b,data)
    x0,x1,y0,y1 =seperate(data) #將+1和-1的資料分離
    textx0,textx1,texty0,texty1 =test(test_data,w,b)#將測試資料依照+1和-1分離
    print('w1 = '+str(round(w[0],1)))
    print('w2 = '+str(round(w[1],1)))
    print('b = '+str(round(b,1)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    l = np.linspace(-10,10)#x軸從-10~10
    ax1.plot(l,(l*w[0]+b)/w[1]*-1)#繪製線性
    plt.scatter(textx0,texty0,c="red",marker ='x',label='-1(Testing)')#測試資料y為-1的點
    plt.scatter(textx1,texty1,c="green",marker ='P',label='1(Testing)')#測試資料y為1的點
    plt.scatter(x0,y0,c="red",marker ='^',label='-1(Training)')#訓練資料y為-1的點
    plt.scatter(x1,y1,c="green",marker ='.',label='1(Training)')#訓練資料y為+1的點
    plt.legend(loc='lower right')#圖例顯示右下角
    plt.show()

def seperate(data):
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for x,y in data:
        x = np.array(x)        
        if y==1:#如果原資料 y值是1的 x1丟入x1 x2丟入y1
            x1 = np.append(x1,x[0])
            y1 = np.append(y1,x[1])
        elif y==-1:#如果原資料 y值是-1的 x1丟入x0 x2丟入y0
            x0 = np.append(x0,x[0])
            y0 = np.append(y0,x[1])
    return x0,x1,y0,y1
def test(data,w,b):#將測試資料依照+1和-1分離
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for x in data:
        x = np.array(x)
        y = int(np.sign(np.dot(w,x)+b)) #計算test資料的y值
        if y ==1:
            x1 = np.append(x1,x[0])
            y1 = np.append(y1,x[1])
        elif y==-1:
            x0 = np.append(x0,x[0])
            y0 = np.append(y0,x[1])
    return x0,x1,y0,y1
def sign_check(w,b,data):#進行pla
    c = 0#確定還有沒有錯誤 0//無須再修正 >0//繼續修正
    terminal = 1#外迴圈判斷值 ==0時結束pla修正
    for x,y in data:
        x = np.array(x)
        if np.sign(np.dot(w,x)+b) != y:#如果 w*x+b 的 sign 與 y不同進行修正
            w = w + (x * y)#修正w
            b = b + y#修正b
            c += 1
    if c == 0:#如果沒錯誤
        terminal = 0#停止修正
        return w, b, terminal   
    return w, b, terminal
main()