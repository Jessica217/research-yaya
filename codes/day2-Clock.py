import time

class Clock:
    def __init__(self,hour,minute,second): # init方法用于初始化类的实例
        self.hour = hour # 实例属性
        self.minute = minute
        self.second = second

    def run(self):
        self.second + 1 # second默认加1

        if self.second == 59:
            self.minute + 1
            self.second = 0
        else:
            second = self.second + 1

    def show(self):
        print("{}:".format(self.hour),"{}:".format(self.minute),"{}".format(self.second))

if __name__ =="__main__":
    Clock = Clock(23,59,58)

    while Clock.hour > 1000:
        time.sleep(1)

    Clock.run()
    Clock.show()