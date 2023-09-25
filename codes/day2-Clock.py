import time


class Clock(object):

    def __init__(self, hour: int = 0, minute: int = 0, second: int = 0):  # init方法用于初始化类的实例
        self.hour = hour  # 实例属性
        self.minute = minute
        self.second = second

    def run(self):
        # self.second + 1  错误的，这样根本没给原先属性+1 应该是x = x + 1 或者x += 1 这种形式
        self.second += 1
        # second = self.second 这句话对吗 有啥用？

        # 正确的进位逻辑
        if self.second == 60:
            self.minute += 1
            self.second = 0
            if self.minute == 60:
                self.hour += 1
                self.minute = 0
                if self.hour == 24:
                    self.hour = 0

    def show(self):
        # print("{}:".format(self.hour), "{}:".format(self.minute), "{}".format(self.second)) 这样写太笨重
        print("%.2d:%.2d:%.2d"%(self.hour, self.minute, self.second)) # %.2d是将一个整数格式化为两位数的字符串，.2是精确到小数点后两位

if __name__ == "__main__": # 此方法用于确定当前脚本是被直接运行，还是被作为模块导入到其他脚本中
    # 如果在当前脚本中，会被直接运行 但如果作为模块导入到其他脚本中，则不会被执行，此方法很适合用于代码的重用和模块化
    Clock = Clock(23, 59, 58)

    # 使用while true进入无限循环 一直打印输出
    while True:
        Clock.run()
        time.sleep(1)
        Clock.show()

# 示例如上，原先分数60，现在理解多了，能不能加上20分QWQ