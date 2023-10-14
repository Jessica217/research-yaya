import time
import datetime
import pytz

class Clock(object): # 基类Clock

    def __init__(self, hour: int = 0, minute: int = 0, second: int = 0 ):  # init方法用于初始化类的实例
        self.hour = hour  # 实例属性
        self.minute = minute
        self.second = second

    def run(self):
        # self.second + 1  错误的，这样根本没给原先属性+1 应该是x = x + 1 或者x += 1 这种形式
        self.second += 1
        # second = self.second 这句话对吗 有啥用？
        # 这句话没有意义（在这个例子） 如果有意义的话 是用一个second作为临时变量复制一份self.second的值

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

class AdvanceClock(Clock): # 创建Clock的子类AdvanceClock,新增加两个属性，timezone和city_name

    def __init__(self, hour, minute, second, timezone:dict = {}, city_name:str = "", timezone_num:str = ""): # 在为子类添加属性时，必须要写父类的属性
        super().__init__(hour, minute, second) # 调用父类的构造方法
        self.timezone = timezone
        self.city_name = city_name
        self.timezone_num = timezone_num

    def get_time(self): # 新增类内方法
        self.timezone_num = datetime.datetime.now() # 当前时实时间
        #self.city_name = input("输入时区的城市名:")

        if self.city_name == "Beijing":
            print("输出时区{}".format(self.city_name),"时间为{}".format(self.timezone_num.strftime("%Y-%m-%d %H:%M:%S")))

        elif self.city_name == "NewYork":
            newyork = pytz.timezone('US/Eastern')
            current_time_ny = self.timezone_num.astimezone(newyork)
            print("输入时区{}".format(self.city_name),"的时间为{}".format(current_time_ny.strftime("%Y-%m-%d %H:%M:%S")))

        elif self.city_name == "London":
            london = pytz.timezone('Europe/London')
            current_time_london = self.timezone_num.astimezone(london)
            print("输入时区{}".format(self.city_name), "的时间为{}".format(current_time_london.strftime("%Y-%m-%d %H:%M:%S")))

if __name__ == "__main__":  # 此方法用于确定当前脚本是被直接运行，还是被作为模块导入到其他脚本中
    # 如果在当前脚本中，会被直接运行 但如果作为模块导入到其他脚本中，则不会被执行，此方法很适合用于代码的重用和模块化
    # Clock = Clock(23, 59, 58)
    # 使用while true进入无限循环 一直打印输出
    while True:
        # 创建子类AdvanceClock的实例
        clock_time = AdvanceClock(hour=23,minute=58,second=59,
                                  timezone={"Beijing":"{timezone_num}", "NewYork":"{timezone_num}", "London":"{timezone_num}"},
                                  city_name = input("请输入时区的城市名:"))

        clock_time.run()
        time.sleep(1)
        clock_time.get_time()


# 示例如上，原先分数80!