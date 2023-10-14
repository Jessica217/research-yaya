import time
from day2_Clock import Clock


class AdvanceClock(Clock):
    def __init__(self, hour: int = 0, minute: int = 0,second: int = 0 , timezone: dict =  {"Beijing":"+8", "NewYork":"-4", "London":"+1"}):
        super(AdvanceClock, self).__init__(hour, minute, second)
        self.timezone = timezone

    def get_time(self,city_name: str):
        time_change = self.timezone[city_name] # 由字典中的key访问value值
        time_change_num = int(time_change) # 将value值整形化
        local_hour = self.hour + time_change_num # 当前时间＋时区差的小时 就是当地时间
        if local_hour > 24: # 如果写成AdvanceClock.run(),则将self.hour送入方法中，而非local_hour
            local_hour = 0
        local_time = "%.2d:%.2d:%.2d"%(local_hour, self.minute, self.second)
        res = "{}的时间是:{}".format(city_name,local_time)
        return res



if __name__ == "__main__":
    advanceclock = AdvanceClock(23,30,58)
    while True:
        advanceclock.run()
        time.sleep(1)
        print(advanceclock.get_time("NewYork"))
