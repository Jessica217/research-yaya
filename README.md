# 项目说明

用于代码训练，记录完整的代码训练内容以及代码，同时更新项目用于发布任务

此外会记录一些相关资料在`./infos`文件夹下

# 命名规范

所有文件命名规范遵从`day{x}-filename.py`的格式，以对应任务的日期编号

涉及多文件命名时，文件夹命名为`day{x}-foldername`，其中的文件依旧按照`day{x}-filename.py`的格式命名

所有与代码任务相关的代码文件均放置于`./codes`文件夹下

# 任务一览

## 2023/9/23 day1

#### 任务目标：熟悉类与对象的概念
文件：

```bash
./codes/day1-BaseFood.py
```

任务内容：

1. 编写基类BaseFood，内置参数为原材料，佐料，内置方法为洗菜，切菜，炒菜，装盘。

2. 在基类BaseFood的基础上编写土豆炒牛肉菜，要求初始化内置参数，并提供烹饪方法，内部调用基类内置方法。

## 2023/9/25 day2

#### 任务目标：面向对象编程基础
文件：

```bash
./codes/day2-Clock.py
```

任务内容:

1. 定义一个数字时钟类

其中包括类内属性`hour: 时`, `minute: 分`, `second: 秒`

类内方法

`run:走字` 即模拟现实时钟的运行规律，调用时second默认+1，同时有进位和置0，比如59秒时执行run，则`minute+1，second=0`

`show:显示时间` 按时:分:秒的形式显示时间，例如`01:09:23`(可能需要了解一下格式输出)

2. 实例化这个类，创建一个Clock对象并初始化为(23,59,58),使用条件循环模拟和`time.sleep(1)`函数模拟实际的时间流逝，执行`Clock.show()`与`Clock.run()`并观察输出

## 2023/9/26 day3

#### 任务目标：面向对象编程基础2
文件：

```bash
./codes/day3-AdvanceClock.py
```

任务内容：

1. 在原先的`Clock`类之上定义它的子类`AdvanceClock`

额外新增类内属性`timezone：时区`, timezone为现实世界中的时区划定标准，其类型为`dict`，记录不同城市的时区信息

默认初始化为`{"Beijing":"{timezone_num}", "NewYork":"{timezone_num}", "London":"{timezone_num}}"`其中timezone_num为真实时区

额外新增类内方法`get_time` 输入时区的城市名，返回指定城市名和其对应时间

2. 实例化这个类，创建一个AdvanceClock对象并初始化为(23,59,58)，使用条件循环模拟和`time.sleep(1)`函数模拟实际的时间流逝

循环执行`AdvanceClock.run()`与`AdvanceClock.get_time()`观察三个不同时区的城市时间输出

## 2023/9/28 day4

#### 任务目标：OS、字符串与文件读写
文件：

```bash
./codes/day4-FileSearch.py
```

任务内容：

声明一个FileSearch类，自定义类内属性与类内方法，但要求要能达成以下功能：

1. 打开指定的文本文件，支持搜索其中的相关内容进行替换
2. 支持将文本文件按照换行切分输出成多个独立文件，名称为源文件名+_{num}，其中num为第几份

实例化这个类，使用随意编写的`.txt`文本文件测试以上功能

## 2023/10/08 day 5

#### 任务目标: 利用request库,re库等爬虫相关库编写一个简单的爬虫

文件：

```bash
./codes/day5-Crawler.py
```

任务内容:

声明一个Crawler类，类内方法必须有 `info_dict` 用于存储爬取的网站信息，其他类内属性与类内方法可自定义，必须能够爬取靶向网站的要求内容。

由于爬虫的特殊性，逻辑多为定制化不可复用，这里给出一个爬虫靶向电影网站：https://ssr1.scrape.center/

要求达到以下功能:

1.爬取靶向网站中的电影名称，评分，英文名，时长等内容，要求存储为dict格式(存储在`info_dict`中)

```python
# 解释以下key值:
# movie_id:自增的id值，由0开始，每个被爬取的电影对应唯一一个id
# movie_name:电影的中文名
# movie_eng_name:电影的英文名
# movie_time:电影的时长，类型为int
# movie_score:电影的评分，类型为float
# 其他相关可爬取的信息也请尽量添加，但不做要求
{"movie_id":{'movie_name':"xxxx", "movie_eng_name":"xxxx", "movie_time":120, "movie_score":9.5},
 "movie_id":{'movie_name':"xxxx", "movie_eng_name":"xxxx", "movie_time":120, "movie_score":9.5}}
```

2.支持将爬取完毕的`info_dict`中的内容存储为`result.txt`文件

注意：原网站中涉及翻页跳转，需要考虑爬取过程中的自动跳转，比如第一页爬取完毕后检查页数进行跳转，直到所有条目爬取完毕.

实例化这个类，对网站中的电影信息进行爬取，并存储为指定文件.