from urllib.request import Request,urlopen
import re
from bs4 import BeautifulSoup
import requests

class Crawler():
    def __init__(self, info_dict: dict = {}, url : str = ''):
        self.url = url
        self.info_dict = info_dict
        self.movie_name_list = []
        self.movie_eng_name_list = []
        self.movie_time_list = []
        self.movie_score_list = []

    # 请求网页,获取网页内容
    def obtain_url_info(self):
        request = requests.get(self.url)
        html = request.content.decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')

        # 获取电影中英名称
        movie_titles = soup.find_all('h2', class_='m-b-sm')
        for title in movie_titles:
            movie_name = title.text.split('-')[0] # movie_name为list
            movie_eng_name = title.text.split('-')[1]
            self.movie_name_list.append(movie_name)
            self.movie_eng_name_list.append(movie_eng_name)

        # 获取电影时长
        info_div = soup.find_all('div', class_='m-v-sm info')
        for info in info_div:
            movie_info = info.text # 获取文本数据
            pattern1 = r"(\d+) 分钟" # 使用正则表达式查找”分钟“前的数字
            movie_times = re.search(pattern1, movie_info)
            if movie_times:
                movie_time = movie_times.group(1)
                self.movie_time_list.append(movie_time)

        # 获取电影评分
        score_info = soup.find_all('p',class_='score m-t-md m-b-n-sm')
        for score in score_info:
            movie_score = score.text
            pattern2 = r'\d+.\d+'
            movie_scores = re.findall(pattern2, movie_score)
            result = ','.join(movie_scores) # 将列表转换为字符串
            self.movie_score_list.append(result)


        # 精髓

        for i in range(len(self.movie_name_list)): # 每个列表的长度一样
            movie_name = self.movie_name_list[i]
            movie_eng_name = self.movie_eng_name_list[i]
            movie_time = self.movie_time_list[i]
            movie_score = self.movie_score_list[i]
            self.info_dict[i] = {'movie_name': movie_name, 'movie_eng_name': movie_eng_name, 'movie_time': movie_time,'movie_score': movie_score}

        print(str(self.info_dict))

        with open('result.txt', 'a', encoding='UTF-8') as f:
            f.write(str(self.info_dict))


if __name__ == '__main__':
    # 搜索所有相关的网页

    for i in range(1, 11):
        url = 'https://ssr1.scrape.center/page/{}'.format(i)
        crawler = Crawler(url = url)
        crawler.obtain_url_info()
        #print(crawler.obtain_url_info())






