from urllib.request import Request,urlopen
import re
from bs4 import BeautifulSoup
import requests



url = 'https://ssr1.scrape.center/'


class Crawler():
    def __init__(self, info_dict: dict = {}):
        self.info_dict = info_dict

    # 请求网页,获取网页内容
    def obtain_url_info(self, url):
        request = requests.get(url)
        html = request.content.decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        # 获取电影中英名称
        # movie_titles = soup.find_all('h2', class_='m-b-sm')
        movie_titles = [i.find('a').text for i in soup.find_all('h2', class_='m-b-sm')]
        print(movie_titles)
        return html

    # 解析网页内容
    def analyze_info(self, html):


        '''for title in movie_titles:
            movie_name = title.text.split('-') # movie_name为字符串类型
            self.info_dict['movie_name'] = movie_name[0]
            self.info_dict['movie_eng_name'] = movie_name[1]
            print(self.info_dict)

        # 获取电影时长
        info_div = soup.find_all('div', class_='m-v-sm info')
        for info in info_div:
            movie_info = info.text # 获取文本数据
            pattern1 = r"(\d+) 分钟" # 使用正则表达式查找”分钟“前的数字
            movie_times = re.search(pattern1, movie_info)
            if movie_times:
                movie_time = movie_times.group(1)
                self.info_dict['movie_time'] = movie_time

                # 获取电影评分
                score_info = soup.find_all('p',class_='score m-t-md m-b-n-sm')
            for score in score_info:
                movie_score = score.text
                pattern2 = r'\d+.\d+'
                movie_scores = re.findall(pattern2, movie_score)
                result = ','.join(movie_scores) # 将列表转换为字符串
                self.info_dict['movie_score'] = result
            print(self.info_dict)'''


if __name__ == '__main__':
    crawler = Crawler()
    crawler.obtain_url_info(url)
    crawler.analyze_info(html=crawler.obtain_url_info(url))


