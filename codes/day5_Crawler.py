from urllib.request import Request,urlopen
import re


url = 'https://ssr1.scrape.center/'


class Crawler():
    def __init__(self, info_dict: dict = {}):
        self.info_dict = info_dict

    def obtain_url_info(self):
        request = Request(url)
        request.add_header("user-agent", "Mozilla/5.0") # 模拟Mozilla浏览器进行爬虫
        response = urlopen(request) # 发送请求
        content = response.read() # 读取相应内容
        print(content)


if __name__ == '__main__':
    crawler = Crawler()
    crawler.obtain_url_info()


