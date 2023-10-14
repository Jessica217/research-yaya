"""
验证输入用户名和QQ号是否有效并给出对应的提示信息

要求：用户名必须由字母、数字或下划线构成且长度在6~20个字符之间，QQ号是5~12的数字且首位不能为0
"""

#正则表达式
import re
def search_QQ():
    name = input("请输入用户名：")
    number = input("请输入QQ号：")

    m1 = re.match(r'^[0-9a-zA-Z]]{6,20}$',name)
    m2 = re.match(r'^[1,9]\d{5,12}$',number)
    if not m1:
        print("请输入正确的用户名：")
    if not m2:
        print("请输入正确的QQ号：")
search_QQ()

