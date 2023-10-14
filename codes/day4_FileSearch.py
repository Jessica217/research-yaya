class FileSearch():
    def __init__(self,txt_file):
        self.txt_file = txt_file

    def operation_on_file(self):
        # 读取文件中的相关内容
        with open(self.txt_file, 'r+') as f:
            file = f.read()
            new_file = file.replace("大学","University") # 将大学替换为university
        with open('new_file.txt', 'w') as n: # 将new_file的内容写入新txt文件
            n.write(new_file)

    def new_part_file(self):
        with open('new_file.txt','r') as p:
            new_part = p.readlines() # 使用readlines时，输出的是数组
            #print(new_part)
        for i in range(len(new_part)):
            with open('new_file-{}.txt'.format(i),'w') as f: # 精髓！
                f.write(new_part[i])


if __name__ == "__main__":
    FileSearch('test').operation_on_file()
    FileSearch('test').new_part_file()


