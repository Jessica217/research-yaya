class BaseFood:

    # 是类的魔法方法，也就是构造函数，它总是在类的实例化中最先执行
    # 它支持参数传入，因此可以通过这样的方式来初始化内置的参数
    def __init__(self, primary_materials:list = [],zuoliao:list = [],cook_function:list = []):
        self.primary_materials = primary_materials
        self.zuoliao = zuoliao
        self.cook_function = cook_function
        self.wash_vegetables()# 洗菜 切菜 装盘是默认的操作，所以写在基类中
        self.cut_vegetables()
        self.finish_dish()

    def get(self):
        print("原材料是:{}".format(self.primary_materials))
        print("佐料是:{}".format(self.zuoliao))
        print("烹饪方法是:{}".format(self.cook_function))
    def wash_vegetables(self):
        print("已经洗好了:{}".format(self.primary_materials))
    def cut_vegetables(self):
        print("已经切好了:{}".format(self.primary_materials))
    def cook_vegetables(self):
        print("已经炒好了:{}".format(self.primary_materials))
    def finish_dish(self):
        print("装盘:{}".format(self.primary_materials))

# 类的继承 class 新类(基类) 可以用基类中的方法，也可以自己新加
class potato_with_beef(BaseFood):
    def dun_beef(self):
        print("要炖:{}".format(self.primary_materials))

# python魔法方法的一种，当在文件中声明时，文件默认执行其中的内容
# 当在别的文件引用此文件时，默认不执行、
# 所以常用于模块的debug和测试
if __name__ == "__main__":
    # 实例化了一个basefood对象
    # basefood = BaseFood(primary_materials=['potato','tomato'],zuoliao=['salt','pepper'],cook_function=['chao','jian'])
    new_dish = potato_with_beef(primary_materials=['potato','beef'],zuoliao=['香料','胡椒'])
    new_dish.dun_beef()


