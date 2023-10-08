class Solution():
    def __init__(self, l1: list=[], l2: list=[]):
        self.l1 = l1
        self.l2 = l2

    def addTwoNumbers(self):
        self.l1.reverse()
        self.l2.reverse()
        self.total_l1 = 0
        self.total_l2 = 0
        self.length_l1 = len(self.l1)
        self.length_l2 = len(self.l2)
        for i in self.l1:
            l1_num = i * 10 ** (self.length_l1 - 1)
            self.length_l1 -= 1
            self.total_l1 += l1_num

        for j in self.l2:
            l2_num = j * 10 ** (self.length_l2 - 1)
            self.length_l2 -= 1
            self.total_l2 += l2_num

            add_result = self.total_l1+self.total_l2
        print([add_result])
        return [add_result]


if __name__ == "__main__":
   soulution = Solution(l1=[2, 3, 4], l2=[5, 6, 7])
   soulution.addTwoNumbers()
