'''class Node(): # 定义结点
    def __init__(self, item):
        self.item = item # item存储数据
        self.next = None # next为指针


class SingleLinkList():
    def __init__(self):
        self.head = None # 头指针为空


if __name__ == "__main__":
    singlelinklist = SingleLinkList() # 创建空链表
    node1 = Node(1) # 创建结点1
    node2 = Node(2) # 创建结点2
    node3 = Node(3)

    singlelinklist.head = node1 # 将node1添加到链表中
    node1.next = node2
    node2.next = node3

    print(singlelinklist.head.item)
    print(singlelinklist.head.next.item)'''

'''题目描述：
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
'''
class ListNode():
    def __int__(self, val = 0, next = None):
        self.val = val
        self.next = next

class Solution():
    # l1 和 l2 为当前遍历的节点，carry 为进位
    def addTwoNumbers(self, l1: [ListNode], l2: [ListNode], carry=0) -> [ListNode]:
        if l1 is None and l2 is None:  # 递归边界：l1 和 l2 都是空节点
            return ListNode(carry) if carry else None  # 如果进位了，就额外创建一个节点
        if l1 is None:  # 如果 l1 是空的，那么此时 l2 一定不是空节点
            l1, l2 = l2, l1  # 交换 l1 与 l2，保证 l1 非空，从而简化代码
        carry += l1.val + (l2.val if l2 else 0)  # 节点值和进位加在一起
        l1.val = carry % 10  # 每个节点保存一个数位

        l1.next = self.addTwoNumbers(l1.next, l2.next if l2 else None, carry // 10)  # 进位
        return l1


solution = Solution()
solution.addTwoNumbers(1,2)





