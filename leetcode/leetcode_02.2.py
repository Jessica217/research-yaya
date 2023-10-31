class Node(): # 定义结点
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
    node3 = Node(3) # 创建结点3

    singlelinklist.head = node1 # 将node1添加到链表中
    node1.next = node2
    node2.next = node3

    print(type(singlelinklist.head.item))
    print(singlelinklist.head.next.item)


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
