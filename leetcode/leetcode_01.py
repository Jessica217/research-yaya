
class Solution:
    def __init__(self,nums:list,target:int):
        self.nums = nums
        self.target = target

    def twoSum(self):
        '''n = len(self.nums)
        for i in range(n):
            for j in range(n):
                if self.nums[i] + self.nums[j] == self.target:
                    if i == j:
                        break;
                    print(i,j)
                    return [i,j]'''

        # 使用哈希表进行遍历
        hashtable = dict() # 字典中的元素不能重复
        for i, num in enumerate(self.nums): # enumerate的意义见 image_crop.py,i = 0, 1, 2 num = 3, 2, 4
            if self.target - num in hashtable:# target-num是字典的键
                #print(hashtable)
                print([hashtable[self.target - num], i])
            hashtable[self.nums[i]] = i # 如果差值target - num不在字典中，将当前元素及其下标加入字典



if __name__ == "__main__":
    solution = Solution(nums=[3, 2, 4], target=6)
    solution.twoSum()

