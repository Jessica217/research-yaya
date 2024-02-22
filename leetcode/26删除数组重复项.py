class Solution:
    def removeDuplicates(self, nums):
        for i in nums:
            for j in nums[1:]:
                if j == i:
                    nums.remove(j)
                print(nums)

solution = Solution()
solution.removeDuplicates([0,0,1,1,1,2,2,3,3,4])