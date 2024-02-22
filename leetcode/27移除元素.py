class Solution:
    def removeElement(self, nums, val):
        while val in nums:
            nums.remove(val)
        return  len(nums),nums


solution = Solution()
solution.removeElement([3,2,2,3], 3)
