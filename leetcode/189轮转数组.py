class Solution:
    def rotate(self, nums: list, k: int):
       ''' if k < len(nums):
            new_nums = nums[len(nums)-k: len(nums)]
            for i in nums[0: len(nums)-k]:
                new_nums.append(i)
        else:
            while

        m = len(new_nums)
        nums[:m] = new_nums
        print(nums)'''

       if k > 0:
            k = k % len(nums)
            nums[:] = nums[-k:] + nums[:-k] # nums[-k:]末位数字




solution = Solution()
solution.rotate([1, 2], 5)

nums = [1, 2, 3]
print(nums[-2:])

