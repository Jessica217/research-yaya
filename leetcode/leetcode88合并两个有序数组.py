class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums1 = nums1 + nums2
        value = 0
        while value in nums1:
            nums1.remove(value)
        if m == 0:
            nums1 = nums2
            # print(nums1)
            return nums1
        elif n == 0:
            nums1 = nums1
            # print(nums1)
            return nums1
        else:
            for i in nums1[0:m]:
                for j in nums1[len(nums1) - n:]:
                    if j < i:
                        index2 = nums2.index(j) + m
                        index1 = nums1.index(i)
                        nums1[index1], nums1[index2] = nums1[index2], nums1[index1]
                    else:
                        nums1 = nums1
            # print("最终结果：", nums1)
            return print(nums1)


solution = Solution()
solution.merge([4,5,6,0,0,0], 3, [1, 2, 3], 3)