nums = [2,3,4]
target = 6
def add():
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] + nums[j] == target:
                print(i)
                print(j)
                return i,j
add()