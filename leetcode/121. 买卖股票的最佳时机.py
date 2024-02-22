class Solution:
    def maxProfit(self, prices: list):
        buy = 0
        max_profit = 0
        for now in range(len(prices)):
            if prices[now] <= prices[buy]: # 买的价格大于当天股票价格，亏本！那说明当天的价格低
                buy = now # 重置买股票的天数，那么选择当前天为新的买入日期！
            else:
                max_profit = max(prices[now] - prices[buy], max_profit)
        print(max_profit)


'''class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 1:
            return 0
        else:
            difference_list = []
            for i in range(1, len(prices) + 1):
                for j in range(i + 1, len(prices) + 1):
                    difference = prices[j - 1] - prices[i - 1]
                    difference_list.append(difference)
                    max_profit = max(difference_list)
            if max_profit <= 0:
                return 0
            else:
                return max_profit'''


solution = Solution()
solution.maxProfit([7, 1, 5, 3, 6, 4])
