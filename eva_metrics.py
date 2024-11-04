from scipy.stats import wilcoxon

def wilcoxon_test(data1, data2):
    """
    Wilcoxon signed-rank test
    """
    stat, p = wilcoxon(data1, data2)
    return stat, p

data1 = [1, 2, 3, 4, 5]
data2 = [2, 3, 4, 5, 6]
data3 = [1.5, 2.2, 3, 4, 5]

stat, p = wilcoxon_test(data1, data2)
stat1, p1 = wilcoxon_test(data1, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))    
print('Statistics=%.3f, p=%.3f' % (stat1, p1)) 