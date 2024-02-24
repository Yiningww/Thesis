import matplotlib.pyplot as plt
import numpy as np
#todo
start_data = "2013"
end_data = "2017"
data = []
# Example data
for i in range(10):
    array = []
    if i==9:
        file = open("output/{}/{}/baseline.out".format(start_data,end_data)).readlines()
    else:
        file = open("output/{}/{}/{}.out".format(start_data,end_data,i)).readlines()
    for line in file:
        if line.startswith("this is adj rsquare"):
            array.append(float(line.split(":")[1]))
    data.append(array)
    print(i, np.median(array))

# Creating the box plot
plt.boxplot(data, showfliers=False)

# Adding titles and labels
plt.title('Box Plot {} to {}'.format(start_data, end_data))
# plt.xticks([1, 2], ['Array1', 'Array2'])
plt.ylabel('Adj R Squared')
plt.xlabel('Features')
plt.show()
exit()
# Creating the box plot
plt.boxplot(data, showfliers=False)

# Adding titles and labels
plt.title('Box Plot {} to {}'.format(start_data, end_data))
# plt.xticks([1, 2], ['Array1', 'Array2'])
plt.ylabel('Adj R Squared')
plt.xlabel('Features')
# Showing the plot
plt.savefig("output/figure/{}_{}.png".format(start_data,end_data))
