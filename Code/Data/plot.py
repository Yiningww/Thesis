import matplotlib.pyplot as plt
import numpy as np
#todo
start_data = "2016"
end_data = "2020"
data = []
data_beta = []
# Example data
for i in range(12):
    array = []
    array_beta = []
    if i==11:
        file = open("output/{}/{}/baseline.out".format(start_data,end_data)).readlines()
    elif i>=10:
        file = open("output/{}/{}/{}.out".format(start_data,end_data,chr(i+55))).readlines()
    else:
        file = open("output/{}/{}/{}.out".format(start_data,end_data,i)).readlines()
    for line in file:
        if line.startswith("this is adj rsquare"):
            array.append(float(line.split(":")[1]))
        elif line.startswith("this is Beta adj rsquare"):
            print(float(line[:-2].split()[-1]))
            array_beta.append(float(line[:-2].split()[-1]))
    data.append(array)
    data_beta.append(array_beta)
    # print(i, np.median(array))
    print(i, np.median(array_beta))

# Creating the box plot
plt.boxplot(data, showfliers=False)

# Adding titles and labels
plt.title('Box Plot {} to {}'.format(start_data, end_data))
# plt.xticks([1, 2], ['Array1', 'Array2'])
plt.ylabel('Adj R Squared second step')
plt.xlabel('Features')
plt.show()
plt.close()

plt.boxplot(data_beta, showfliers=False)

# Adding titles and labels
plt.title('Box Plot {} to {}'.format(start_data, end_data))
# plt.xticks([1, 2], ['Array1', 'Array2'])
plt.ylabel('Adj R Squared first step')
plt.xlabel('Features')
plt.show()
exit()

# Creating the box plot
plt.boxplot(data, showfliers=False)

# Adding titles and labels
plt.title('Box Plot {} to {}'.format(start_data, end_data))
# plt.xticks([1, 2], ['Array1', 'Array2'])
plt.ylabel('Adj R Squared second step')
plt.xlabel('Features')
# Showing the plot
plt.savefig("output/figure/{}_{}_3M.png".format(start_data,end_data))
