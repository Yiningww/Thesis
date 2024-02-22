import matplotlib.pyplot as plt
#todo
start_data = "2015-01-31"
end_data = "2015-12-31"
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

# Creating the box plot
plt.boxplot(data)

# Adding titles and labels
plt.title('Box Plot {} to {}'.format(start_data, end_data))
# plt.xticks([1, 2], ['Array1', 'Array2'])
plt.ylabel('Adj R Squared')
plt.xlabel('Features')


# Showing the plot
plt.savefig("output/figure/{}_{}.png".format(start_data,end_data))
