import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()


import pandas as pd # for data frames 

data = pd.read_csv('train_histroy\losses.csv')

plt.plot(data['lstm2'])
plt.plot(data['gru2'])
plt.plot(data['birnn2'])
plt.title('Bi Layer Model Performance')
plt.xlabel('Num of epochs')
plt.ylabel('Loss')
plt.legend(['lstm2', 'gru2', 'birnn2'], loc='upper right')
plt.savefig('train_histroy/inter_bi.png')