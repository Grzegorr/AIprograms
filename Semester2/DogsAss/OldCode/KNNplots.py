import matplotlib.pyplot as plt

samples_no = [500, 1500, 3000]
samples_no2 = [500, 1500]
accuracy10KNN = [55.8,  51.9, 53.8]
accuracy10GNB = [59.5,  59.5, 59.8]
accuracy5KNN = [55,  54.8]
accuracy5GNB = [58,  57.3, 57.7]


plt.plot(samples_no, accuracy10KNN, label = "Scale 1/10, KNN")   
plt.plot(samples_no, accuracy10GNB, label = "Scale 1/10, GNB" )
plt.plot(samples_no2, accuracy5KNN, label = "Scale 1/5, KNN" )
plt.plot(samples_no, accuracy5GNB, label = "Scale 1/5, GNB")   
plt.xlim(400, 3200)
plt.ylim(51, 60)
plt.ylabel('Percentage of correctly classifier pictures (%)')
plt.xlabel('No. of pictures used to train the classifier')
plt.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=2)
plt.show()

samples_no = [500, 1500, 3000]
samples_no2 = [500, 1500]
time10KNN = [12.34, 95.85 , 334.79]
time10GNB = [0.82, 2.45 , 5.20]
time5KNN = [47.03,  361.05]
time5GNB = [3.15, 9.23, 19.96]


plt.plot(samples_no, time10KNN, label = "Scale 1/10, KNN")   
plt.plot(samples_no, time10GNB, label = "Scale 1/10, GNB" )
plt.plot(samples_no2, time5KNN, label = "Scale 1/5, KNN" )
plt.plot(samples_no, time5GNB, label = "Scale 1/5, GNB")   
plt.xlim(400, 3200)
plt.ylim(0, 370)
plt.ylabel('Time neede to train the clasifiers (seconds)')
plt.xlabel('No. of pictures used to train the classifier')
plt.legend(bbox_to_anchor=(0, 1), loc='lower left', ncol=2)
plt.show()
