# Classification Problem 
from sklearn.neighbors import KNeighborsClassifier
X = [
    [180 , 7],
    [200 , 7.5],
    [250 , 8],
    [300 , 8.5],
    [330 , 9],
    [360 , 9.5]
    ]

y = [0 , 0 , 0 , 1 , 1 , 1] # 0 = apple , 1 = orange

model = KNeighborsClassifier(n_neighbors=3)
model = model.fit(X , y)

weight = float(input('Enter fruit weight(Gram)'))
size = float(input('Enter fruit size(centimeter)'))

y_predict = model.predict([[weight , size]])[0]

if y_predict == 1:
    print('This is likely an Apple')
else:
    print('This is likely an Orange')    