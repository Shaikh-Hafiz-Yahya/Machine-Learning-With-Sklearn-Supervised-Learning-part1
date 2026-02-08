from sklearn.tree import DecisionTreeClassifier

# There are two types of features (1.feature1=fruit_size(cm) , 2.feature2 = color_code) 
X = [
    [17.2 , 10],  #Apple
    [15.01 , 11], #Apple
    [11.99 , 5], #Orange
    [12 , 6], #orange
    [10.01 , 7], #orange
    [19.91 , 12] #Apple 
    ]

# target var / dependent var / output
y = [1 , 1 , 0 , 0 , 0 , 1] #Apple = 1 , Orange = 0

model = DecisionTreeClassifier()
model = model.fit(X , y)

fruit_size = float(input('Enter fruit size(cm):'))
fruit_color_code = int(input('Enter color code:'))

actual_input = model.predict([[19.91 , 12]])
y_predict = model.predict([[fruit_size , fruit_color_code]])

if y_predict == 1 and actual_input == 1:
    print('This is Apple')
else:
    print('This is Orange')    


