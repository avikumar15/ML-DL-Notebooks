import numpy as np 
import csv
import os
import math
from sklearn.linear_model import LinearRegression
import pandas as pd


here = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(here, 'test_scores.csv')

def predict_using_sklean():
    df = pd.read_csv(filepath)
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0

    n = len(x)

    alpha = 0.0002

    iterations = 1000000

    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr 

        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])

        md = (-2/n)*sum(x*(y-y_predicted))
        bd = (-2/n)*sum(y-y_predicted)

        m_curr = m_curr - alpha * md 
        b_curr = b_curr - alpha * bd 

        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break

        cost_previous = cost

        print("m is {} b is {} cost is {} iteration is {}".format(m_curr, b_curr, cost, i))
    
extractedData = dict()

extractedData['name'] = []
extractedData['math'] = []
extractedData['cs'] = []

with open(filepath, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    for row in csvreader:
        extractedData['name'] += [row[0]]
        extractedData['math'] += [int(row[1])]
        extractedData['cs'] += [int(row[2])]

print()
print(extractedData['name'])
print(extractedData['math'])
print(extractedData['cs'])
print()

x = np.array(extractedData['math'])
y = np.array(extractedData['cs'])

gradient_descent(x,y)

print('Predicting from the library: ')

m_sklearn, b_sklearn = predict_using_sklean()
print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))
