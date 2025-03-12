import pickle
from sklearn.preprocessing import StandardScaler
mlmodel = open('model.pkl', 'rb')
scalermodel = open('scaler.pkl', 'rb')

clfobject = pickle.load(mlmodel)
scaler = pickle.load(scalermodel)

test_values = scaler.fit_transform([[4.9,196.0],[5.1,128.0],[7.0,64.0]])
print(test_values)

result = clfobject.predict(test_values)

print(result)
