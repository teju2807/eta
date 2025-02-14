import pickle
import pandas as pd

def predict(data):
    model = pickle.load(open('models/model.pkl', 'rb'))
    input_df = pd.DataFrame(data)
    predictions = model.predict(input_df)
    return predictions.tolist()