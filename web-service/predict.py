import mlflow
# import scipy
import xgboost as xgb
from flask import Flask, jsonify, request
from sklearn.feature_extraction import DictVectorizer

model_uri = "./models/models_mlflow"
booster = mlflow.xgboost.load_model(model_uri)


def prepare_features(house):
    features = {}
    features['area_living'] = house['area_living']
    features['area_land'] = house['area_land']
    features['n_rooms'] = house['n_rooms']
    features['year'] = house['year']
    features['price'] = house['price']
    return features


def predict(features):
    dv = DictVectorizer()
    _ = dv.fit_transform(features)

    X_pred = dv.transform(features)
    target = [features["price"]]

    pred = xgb.DMatrix(X_pred, label=target)
    preds = booster.predict(pred)
    return float(preds[0])


app = Flask('price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    house = request.get_json()

    features = prepare_features(house)
    pred = predict(features)

    result = {'price': pred}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
