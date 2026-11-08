import json
import os

from src.data import load_data, prepare_features
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict


def main():

    # Carregar os dados - data.py
    points = load_data("data/raw/points.csv")

    # Preparar features - data.py
    features, scaler = prepare_features(points)

    # Treinar modelo - n_clusters é o ponto de escalabilidade - train.py
    model = train_model(features, n_clusters=5)

    # Avaliar, gera inertia e silhouette_score - evaluate.py
    metrics = evaluate_model(model, features)

    # Predição de exemplo (teste para validação da pipeline)
    result = predict(model, scaler, lat=-23.55, lng=-46.63)

    # Salva output
    output = {"metrics": metrics, "sample_prediction": result}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Pipeline executado com sucesso!")
    print(json.dumps(output, indent=2))



if __name__ == "__main__":
    main()
