from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np

app = Flask(__name__)

# Cargar el modelo ONNX
ort_session = ort.InferenceSession("modelo_xor.onnx")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return "Bienvenido a la página de inicio. Puedes hacer una solicitud POST a esta misma dirección para realizar inferencias XOR."
    elif request.method == "POST":
        try:
            data = request.get_json()
            x1 = data["x1"]
            x2 = data["x2"]
            input_data = np.array([[x1, x2]], dtype=np.float32)
            ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            ort_outs = ort_session.run(None, ort_inputs)
            resultado = (ort_outs[0] > 0.5).item()
            return jsonify({"resultado": resultado})
        except Exception as e:
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run()
