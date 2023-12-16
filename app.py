import torch
import torch.nn as nn
import torch.optim as optim

########################
# MODELO XOR CON PYTORCH
########################
# Definir los datos de entrada y salida
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# Definir la arquitectura de la red neuronal
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# Inicializar el modelo
model = XORModel()

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Entrenamiento
for epoch in range(100000):
    # Forward Pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward Pass and Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"Loss: {loss.item()}")

# Evaluar el modelo
with torch.no_grad():
    predicted = model(X)
    predicted = (predicted > 0.5).float()
    accuracy = (predicted == y).sum().item() / y.size(0) * 100

print(f"Exactitud del modelo: {accuracy}%")
# Guardar el modelo en un archivo
# torch.save(model.state_dict(), "modelo_xor.pt")
print("Empieza la exportación del modelo en formato ONNX")
import torch.onnx
import torch

# Guardar el modelo en formato .pth
# torch.save(model.state_dict(), "modelo_xor.pth")

# Exportar el modelo a formato ONNX
dummy_input = torch.tensor([[0, 1]], dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    "modelo_xor.onnx",
    verbose=True,
    input_names=["x1", "x2"],
    output_names=["resultado"],
)
print("Modelo exportado correctamente")
