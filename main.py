import os.path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Pobranie danych (zakładamy, że dane są w pliku)
file_path = 'YFData/TSLA_historical_data.csv'

data = pd.read_csv(file_path, sep=';')
stock_name = os.path.basename(file_path).replace('_historical_data.csv','')

# 2. Konwersja danych na odpowiednie kolumny
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# 3. Pobieramy kolumnę "Close"
prices = data['Close'].values
prices = prices.reshape(-1, 1)

# 4. Normalizacja danych
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# 5. Przygotowanie danych do LSTM
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_prices, time_step)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float().view(-1, 1)

X = X.view(X.shape[0], time_step, 1)

# 6. Definicja modelu LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 7. Inicjalizacja modelu, funkcji kosztu i optymalizatora
model = LSTM(input_size=1, hidden_size=50, output_size=1, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. Trening modelu
epochs = 100
for epoch in range(epochs):
    model.train()
    outputs = model(X)
    optimizer.zero_grad()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 9. Prognozowanie na podstawie danych treningowych
model.eval()
predicted = model(X).detach().numpy()
predicted = scaler.inverse_transform(predicted)

# 10. Wizualizacja wyników
plt.figure(figsize=(14, 7))
plt.plot(prices, label='Prawdziwe ceny')
plt.plot(np.arange(time_step, len(predicted) + time_step), predicted, label='Prognozowane ceny')
plt.xlabel('Indeks')
plt.ylabel('Cena')
plt.title(f'Prognozowanie cen akcji {stock_name}')
plt.legend()
plt.show()

# 11. Prognozowanie przyszłych cen
def predict_future(model, data, scaler, time_step, future_days=30):
    inputs = data[-time_step:]
    predicted_prices = []
    for _ in range(future_days):
        inputs_tensor = torch.from_numpy(inputs).float().view(1, time_step, 1)
        predicted_price = model(inputs_tensor).detach().numpy()
        predicted_price_original_scale = scaler.inverse_transform(predicted_price)
        predicted_prices.append(predicted_price_original_scale[0, 0])
        inputs = np.append(inputs[1:], predicted_price, axis=0)
    return predicted_prices

future_days = 30
predicted_future_prices = predict_future(model, scaled_prices, scaler, time_step, future_days)

# 12. Wyświetlamy prognozy przyszłych cen
plt.figure(figsize=(14, 7))
plt.plot(range(len(prices)), prices, label='Prawdziwe ceny')
plt.plot(range(len(prices), len(prices) + future_days), predicted_future_prices, label='Prognozowane przyszłe ceny')
plt.xlabel('Indeks')
plt.ylabel('Cena')
plt.title(f'Prognozowanie przyszłych cen akcji {stock_name}')
plt.legend()
plt.show()

# 13. Generowanie dodatkowego wykresu dla ostatnich 6 miesięcy i prognozy na przyszły miesiąc
months_in_days = 30 * 6
recent_prices = prices[-months_in_days:]

# Prognozowanie ceny na następny miesiąc
predicted_next_month_price = predicted_future_prices[0]
last_price = recent_prices[-1]

# Obliczanie procentowej zmiany
percentage_change = ((predicted_next_month_price - last_price) / last_price) * 100
percentage_change = percentage_change.item()

# Kolor linii wzrostu (czerwony w tym przypadku)
line_color = 'red'

# Generowanie wykresu
plt.figure(figsize=(14, 7))
plt.plot(range(len(recent_prices)), recent_prices, label='Ostatnie 6 miesięcy - Prawdziwe ceny', color='blue')
plt.plot(range(len(recent_prices), len(recent_prices) + future_days), predicted_future_prices, label='Prognozowane ceny - Następny miesiąc', color=line_color)

# Dodanie informacji o procentowej zmianie na wykresie
plt.text(len(recent_prices) + future_days - 1, predicted_future_prices[-1],
         f'Prognozowana zmiana: {percentage_change:.1f}%',
         horizontalalignment='right',
         verticalalignment='bottom',
         fontsize=12, color=line_color, weight='bold')

plt.xlabel('Indeks')
plt.ylabel('Cena')
plt.title(f'Ostatnie 6 miesięcy cen akcji oraz prognoza na przyszły miesiąc {stock_name}')
plt.legend()
plt.show()
