
#Forecasting Ecpnomic Ression Probabilities Using State Space Models and Monte Carlo Simulations

import numpy as np 
import pandas as pd
import statsmodels.api as sm 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn 
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

# Loading the DataFrame
df = pd.read_csv('~/Desktop/e_f_p/Projects/Project-4/gdp_recession_II.csv')

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
df.rename(columns={'JHGDPBRINDX': 'Percentage_Points'}, inplace=True)



#Plotting the time series
df['Percentage_Points'].plot(figsize=(10, 6), title='GDP Percentage Points Time Series Plot')
plt.xlabel('Year (quarterly frequency)')
plt.ylabel('Percentage Points')
plt.show()



#plotting the autocorrelation plot
plot_acf(df['Percentage_Points'])
plt.xlabel('Quarterly Lags')
plt.ylabel('Correlation Coefficient')
plt.title('ACF of GDP Percentage Points')
plt.show()

#Plotting the partical autocorrelation plot 
plot_pacf(df['Percentage_Points'])
plt.xlabel('Quarterly Lags')
plt.ylabel('Correlation Coefficient')
plt.title('PACF of GDP Percentage Points')
plt.show()


# Decomposition of Time Series 

decomposition = seasonal_decompose(df['Percentage_Points'], model='additive')


# Plot of decomposed components (Time Series, Trend, Seasonal, Residuals)
fig = decomposition.plot()
fig.set_size_inches(10, 8)
fig.suptitle('Time Series Decomposition of GDP Percentage Points')
plt.show()

# General summary statistics
print(df.describe())

# Performing Augmented Dickey-Fuller test to test stationarity 
adf_test = adfuller(df['Percentage_Points'])

print('ADF Statistic: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])
print('Critical Values:')
for key, value in adf_test[4].items():
    print('\t%s: %.3f' % (key, value))

# Interpretation
if adf_test[1] > 0.05:
    print("The time series is not stationary.")
else:
    print("The time series is stationary.")

# Auto-fitting an ARMA model

auto_arima_model = auto_arima(df['Percentage_Points'], start_p=1, start_q=1, 
                              max_p=5, max_q=5, seasonal=True, d=0, trace=True, 
                              error_action= 'ignore', supress_warnings=True, stepwise=True)


print(auto_arima_model.summary())


# Train and Test Data 

# Number of observations
n = 224
num_rows, num_columns = df.shape
print("Number of total observations:", num_rows)

#splitting 80/20
split_index = int(n * 0.8)


# Splitting the DataFrame
train = df.iloc[:split_index]
test = df.iloc[split_index:]

print(f"Training Set: {len(train)} observations")
print(f"Test Set: {len(test)} observations")


#Auto-fitting an SARIMA model on whole dataset

arima_df = auto_arima(df['Percentage_Points'], start_p=1, start_q=1, 
                              max_p=5, max_q=5, seasonal=True, d=0, trace=True, 
                              error_action= 'ignore', supress_warnings=True, stepwise=True)


print(arima_df.summary())


#Auto-fitting an SARIMA model on whole dataset

arima_train = auto_arima(train['Percentage_Points'], start_p=1, start_q=1, 
                              max_p=5, max_q=5, seasonal=True, d=0, trace=True, 
                              error_action= 'ignore', supress_warnings=True, stepwise=True)


print(arima_train.summary())


from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")


# Fitting SARIMA(1,0,1) to training set 
arima1_0_1 = ARIMA(train['Percentage_Points'], order=(1,0,1))
results1_0_1= arima1_0_1.fit()

#Plotting diagnostics 
fig1 = results1_0_1.plot_diagnostics(figsize = (15,12))
fig1.suptitle('SARIMA(1,0,1)', fontsize=16)

#Fitting SARIMA(2,0,2) to training set 
arima2_0_2 = ARIMA(train['Percentage_Points'], order=(2,0,2))
results2_0_2 = arima2_0_2.fit()

#Plotting diagnostics
fig2=results2_0_2.plot_diagnostics(figsize = (15,12))
fig2.suptitle('SARIMA(2,0,2)', fontsize=16)


plt.show()
print(results1_0_1.summary())
print(results2_0_2.summary())



# Forecast length of test data 
arima_forecast = results2_0_2.forecast(steps=len(test))


# Plotting the historical training data
train['Percentage_Points'].plot(label=' Historical (Train) Data', figsize=(12, 6))

# Assuming 'arima_forecast' is a Series or similar with a datetime index
# Plotting the forecasted values
arima_forecast.plot(style='--', label='Forecast')

# Assuming 'test' is your test DataFrame and has the same structure as 'train'
# Plotting the actual test data for comparison
test['Percentage_Points'].plot(label='Actual (Test) Data')

# Adding plot labels and title
plt.xlabel('Time')
plt.ylabel('Percentage Points')
plt.title('ARIMA(2,0,2) Forecast vs Actual Test Data')
plt.legend()
plt.show()




#Fitting and Forecasting with Hidden Markov Chain Model (HMM)

from sklearn.preprocessing import StandardScaler
import hmmlearn 
from hmmlearn import hmm

train_data = train['Percentage_Points'].values.reshape(-1, 1)  # Reshaping for HMM
test_data = test['Percentage_Points'].values.reshape(-1, 1)

# Scaling the data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
#test_scaled = scaler.transform(test_data)
test_scaled = scaler.transform(test['Percentage_Points'].values.reshape(-1, 1))



# Choose the number of hidden states
n_components = 3  

# Create and fit the Gaussian HMM
hmm_model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
hmm_model.fit(train_scaled)

hidden_states_test = hmm_model.predict(test_scaled)


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Visualization
plt.figure(figsize=(15, 8))

# Plotting historical (train) data
plt.plot(train.index, scaler.inverse_transform(train_scaled).flatten(), label='Historical Data', color='magenta')

# Plotting actual test data
plt.plot(test.index, test['Percentage_Points'], label='Actual Test Data', color='black')

# Since plotting states directly as forecasted isn't straightforward, we'll indicate state changes with colors
colors = plt.cm.jet(np.linspace(0, 1, n_components))
for i, state in enumerate(hidden_states_test):
    plt.axvline(test.index[i], color=colors[state], linestyle='--', alpha=0.5)

# Create custom legend handles
custom_lines = [Line2D([0], [0], color='magenta', lw=2, label='Historical Data'),
                Line2D([0], [0], color='black', lw=2, label='Actual Test Data'),
                Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Forecasted (Blue)'),
                Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Forecasted (Orange)'),
                Line2D([0], [0], color='green', lw=2, linestyle='--', label='Forecasted (Green)')]

# You need to extract labels from the custom_lines
labels = [line.get_label() for line in custom_lines]

# Then create a custom legend
plt.legend(custom_lines, labels, loc='best')


# Set title and labels
plt.title('Historical, Test Data vs. HMM State Predictions')
plt.xlabel('Time')
plt.ylabel('Percentage Points')

# Display the plot
plt.show()



plt.figure(figsize=(15, 7))

# Plot historical data
plt.plot(train.index, train['Percentage_Points'], label='Historical (Train) Data', color='magenta')

# Plot actual test data
plt.plot(test.index, test['Percentage_Points'], label='Actual (Test) Data', color='black')

# Overlay the HMM predicted states on the test data
# Each state will be represented by a horizontal line at the mean value of the state
state_means = scaler.inverse_transform(hmm_model.means_)
for i, state in enumerate(np.unique(hidden_states_test)):
    # Extract the part of the test set where this state occurs
    state_mask = hidden_states_test == state
    # Plot horizontal lines at the mean of each state across the portions of the test set where that state occurs
    plt.hlines(state_means[state], xmin=test.index[state_mask].min(), xmax=test.index[state_mask].max(), 
               color='red', linestyles='dashed', label=f'HMM Predicted State {state}' if i == 0 else "", 
               alpha=0.7)

plt.title('Historical, Test Data vs. HMM Predicted States Overlay')
plt.xlabel('Time')
plt.ylabel('Percentage Points')
# Create a custom legend to avoid duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()


