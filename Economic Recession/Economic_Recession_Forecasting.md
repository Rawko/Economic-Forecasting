```Python
#Forecasting Ecpnomic Ression Probabilities Using State Space Models and Monte Carlo Simulations

import numpy as np 
import pandas as pd
import statsmodels.api as sm 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn 



# Loading the DataFrame
df = pd.read_csv('~/Desktop/e_f_p/Projects/Project-4/gdp_recession_II.csv')

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
df.rename(columns={'JHGDPBRINDX': 'Percentage_Points'}, inplace=True)

print(df.head())



#Plotting the time series
df['Percentage_Points'].plot(figsize=(10, 6), title='GDP Percentage Points Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Percentage Points')
plt.show()

#plotting the autocorrelation plot (Yule-Walker)
plot_acf(df['Percentage_Points'])
plt.title('ACF of GDP Percentage Points')
plt.show()

#Plotting the partical autocorrelation plot (Yule-Walker)
plot_pacf(df['Percentage_Points'])
plt.title('PACF of GDP Percentage Points')
plt.show()
