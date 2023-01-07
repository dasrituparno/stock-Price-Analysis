import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from streamlit_option_menu import option_menu

import pandas
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()


# hide the right side streamlit menue    button
# Referance ~ '''https://towardsdatascience.com/5-ways-to-customise-your-streamlit-ui-e914e458a17c'''
'''st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
'''

#For Condensing the Layout 
#Referance ~ see above ...
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Stock Market Analysis & Prediction',
                          
                          ['Dataset','Statistical Analysis',
                           'Prediction with M.L.'],
                          icons=['plus','plus','plus'],
                          default_index=0)


user_input=st.text_input('Enter Stock Ticker','CIPLA.NS')
data=pdr.get_data_yahoo(user_input,start='2017-04-03',end='2023-01-05')

data1=data.reset_index().copy()
Total_data=data1.iloc[:,4:5].copy() #with this i am going to work
NCP1=Total_data.loc[0:742,:] #NCP1= Non covid period 1(2017-2020)
CP=Total_data.loc[742:988,:] #CP= Covid period(2020-2021)
NCP2=Total_data.loc[990:,:] #NCP=Non Covid period2(2021~present) here present time will vary

if (selected == 'Dataset'):
        

        st.title('Dataset ~')


        #describing the data
        st.subheader('Data from 2017 to 2022')
        st.text("First 5 stock prices acoording the date ~")
        st.write(data.head())
        st.text("last 5 stock prices according to the date ~")
        st.write(data.tail())
        st.write(data.describe())

        #modifying the dataset


        st.text('''Here we have considered (2017.April ~ 2020.March_end) & (2021.April ~ Present)
        Datasetes as Non-Covid Period Datasets & (2020.April ~ 2021.March end ) Dataset as Covid period Dataset. 
        For simplicity purpose here we are considering the "Close" price data as our target variable.''')

        st.text("First Non-Covid Dataset ~")
        st.write(NCP1.describe())
        st.text("Second Non-Covid Dataset ~")
        st.write(NCP2.describe())
        st.text("Covid period Dataset ~")
        st.write(CP.describe())

        #Visualization
        st.subheader('Closing Price Vs Time Chart(2017 ~ Pressent)')
        fig=plt.figure(figsize=(12,6))
        plt.plot(data.Close)
        st.pyplot(fig)


if (selected == 'Statistical Analysis'):

        #Statistical Analysis
        st.subheader("Graphical Analysis of the data ~")
        st.subheader('ploting Closing Price of the 2 Non-Covid periods')


        df1 = data[['Close']]

        sma = df1.rolling(window=20).mean().dropna()
        rstd = df1.rolling(window=20).std().dropna()

        upper_band = sma + 2 * rstd
        lower_band = sma - 2 * rstd

        upper_band = upper_band.rename(columns={'Close': 'upper'})
        lower_band = lower_band.rename(columns={'Close': 'lower'})
        bb = df1.join(upper_band).join(lower_band)
        bb = bb.dropna()

        buyers = bb[bb['Close'] <= bb['lower']]
        sellers = bb[bb['Close'] >= bb['upper']]

        # Plotting

        import plotly.io as pio
        import plotly.graph_objects as go

        pio.templates.default = "plotly_dark"

        fig11 = go.Figure()
        fig11.add_trace(go.Scatter(x=lower_band.index, 
                                y=lower_band['lower'], 
                                name='Lower Band', 
                                line_color='rgba(173,204,255,0.2)'
                                ))
        fig11.add_trace(go.Scatter(x=upper_band.index, 
                                y=upper_band['upper'], 
                                name='Upper Band', 
                                fill='tonexty', 
                                fillcolor='rgba(173,204,255,0.2)', 
                                line_color='rgba(173,204,255,0.2)'
                                ))
        fig11.add_trace(go.Scatter(x=df1.index, 
                                y=df1['Close'], 
                                name='Close', 
                                line_color='#636EFA'
                                ))
        fig11.add_trace(go.Scatter(x=sma.index, 
                                y=sma['Close'], 
                                name='SMA', 
                                line_color='#FECB52'
                                ))
        fig11.add_trace(go.Scatter(x=buyers.index, 
                                y=buyers['Close'], 
                                name='Buyers', 
                                mode='markers',
                                marker=dict(
                                color='#00CC96',
                                size=10,
                                )
                                ))
        fig11.add_trace(go.Scatter(x=sellers.index, 
                                y=sellers['Close'], 
                                name='Sellers', 
                                mode='markers', 
                                marker=dict(
                                color='#EF553B',
                                size=10,
                                )
                                ))
        st.plotly_chart(fig11, use_container_width=False, sharing="streamlit", theme="streamlit")



        st.text("Histogram")
        #1st non covid period
        fig3=plt.figure(figsize=(10,4))
        plt.hist(NCP1["Close"],bins="rice",label="Daily Close Price")
        plt.legend()
        st.pyplot(fig3)
        #2nd non covid period
        fig4=plt.figure(figsize=(10,4))
        plt.hist(NCP2["Close"],bins="rice",label="Daily Close Price")
        plt.legend()
        st.pyplot(fig4)


        st.text("Box Plot")
        #1st non covid period
        fig5=plt.figure(figsize=(10,4))
        plt.boxplot(NCP1["Close"],labels=["Daily Close Price"])
        st.pyplot(fig5)
        #2nd non covid period
        fig6=plt.figure(figsize=(10,4))
        plt.boxplot(NCP2["Close"],labels=["Daily Close Price"])
        st.pyplot(fig6)
        st.text('''As we can see, it’s full of outliers. The interquartile range 
        (i.e. the height of the box) is quite narrow if compared
        with the distribution total range.This phenomenon is called 
        fat tails and it’s very common in stock analysis.''')

        st.subheader('Descriptive Analysis of the data ~')  # Descriptive Analysis

        st.text("for first Non_Covid period~")
        st.write(NCP1["Close"].describe())
        st.text("For 2nd Non_Covid Period~")
        st.write(NCP2["Close"].describe())

        st.text('''here Positive Mean Price indicates Positive drift of the Closing Price time series.
        standerd deviation is more than an order of magnitude higher than the mean value. It’s clearly the effect of outliers. 
        In stock price analysis, the standard deviation is a measure of the risk and 
        such a high standard deviation is the reason why stocks are considered risky assets.

        Median is also not so different from the mean value, so we might think that the distribution is symmetrical.
        Let’s check the skewness of the distribution to better assess the symmetry:''')

        from scipy.stats import skew
        st.text("Skewness of the both Closing Price datasets")
        st.write(skew(NCP1["Close"]))
        st.write(skew(NCP2["Close"]))
        st.text('''For first Covid Period the skewnwss is negative, so we can assume that the distribution is not symmetrical
        (i.e. null skewness) and that the left tail as a not neglectable weight.
        and for the 2nd Covid Period the skewness is positive , so we can assume that the distribution is not symmetrical 
        and the right tail as a not neglectable weight.

        If we perform a test on the skewness, we find:''')

        st.text("Skewness Test~")
        from scipy.stats import skewtest
        st.write(skewtest(NCP1["Close"]))
        st.write(skewtest(NCP2["Close"]))
        st.text('''The very low p-value suggests that the skewness of the distribution can’t be neglected,
        so we can’t assume that it’s symmetrical.
        Finally, we can measure the kurtosis (scipy normalizes the kurtosis so that it is 0 for a normal distribution)''')

        from scipy.stats import kurtosis
        st.text('Kurtosis of the both Closing Price datasets~')
        st.write(skew(NCP1["Close"]))
        st.write(skew(NCP2["Close"]))
        st.text('''It’s very different from zero, so the distribution is quite different from a normal one. 
        One is Platikurtic and other one is leptokurtic

        A kurtosis test gives us these results:''')

        st.text("Kurtosis Test~")
        from scipy.stats import kurtosistest
        st.write(kurtosistest(NCP1["Close"]))
        st.write(kurtosistest(NCP2["Close"]))
        st.text('''Again, a very small p-value lets us reject the null hypothesis that the kurtosis is the same as a normal distribution (which is 0).

        Are returns normally distributed? Although the statistically significant high values of kurtosis and skewness already tell us that the returns aren’t normally distributed,
        a Q-Q plot will give us graphically clear information.''')


        # Q-Q Plot

        st.subheader("Seasonal Decomposition ~")
        st.set_option('deprecation.showPyplotGlobalUse', False)  #To ignore the error what was occuring
        from statsmodels.tsa.seasonal import seasonal_decompose
        result1=seasonal_decompose(NCP1['Close'], model='multiplicable', period=24)
        plt.figure(figsize=(20,8))
        result1.plot()
        st.pyplot()

        from statsmodels.tsa.seasonal import seasonal_decompose
        result2=seasonal_decompose(NCP2['Close'], model='multiplicable', period=24)
        plt.figure(figsize=(20,8))
        result2.plot()
        st.pyplot()

        st.text('''Here in the both non covid period we are totally unalbe to capture a proper trend 
        for that reason although having a seasonality in the data we try to perform "Moving Average" Method to compare the trend''')




        



        #Moving Average plots
        st.subheader("Trend Analysis of the data ~")
        st.subheader('Closing Price Vs Time chart with 100MA')
        st.text('''In real life most of the Technical Analyst follows the strategy that if the 100day MA
        crosses above the 200day MA then there is an upward trend and if the 100day MA crosses below the 200day MA 
        then it is the starting of an downward trend''')

        ma100=data.Close.rolling(100).mean()
        fig=plt.figure(figsize=(12,6))
        plt.plot(ma100,'r')
        plt.plot(data.Close,'b')
        st.pyplot(fig)


        #Moving Average plots
        st.subheader('Closing Price Vs Time chart with 100MA & 200MA')
        ma100=data.Close.rolling(100).mean()
        ma200=data.Close.rolling(200).mean()
        fig=plt.figure(figsize=(12,6))
        plt.plot(ma100,'r')
        plt.plot(ma200,'g')
        plt.plot(data.Close,'b')
        st.pyplot(fig)


if (selected == 'Prediction with M.L.'):
        st.title("Implimenting L.S.T.M model for predicting the future ~")
        #Splitting Data into Training and Testing
        import pandas as pd
        data_training=pd.DataFrame(data["Close"][0:int(len(data)*0.80)])
        data_test=pd.DataFrame(data['Close'][int(len(data)*0.80):int(len(data))])

        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))

        data_training_array=scaler.fit_transform(data_training)


        #Load My Model
        model=load_model('sarima_model.h5')


        #testing part
        past_100_days=data_training.tail(100)
        final_df=past_100_days.append(data_test,ignore_index=True)
        input_data=scaler.fit_transform(final_df)

        #Testing Part
        x_test=[]
        y_test=[]

        for i in range(100,input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i,0])
        x_test,y_test=np.array(x_test),np.array(y_test)

        y_predicted=model.predict(x_test)

        scaler=scaler.scale_
        scale_factor=1/scaler[0]
        y_predicted=y_predicted*scale_factor
        y_test=y_test*scale_factor



        #Final Graph
        st.subheader('Predictions Vs Original')
        fig2=plt.figure(figsize=(12,6))
        plt.plot(y_test,'b',label='original price')
        plt.plot(y_predicted,'r',label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

        from sklearn.metrics import mean_absolute_error as mae
  
        # calculate MAE
        error = mae(y_test, y_predicted)
        # display
        st.text("MAE ~")
        st.write(error)