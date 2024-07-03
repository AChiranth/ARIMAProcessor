import pandas as pd
import numpy as np
%matplotlib inline
import cufflinks as cf
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

cf.go_offline()



"""
Various libraries needed.
"""

class ARIMAProcessor:
    def __init__(self, csv):
        self.csv = str(csv)
        self.entire_df = pd.read_csv(self.csv)
        self.filtered_df = None
        
        self.training_df = None
        self.seasonal = False
        self.m = 1
        
        self.model = None
        self.forecast = None
        self.forecast_lower = pd.Series()
        self.forecast_upper = pd.Series()
        
    """
    csv is read and saved as instance variable
    """
    
    def filterDf(self, value_column, date_column = "date"):
        
        if isinstance(self.entire_df[value_column].iloc[0], str):
            self.entire_df[value_column] = self.entire_df[value_column].str.replace(',', "").astype(int)
        
        if type(date_column) is str:
            self.filtered_df = self.entire_df[[date_column, value_column]]
            self.filtered_df[date_column] = pd.to_datetime(self.training_df[date_column])
            self.filtered_df.set_index(date_column, inplace = True)
        
        if type(date_column) is list:
            date_column.append(value_column)
            self.filtered_df = self.entire_df[date_column]
            self.filtered_df["date"] = pd.to_datetime(self.filtered_df.iloc[:,0].astype(str) + "-" + self.filtered_df.iloc[:,1].astype(str).str.zfill(2))
            self.filtered_df.set_index("date", inplace = True)
            del self.filtered_df[self.filtered_df.columns[0]]
            del self.filtered_df[self.filtered_df.columns[0]]
            
        
    """
    entire dataframe is filtered down to just value and date column. date is index and is a datetime object
    """
    
    def windowSize(self, end_date = "end"):
        non_seasonal_lengths = [8, 10, 12, 14, 16, 18, 20, 22, 24]
        seasonal_lengths = [12, 14, 16, 18, 20, 22, 24]
        
        seasonalities = [True, False]
        
        overall_AIC = 1000000000000000 # Dummy value, will definitely get an AIC lower than this
        
        
        if end_date == "end":
            #First check the non-seasonal pipeline
            for length in non_seasonal_lengths:
                dummy_df = self.filtered_df.iloc[(length * -1):] # Dummy df that contains the last x rows
                dummy_model = auto_arima(dummy_df, seasonal = False, stepwise = True, suppress_warnings = True)
                if dummy_model.aic() < overall_AIC:
                    self.training_df = dummy_df
                    overall_AIC = dummy_model.aic()
                    
            #Next check the seasonal pipeline
            for length in seasonal_lengths:
                dummy_df = self.filtered_df.iloc[(length * -1):] # Dummy df that contains the last x rows
                dummy_model = auto_arima(dummy_df, seasonal = True, stepwise = True, suppress_warnings = True)
                if dummy_model.aic() < overall_AIC:
                    self.training_df = dummy_df
                    self.seasonal = True
                    self.m = 12
                    overall_AIC = dummy_model.aic()
                        
        
        if end_date != "end":
            DT_end_date = pd.to_datetime(end_date)
            numerical_end_date = self.filtered_df.index.get_loc(DT_end_date)
            #First check the non-seasonal pipeline
            for length in non_seasonal_lengths:
                dummy_df = self.filtered_df[(numerical_end_date - (length - 1)):(numerical_end_date + 1)]
                dummy_model = auto_arima(dummy_df, seasonal = False, stepwise = True, suppress_warnings = True)
                if dummy_model.aic() < overall_AIC:
                    self.training_df = dummy_df
                    overall_AIC = dummy_model.aic()
                
                
            #Next check the seasonal pipeline
            for length in seasonal_lengths:
                
                dummy_df = self.filtered_df[(numerical_end_date - (length - 1)):(numerical_end_date + 1)]
                dummy_model = auto_arima(dummy_df, seasonal = True, m=12, stepwise = True, suppress_warnings = True)
                if dummy_model.aic() < overall_AIC:
                    self.training_df = dummy_df
                    self.seasonal = True
                    self.m = 12
                    overall_AIC = dummy_model.aic()
                    

    
    """
    going through various window lengths, seasonalities, and m values to find the best training dataframe. AIC comparison is the method to find best
    training dataset. end_date refers to the last training date used
    """
                
    def generateModel(self):
        self.model = auto_arima(self.training_df, seasonal = self.seasonal, m = self.m if self.seasonal else 1, stepwise = True, suppress_warnings = True)
    
    """
    auto arima method used to generate arima model based upon the selected best training dataframe
    """
    
    def forecastData(self, no_steps = 4, conf_int = False):
        if not conf_int:
            self.forecast = self.model.predict(n_periods = no_steps)
        if conf_int:
            self.forecast, forecast_ci = self.model.predict(n_periods = no_steps, return_conf_int = True)
            self.forecast_lower = pd.Series(forecast_ci[:,0], index = self.forecast.index)
            self.forecast_upper = pd.Series(forecast_ci[:,1], index = self.forecast.index)
    """
    creating forecast based upon the arima model. number of steps and presence of confidence interval are parameters
    """
    
    def graph(self, library = "matplotlib"):
        arima_df = pd.DataFrame(index = self.training_df.index)
        arima_index = arima_df.index.append(self.forecast.index)
        arima_df = arima_df.reindex(arima_index)
        arima_df["Training Data"] = self.training_df.iloc[:, 0]
        arima_df["ARIMA Prediction"] = self.forecast
        
        
        
        column_title = self.filtered_df.columns[0]
        result = False
        
        for index in self.forecast.index:
            try:
                if pd.notna(self.filtered_df.loc[index, column_title]):
                    result = True
            except KeyError:
                result = False
        
        
        if np.isnan(arima_df["Training Data"].iloc[-1]) and result:
            arima_df["Real Data"] = self.filtered_df.loc[self.forecast.index]
        
        """
        steps needed to create arima_df, which will have training data, ARIMA prediction data, and potentially
        real data
        """
        
        
        if library == "matplotlib":
            if not self.forecast_upper.empty: #Matplotlib with confidence interval present
                
                if "Real Data" in arima_df.columns:
                    plt.plot(arima_df["Real Data"].index, arima_df["Real Data"].values, label = "Real Data")
                
                plt.plot(self.forecast.index, self.forecast.values, label = "ARIMA Prediction")
                plt.fill_between(self.forecast.index, self.forecast_lower.values, self.forecast_upper.values, alpha = 0.5, label = "Confidence Interval")
                plt.plot(arima_df["Training Data"].index, arima_df["Training Data"].values, label = "Training Data")
                plt.title("ARIMA Prediction of Data")
                plt.xlabel("Date")
                plt.ylabel("Count")
                plt.legend()
                plt.show()
                
            elif self.forecast_upper.empty: # Matplotlib with confidence interval not present
                
                if "Real Data" in arima_df.columns:
                    plt.plot(arima_df["Real Data"].index, arima_df["Real Data."].values, label = "Real Data")
                
                plt.plot(self.forecast.index, self.forecast.values, label = "ARIMA Prediction")
                plt.plot(arima_df["Training Data"].index, arima_df["Training Data"].values, label = "Training Data")
                plt.title("ARIMA Prediction of Data")
                plt.xlabel("Date")
                plt.ylabel("Count")
                plt.legend()
                plt.show()
                    
                    
        elif library == "plotly":
            if not self.forecast_upper.empty: #Plotly with confidence interval present
                
                arima_df["Lower Bound"] = self.forecast_lower
                arima_df["Upper Bound"] = self.forecast_upper
                
                training_trace = go.Scatter(x = arima_df.index, y = arima_df["Training Data"], mode = "lines", name = "Training Data")
                arima_trace = go.Scatter(x = arima_df.index, y = arima_df["ARIMA Prediction"], mode = "lines", name = "ARIMA Prediction")
                upper_trace = go.Scatter(x = arima_df.index, y = arima_df["Upper Bound"], mode = "lines", name='Upper Bound', showlegend=True, line=dict(color='rgba(173, 216, 230, 0.5)'))
                lower_trace = go.Scatter(x = arima_df.index, y = arima_df["Lower Bound"], mode = "lines", fill='tonexty', fillcolor='rgba(173, 216, 230, 0.5)', line=dict(color='rgba(173, 216, 230, 0.5)'), name='Lower Bound', showlegend=True)
                
                my_data = [training_trace, arima_trace, upper_trace, lower_trace]
                
                if "Real Data" in arima_df.columns:
                    real_trace = go.Scatter(x = arima_df.index, y = arima_df["Real Data"], mode = "lines", name = "Real Data")
                    my_data = [training_trace, arima_trace, upper_trace, lower_trace, real_trace]
                
                my_layout = go.Layout(title='ARIMA Prediction of Data', xaxis=dict(title='Date'),yaxis=dict(title='Count'), paper_bgcolor='rgb(243, 243, 243)', plot_bgcolor='rgb(243, 243, 243)', font=dict(color='rgb(0, 0, 0)'), title_font=dict(size=20))
                fig = go.Figure(data = my_data, layout = my_layout)
                iplot(fig)
                
            elif self.forecast_upper.empty: #Plotly with confidence interval not present
                arima_df.iplot(kind = "line", xTitle = "Date", yTitle = "Count", title = "ARIMA Prediction of Data", layout_update = {'yaxis': {'type': "linear"}})
    
    """
    creating comprehensive graph using matploblib or plotly. plotting the training data, arima predicted data, upper and
    lower bounds and potentially even the real data
    """
    
    def process(self, valueCol, dateCol = "date", endDate = "end", steps = 4, ci = False, lib = "matplotlib"):
        self.filterDf(value_column=valueCol, date_column=dateCol)
        self.windowSize(end_date=endDate)
        self.generateModel()
        self.forecastData(no_steps=steps, conf_int=ci)
        self.graph(library=lib)
    
