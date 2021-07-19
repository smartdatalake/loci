import plotly.offline as pyo
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
from os import walk
import plotly.graph_objects as go
import plotly.express as pex
import string
from scipy.stats import zscore
from PIL import ImageColor
from statsmodels.tsa.seasonal import seasonal_decompose
from statistics import mean
from sklearn.cluster import DBSCAN
import plotly.express as px
import ruptures as rpt
import copy
import os
import math

class DictionarySizeIsNotSupported(Exception): pass
class StringsAreDifferentLength(Exception): pass
class OverlapSpecifiedIsNotSmallerThanWindowSize(Exception): pass

pyo.init_notebook_mode()

#---------------------------------------------------------#
#------------------------- SAX ---------------------------#
#--- source: https://github.com/serval-snt-uni-lu/dsco ---#
#---------------------------------------------------------#
class SAX(object):
    """
    This class is for computing common things with the Symbolic
    Aggregate approXimation method.  In short, this translates
    a series of data to a string, which can then be compared with other
    such strings using a lookup table.
    """

    def __init__(self, wordSize = 8, alphabetSize = 7, epsilon = 1e-6):

        if alphabetSize < 3 or alphabetSize > 20:
            raise DictionarySizeIsNotSupported()
        self.aOffset = ord('a')
        self.wordSize = wordSize
        self.alphabetSize = alphabetSize
        self.eps = epsilon
        self.breakpoints = {'3' : [-0.43, 0.43],
                            '4' : [-0.67, 0, 0.67],
                            '5' : [-0.84, -0.25, 0.25, 0.84],
                            '6' : [-0.97, -0.43, 0, 0.43, 0.97],
                            '7' : [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                            '8' : [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
                            '9' : [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
                            '10': [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
                            '11': [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
                            '12': [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
                            '13': [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
                            '14': [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
                            '15': [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5],
                            '16': [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53],
                            '17': [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56],
                            '18': [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59],
                            '19': [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62],
                            '20': [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]
                            }
        self.beta = self.breakpoints[str(self.alphabetSize)]
        self.build_letter_compare_dict()
        self.scalingFactor = 1

    def get_breakpoints(self):
        return self.beta.copy()


    def to_letter_rep(self, x):
        """
        Function takes a series of data, x, and transforms it to a string representation
        """
        (paaX, indices) = self.to_PAA(self.normalize(x))
        self.scalingFactor = np.sqrt((len(x) * 1.0) / (self.wordSize * 1.0))
        return (self.alphabetize(paaX), indices)

    def normalize(self, x):
        """
        Function will normalize an array (give it a mean of 0, and a
        standard deviation of 1) unless it's standard deviation is below
        epsilon, in which case it returns an array of zeros the length
        of the original array.
        """
        X = np.asanyarray(x)
        if np.nanstd(X) < self.eps:
            res = []
            for entry in X:
                if not np.isnan(entry):
                    res.append(0)
                else:
                    res.append(np.nan)
            return res
        return (X - np.nanmean(X)) / np.nanstd(X)

    def to_PAA(self, x):
        """
        Function performs Piecewise Aggregate Approximation on data set, reducing
        the dimension of the dataset x to w discrete levels. returns the reduced
        dimension data set, as well as the indices corresponding to the original
        data for each reduced dimension
        """
        n = len(x)
        stepFloat = n/float(self.wordSize)
        step = int(math.ceil(stepFloat))
        frameStart = 0
        approximation = []
        indices = []
        i = 0
        while frameStart <= n-step:
            thisFrame = np.array(x[frameStart:int(frameStart + step)])
            approximation.append(np.mean(thisFrame))
            indices.append((frameStart, int(frameStart + step)))
            i += 1
            frameStart = int(i*stepFloat)
        return (np.array(approximation), indices)

    def alphabetize(self,paaX):
        """
        Converts the Piecewise Aggregate Approximation of x to a series of letters.
        """
        alphabetizedX = ''
        for i in range(0, len(paaX)):
            letterFound = False
            for j in range(0, len(self.beta)):
                if np.isnan(paaX[i]):
                    alphabetizedX += '-'
                    letterFound = True
                    break
                if paaX[i] < self.beta[j]:
                    alphabetizedX += chr(self.aOffset + j)
                    letterFound = True
                    break
            if not letterFound:
                alphabetizedX += chr(self.aOffset + len(self.beta))
        return alphabetizedX

    def compare_strings(self, sA, sB):
        """
        Compares two strings based on individual letter distance
        Requires that both strings are the same length
        """
        if len(sA) != len(sB):
            raise StringsAreDifferentLength()
        list_letters_a = [x for x in sA]
        list_letters_b = [x for x in sB]
        mindist = 0.0
        for i in range(0, len(list_letters_a)):
            if list_letters_a[i] != '-' and list_letters_b[i] != '-':
                mindist += self.compare_letters(list_letters_a[i], list_letters_b[i])**2
        mindist = self.scalingFactor* np.sqrt(mindist)
        return mindist

    def compare_letters(self, la, lb):
        """
        Compare two letters based on letter distance return distance between
        """
        return self.compareDict[la+lb]

    def build_letter_compare_dict(self):
        """
        Builds up the lookup table to determine numeric distance between two letters
        given an alphabet size.  Entries for both 'ab' and 'ba' will be created
        and will have identical values.
        """

        number_rep = range(0,self.alphabetSize)
        letters = [chr(x + self.aOffset) for x in number_rep]
        self.compareDict = {}
        for i in range(0, len(letters)):
            for j in range(0, len(letters)):
                if np.abs(number_rep[i]-number_rep[j]) <=1:
                    self.compareDict[letters[i]+letters[j]] = 0
                else:
                    high_num = np.max([number_rep[i], number_rep[j]])-1
                    low_num = np.min([number_rep[i], number_rep[j]])
                    self.compareDict[letters[i]+letters[j]] = self.beta[high_num] - self.beta[low_num]

    def sliding_window(self, x, numSubsequences = None, overlappingFraction = None):
        if not numSubsequences:
            numSubsequences = 20
        self.windowSize = int(len(x)/numSubsequences)
        if not overlappingFraction:
            overlappingFraction = 0.9
        overlap = self.windowSize*overlappingFraction
        moveSize = int(self.windowSize - overlap)
        if moveSize < 1:
            raise OverlapSpecifiedIsNotSmallerThanWindowSize()
        ptr = 0
        n = len(x)
        windowIndices = []
        stringRep = []
        while ptr < n-self.windowSize+1:
            thisSubRange = x[ptr:ptr+self.windowSize]
            (thisStringRep,indices) = self.to_letter_rep(thisSubRange)
            stringRep.append(thisStringRep)
            windowIndices.append((ptr, ptr+self.windowSize))
            ptr += moveSize
        return (stringRep,windowIndices)

    def batch_compare(self, xStrings, refString):
        return [self.compare_strings(x, refString) for x in xStrings]

    def set_scaling_factor(self, scalingFactor):
        self.scalingFactor = scalingFactor

    def set_window_size(self, windowSize):
        self.windowSize = windowSize
        
        
#-------------------------------------#
#------ OTHER TS FUNCTIONS -----------#
#-------------------------------------#
def read_files(my_path, date_column, data_column):
    df_array = []
    try:
        filenames = next(walk(my_path), (None, None, []))[2]
    except:
        return -1, -1, -1

    if len(filenames) > 0:
        count = 0
        start_date = None
        end_date = None
        for filename in filenames:
            if filename == '.DS_Store':
                continue
            try:
                df = pd.read_csv(my_path + filename, header=None)
            except:
                continue
                
            if count == 0:
                start_date = df.iloc[0][date_column]
                end_date = df.iloc[-1][date_column]
                count += 1

            df['normalized'] = zscore(df[data_column])
            df_array.append(df)

    return df_array, filenames, start_date, end_date

def read_file(my_file):
    try:
        df = pd.read_csv(my_file, header=None)
    except:
        return pd.DataFrame()

    return df
        
def get_date_range(start, end, intv):
    from datetime import datetime
    start = datetime.strptime(start,"%m/%d/%Y")
    end = datetime.strptime(end,"%m/%d/%Y")
    diff = (end  - start ) / intv
    for i in range(intv):
        yield (start + diff * i).strftime("%m/%d/%Y")
    yield end.strftime("%m/%d/%Y")    

def create_sankey(df_array, alphabet_size, word_size, begin, end, from_value, to_value):

    sax = SAX(wordSize = word_size, alphabetSize = alphabet_size)
    breakpoints = sax.get_breakpoints()
    breakpoints.insert(0, float("-inf"))
    breakpoints.append(float("+inf"))
    date_range = list(get_date_range(begin, end, word_size))
    alphabet = dict(enumerate(string.ascii_lowercase))
    alphabet_inv = dict(zip(string.ascii_lowercase, range(0,26)))

    label = []
    node_x = []
    node_y = []
    step_x = 1/(word_size)
    step_y = 1/(alphabet_size)
    curr_x = 0
    curr_y = 0
    dates = []
    breakpoint_intervals = []
    for i in range(0, word_size):
        curr_x = i*step_x
        for j in range(0, alphabet_size):
            curr_y = j*step_y
            label.append(alphabet[j])
            node_x.append(curr_x+0.01)
            node_y.append(curr_y+0.01)
            dates.append(date_range[i])
            breakpoint_intervals.append('from ' + str(breakpoints[alphabet_size-j-1]) + ' to ' + str(breakpoints[alphabet_size-j]))

    source_target_value = {}
    node_set = []
    for df in df_array:
        df[0] = pd.to_datetime(df[0])  
        mask = (df[0] >= begin) & (df[0] <= end)
        df = df.loc[mask]

        if df['normalized'].mean() < from_value or df['normalized'].mean() > to_value:
            continue

        df = df.drop(['normalized'], axis=1)
        signal = df.iloc[:,2:].values
        sax_rep = sax.to_letter_rep(signal)[0]

        for i in range(1, len(sax_rep)):
            source_target = (alphabet_inv[sax_rep[i-1]]+((i-1)*alphabet_size), alphabet_inv[sax_rep[i]]+(i*alphabet_size))
            if source_target not in source_target_value.keys():
                source_target_value[source_target] = 1
            else:
                source_target_value[source_target] = source_target_value[source_target] + 1
            node_set.append(source_target[0])
            node_set.append(source_target[1])

    node_x_pos = []
    node_y_pos = []
    for i in set(node_set):
        node_x_pos.append(node_x[i])
        node_y_pos.append(node_y[i])

    source = []
    target = []
    value = []
    for key in source_target_value:
        source.append(key[0])
        target.append(key[1])
        value.append(int(source_target_value[key]))

    pal1 = sns.color_palette("Spectral", alphabet_size)
    pal1 = pal1.as_hex()

    node_colors = []
    for i in range(0, len(label)):
        node_colors.append(pal1[int(i % alphabet_size)])

    link_colors = [node_colors[src] for src in source]
    link_breakpoint_intervals = [breakpoint_intervals[src] for src in source]

    link_colors_rgb = []
    count = 0
    for color in link_colors:
        rgb = ImageColor.getcolor(color, "RGB")
        link_colors_rgb.append('rgba(' + str(rgb[0]) + ',' + str(rgb[1]) + ',' + str(rgb[2]) + ',' + str(value[count]/(max(value))) + ')')
        count += 1

    fig = go.Figure(data=[go.Sankey(
        arrangement = 'snap',
        node = dict(
          pad = 15,
          thickness = 10,
          line = dict(color = "black", width = 1.75),
          #label = label,
          customdata = dates,
          x=node_x_pos,
          y=node_y_pos,
          color = node_colors
        ),
        link = dict(
          source = source,
          target = target,
          value = value,
          line = dict(color = "black", width = 0.75),
          color =  link_colors_rgb,
          customdata = link_breakpoint_intervals,
          hovertemplate='Outgoing %{value} <br /> From %{source.customdata} to %{target.customdata} <br /> Value range %{customdata}<extra></extra>',
    ))])

    fig.update_layout(autosize=True, font_size=16)
    
    return fig


def get_gain_index(trend,seasonality,model,real_values):
    error1 = [] # error between trend vs actual values
    error2 = [] # error between trend + seasonality vs actual values
    
    for i in range(0,len(real_values)):
        if trend[i] != None and real_values[i] != 0:
            if model == 'Additive':
                predicted = trend[i] 
                real = real_values[i]
                error1.append(abs(real-predicted)/abs(real))
                
                predicted = trend[i] + seasonality[i]
                real = real_values[i]
                error2.append(abs(real-predicted)/abs(real))            
            
            if model == 'Multiplicative':
                predicted = trend[i] 
                real = real_values[i]
                error1.append(abs(real-predicted)/abs(real))
                
                predicted = trend[i] * seasonality[i]
                real = real_values[i]
                error2.append(abs(real-predicted)/abs(real))
        
    error =  (mean(error1) - mean(error2)) / mean(error1)          
    
    return error


def extract_best_period(ts, dates, periods, model):
    # Define gain_index and period
    gain_index = float('-inf') 
    p = -1 # best period
    periods_gain_indexes = {} 
    m = model
    
    # Apply z-normalisation to the time series
    ts = zscore(ts)
    
    # Check if dates are provided and create a df
    if dates != 'None':
        
        # Create df
        d = {'DATE':dates,'values':ts}
        ts = pd.DataFrame(d)
        
        # create datetime index passing the datetime series
        datetime_index =ts['DATE']
        
        # Complete the call to convert the date column
        datetime_index =  pd.to_datetime(datetime_index,format='%m/%d/%Y')
        ts=ts.set_index(datetime_index)
        # we don't need the column anymore
        ts.drop('DATE',axis=1,inplace=True)
        ts = ts['values'].tolist()
    
    # Find the period with the smallest mean_abs error
    for i in periods:
        
        # Seasonal decomposition
        result = seasonal_decompose(ts, period=i,model=m)
        
        trend = np.where(np.isnan(result.trend), None, result.trend)
        seasonality = np.where(np.isnan(result.seasonal),None,result.seasonal)
        res_error = np.where(np.isnan(result.resid), None, result.resid)
        
        g = get_gain_index(trend,seasonality,model=m,real_values=ts)
        
        if g > gain_index:
            gain_index = g
            p = i
            
        periods_gain_indexes.update({i:g})
        
    return p,periods_gain_indexes


def seasonal_decomposition(ts_df, date_column, data_column, periods, m):
    best_period, gain_indexes = extract_best_period(ts_df[data_column].tolist(), ts_df[date_column].tolist(), periods, model=m)
    result = seasonal_decompose(ts_df[data_column].tolist(), period=best_period, model=m)
    
    df = pd.DataFrame(
    {'Date': ts_df[date_column].tolist(),
     'Trend': result.trend,
     'Seasonality': result.seasonal,
     'Residual': result.resid,
    })
    df.set_index('Date')
    
    df.index = pd.to_datetime(df.index)
    
    fig1 = px.line(df, x="Date", y="Trend", title='Trend')
    fig2 = px.line(df, x="Date", y="Seasonality", title='Seasonality')
    fig3 = px.line(df, x="Date", y="Residual", title='Residual')
    
    return result, best_period, gain_indexes, fig1, fig2, fig3

def rate_change(data, changepoints):
    
    # Initialise a dictionary that holds the changing points and their rate of change
    rate_changes = {}
    dir_rate_changes = {}
        
    # Convert the data into pandas format
    df = pd.DataFrame(data,columns=['values'])
    
    for i in range(len(changepoints)-2):
        
        # Take the corresponding segments
        segment1 = df[changepoints[i]:changepoints[i+1]]
        segment2 = df[changepoints[i+1]:changepoints[i+2]]
        
        # Calculate their average value
        mean1 = segment1.mean().values[0]
        mean2 = segment2.mean().values[0]

        # Calculate their absolute percentage difference
       
        abs_percentage_diff = abs((mean2 - mean1) / ((mean1+mean2)/2))
       
        dir_percentage_diff = ((mean2 - mean1) / ((mean1+mean2)/2))
        
        # Update rate_changes
        rate_changes[changepoints[i+1]] = abs_percentage_diff 
        dir_rate_changes[changepoints[i+1]] = dir_percentage_diff
        
       
    return rate_changes, dir_rate_changes

def change_setection_single(ts_df, model, min_size):
    signal = ts_df.iloc[:,2:].values

    algo = rpt.Pelt(model=model, min_size=min_size, jump=5).fit(signal)
    my_bkps = algo.predict(pen=3)

    fig, (ax,) = rpt.display(signal, my_bkps, figsize=(16, 10))
    return fig

def change_detection_collective(ts_df_array, filenames, model, min_size, eps, min_samples, date_column, data_column,):
    # Create Final Dataset
    final_data = pd.DataFrame(columns=['timestamps','rate_change','dir_rate_change','name'])
    num_of_stock = 1
    sftp_v = False

    # Iterate through the provided list with the stocks
    for i in range(0, len(ts_df_array)):

        values = ts_df_array[i][data_column].tolist() # closing values
        stock_dates = ts_df_array[i][date_column].tolist() # dates of the specific stock
        d = values
        entry = num_of_stock
        num_of_stock = filenames[i]
        
        d = np.array(d, dtype=np.float32)

        # Set up parameters for ruptures
        # ------------------------------
        jump = 5
        signal = d

        # Fit the model
        # -------------
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
        my_bkps = algo.predict(pen=3)


        # Calculate the rate of changes 
        # -----------------------------
        my_bkps.insert(0, 0)
        results,dir_results = rate_change(signal,my_bkps)


        # Create the final dataset
        # -----------------------
        df = pd.DataFrame(list(results.items()),columns = ['timestamps','rate_change'])                 
        dir_df = pd.DataFrame(list(dir_results.items()),columns = ['timestamps','dir_rate_change'])               
        dir_rate_values = dir_df['dir_rate_change'].tolist()         
        df['dir_rate_change'] = dir_rate_values
        df['name'] = entry   


        # Add dates if exist
        # ------------------        
        changepoints = df['timestamps'].tolist()
        changepoints_dates = []
        for i in changepoints:
            changepoints_dates.append(stock_dates[i])
        df.insert(2, 'date', changepoints_dates)
        final_data = pd.concat([final_data, df], ignore_index=True)

    # Sort the changing points according to rate change index
    # -------------------------------------------------------
    final_data  = final_data.sort_values(by='rate_change', ascending=False,ignore_index=True)


    # DBSCAN for identifying Global - Local changes
    # ----------------------------------------------

    # Τake the extracted timestamps of the changing points from the whole collection
    timestamps = final_data['timestamps'].values
    
    X = timestamps.reshape(-1,1)

    # Call the DBSCAN algorithm
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Add labels to the data
    final_data['cluster label'] = labels


    # Cluster datarame
    # ----------------

    # Create a new df for the cluster dataframe
    data_changes = copy.deepcopy(final_data)

    # convert timestamps column to floats
    data_changes['timestamps'] = data_changes['timestamps'].astype(float)



    # get mean rate_change
    data_clusters = data_changes.groupby('cluster label', as_index=False) ['rate_change','dir_rate_change'].mean()

    # get min-max date of each cluster
    data_clusters['min_timestamp'] = data_changes.groupby('cluster label', as_index=False)['timestamps'].min()['timestamps']

    data_clusters['max_timestamp'] = data_changes.groupby('cluster label', as_index=False)['timestamps'].max()['timestamps']

    # add an extra column that will hold info regarding with the number of changing points at each cluster
    data_clusters['counts'] = data_clusters['cluster label'].map(data_changes["cluster label"].value_counts())

    # add score column
    data_clusters['cluster_abs_score'] = data_clusters['rate_change'] * data_clusters['counts']

    # add  dir score column
    data_clusters['cluster_dir_score'] = data_clusters['dir_rate_change'] * data_clusters['counts']

    # Check if dates exist and add the min max date to the cluster dataframe
    if 'date' in data_changes.columns:
        min_dates = []
        max_dates = []
        for index, row in data_clusters.iterrows():
            min_dates.append(data_changes.loc[(data_changes['cluster label'] == row['cluster label']) & (data_changes['timestamps'] == row['min_timestamp'])]['date'].reset_index(drop=True)[0])
            max_dates.append(data_changes.loc[(data_changes['cluster label'] == row['cluster label']) & (data_changes['timestamps'] == row['max_timestamp'])]['date'].reset_index(drop=True)[0])


        data_clusters['min_dates'] = min_dates
        data_clusters['max_dates'] = max_dates

    # Remove the local changes cluster 
    data_clusters = data_clusters.iloc[1: , :]

    # Sort the df according to the score 
    data_clusters = data_clusters.sort_values(by=['cluster_abs_score'],ascending=False)
    data_clusters = data_clusters.reset_index(drop=True)

    return final_data, data_clusters