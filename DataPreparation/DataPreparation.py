from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
import matplotlib.pyplot as plt
import os
from DataPreparation import *
from pyspark.sql.functions import monotonically_increasing_id
import numpy as np
import seaborn as sns
from IPython.display import display, HTML, Markdown, Latex
import math
import itertools
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import math
from itertools import combinations

### Initialize SparkSession
spark = SparkSession.builder.appName("AirlineSatisfaction").getOrCreate()
sc = spark.sparkContext                 # shorthand

def read_data(split='train', scales=[], encode=False, keep_y=False):
    '''
    Reads the data from the csv files and returns the x_data and y_data
    split: 'train', 'val', 'all' decides which split of the data to return
    scales: list of strings from the set {'categorical', 'ordinal', 'numerical'} decides which columns to return
    encode: boolean, if True, performs frequency encoding on the categorical columns
    '''
    ### set paths
    module_dir = os.path.dirname(__file__)
    train_path = os.path.join(module_dir, '../DataFiles/airline-train.csv')
    val_path = os.path.join(module_dir, '../DataFiles/airline-test.csv')
    
    ### read the required split of data (and drop useless columns)
    if split == 'train':  
        dataset = spark.read.csv(train_path, header=True, inferSchema=True).drop('_c0').drop('id')
    elif split == 'val':
        dataset = spark.read.csv(val_path, header=True, inferSchema=True).drop('_c0').drop('id')
    elif split == 'all':
        train_dataset = spark.read.csv(train_path, header=True, inferSchema=True).drop('_c0').drop('id')
        val_dataset = dataset.union(spark.read.csv(val_path, header=True, inferSchema=True).drop('_c0').drop('id'))
        dataset = train_dataset.union(val_dataset)
           

    ### form y_data and x_data from dataset
    y_data = dataset.select('satisfaction')             # select the satisfaction column
    indexer = StringIndexer(inputCol='satisfaction', outputCol='satisfaction_index')
    y_data = indexer.fit(y_data).transform(y_data)
    y_data = y_data.select('satisfaction_index')
    if not keep_y:
        x_data = dataset.drop('satisfaction')               # drop the satisfaction column
        
    # convert the Arrival Delay in Minutes column to integer
    x_data = x_data.withColumn('Arrival Delay in Minutes', x_data['Arrival Delay in Minutes'].cast(IntegerType()))
    
    
    ### select specific columns from x if specified
    if type is not None:
        columns_struct = {                      # A dictionary to store the names of columns of each type
            'categorical': [],
            'ordinal': [],
            'numerical': []
        }
        for col in x_data.columns:
            if x_data.schema[col].dataType == StringType():
                columns_struct['categorical'].append(col)
            elif x_data.schema[col].dataType == IntegerType() and x_data.select(col).distinct().count() <= 6:
                columns_struct['ordinal'].append(col)
            else:
                columns_struct['numerical'].append(col)
        
        selected_cols_types = []
        for col_type in scales:                # Now lets merge the lists of columns of the required types
            selected_cols_types.extend(columns_struct[col_type])
        x_data = x_data.select(selected_cols_types)

    ### if required, perform frequency encoding on the categorical columns
    if encode == True:
        norm = x_data.count()                  # total number of rows
        for col in x_data.columns:
            if x_data.schema[col].dataType == StringType():
                col_freq = x_data.groupBy(col).count()                      # This yields a table with (col, count) columns
                col_freq = col_freq.withColumnRenamed('count', col+'_freq') # Rename the count column to col_freq
                x_data = x_data.join(col_freq, col)                         # There will always be a one-to-one match 
                x_data = x_data.drop(col)
                # normalize the frequencies
                x_data = x_data.withColumn(col+'_freq', x_data[col+'_freq']/norm)
        
    
    return x_data, y_data



def prior_class_distribution(y_data):
    """
    Returns the prior class distribution of the dataset
    """
    ### Group by to get the two counts
    classes = y_data.groupBy('satisfaction_index').count()
    # now the result is small enough to fit in memory, so convert it to pandas
    classes = classes.toPandas()
    
    ### plot the distribution of the satisfaction column using matplotlib 
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 200        # increase plot resolution
    plt.figure(figsize=(10, 5))
    plt.margins(0.1)                        # some margin
    plt.bar(['neutral or dissatisfied', 'satisfied'], classes['count'], color='teal', width=0.5)
    # include counts on top of the bars
    for i, count in enumerate(classes['count']):
        plt.text(x=i-0.05, y=count+1000, s=count)       # text s at x, y
        
    plt.title('Prior Class Distribution of the Dataset')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Number of Samples')
    plt.show()
    
    


def feature_histograms_analysis(x_data, y_data=None, per_class=False):
    '''
    Generates a series of three histogram plots for each group of columns of the same type (categorical, ordinal, numerical).
    
    If per_class is True and y_data is given this calls a more complex function to generate the histograms. 
    Both could be merged but the resulting function would become too complex to later debug.
    
    '''
    
    def grid_plot_generator(x_data,  rows, cols, title, figsize=(20, 10), numerical=False):
        """
        Generates a rows x cols grid of plots for the given data where each plot is a bar chart.
        If data is numerical, it uses a histogram instead.
        """
        ### Set up plt grid
        plt.style.use('dark_background')
        plt.rcParams['figure.dpi'] = 200        # increase plot resolution
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
        
        ### For each column plot in the right subplot of the grid
        for i, col in enumerate(x_data.columns):
            
            if rows == 1:
                ax = axs[i % cols]                      # the line below wouldn't work for one row
            else:
                ax = axs[i // cols, i % cols]
                
            if not numerical:                           # bar chart works in this case
                x_data_hist = x_data.groupBy(col).count().sort(col, ascending=True).toPandas()
                labels, counts = list(x_data_hist[col]), list(x_data_hist['count'])
                ax.bar(labels, counts, color='teal')

            else:                                       # histogram but sample data to fit in ram first
                x_data_sample = x_data.sample(fraction=1.0, seed=3).toPandas()[col]
                
                ax.hist(x_data_sample, bins=20, color='teal')
                
            ax.set_title(col)
            ax.set_xlabel('')
            ax.tick_params(labelsize=9)

        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.87)

    if per_class == False:
        x_data_cat = x_data.select([ col for col in x_data.columns if x_data.schema[col].dataType == StringType() ])
        x_data_ord = x_data.select([ col for col in x_data.columns if x_data.schema[col].dataType == IntegerType() and x_data.select(col).distinct().count() <= 6 ])
        x_data_num = x_data.select([ col for col in x_data.columns if x_data.schema[col].dataType == IntegerType() and x_data.select(col).distinct().count() > 6 ])        
    
        grid_plot_generator(x_data_cat, 1, 4, "Distribution of Nominal Columns", figsize=(25,5))
        grid_plot_generator(x_data_ord, 2, 7, "Distribution of Ordinal Columns", figsize=(25, 8))
        grid_plot_generator(x_data_num, 1, 4, "Distribution of Numerical Columns", figsize=(25,5), numerical=True)

    else:
        assert y_data != None, "y_data cannot be None if per_class is True"
        # call the per_class function
        feature_histograms_analysis_perclass(x_data, y_data)
    
    
    

def feature_histograms_analysis_perclass(x_data, y_data):
    '''
    This function modifies the one above to handle the per_class case by using stacked bar charts and double plotting
    in case of numerical data
    '''
    def grid_plot_generator(x_data, rows, cols, title, figsize=(20, 10), numerical=False):
        """
        Generates a rows x cols grid of plots for the given data where each plot is a bar chart.
        If data is numerical, it uses a histogram instead.
        """
        # partition x_data into two sets depending on value of y_data
        x_data_id, y_data_id = x_data.withColumn('id', monotonically_increasing_id()), y_data.withColumn('id', monotonically_increasing_id())
        x_data_0 = x_data_id.join(y_data_id, x_data_id.id == y_data_id.id).filter(y_data_id.satisfaction_index == 0).drop(y_data_id.satisfaction_index).drop('id')
        x_data_1 = x_data_id.join(y_data_id, x_data_id.id == y_data_id.id).filter(y_data_id.satisfaction_index == 1).drop(y_data_id.satisfaction_index).drop('id')
        
        
        # Make a 2x7 plot for the 14 numerical columns
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
        plt.style.use('dark_background')
        plt.rcParams['figure.dpi'] = 200        # increase plot resolution

        columns = x_data.columns
        if numerical:    columns = columns + columns
        
        for i, col in enumerate(columns):
            if rows == 1:
                ax = axs[i % cols]
            else:
                ax = axs[i // cols, i % cols]
            
            if not numerical:
                x_data_1_hist = x_data_1.groupBy(col).count().sort(col, ascending=True).toPandas()
                labels_1, values_1 = list(x_data_1_hist[col]), list(x_data_1_hist['count'])
                x_data_0_hist = x_data_0.groupBy(col).count().sort(col, ascending=True).toPandas()
                labels_0, values_0 = list(x_data_0_hist[col]), list(x_data_0_hist['count'])  
                
                # if a label 0 doesnt exist, add it with count 0
                for label in labels_1:
                    if label not in labels_0:
                        labels_0.append(label)
                        values_0.append(0)
                labels_0, values_0 = zip(*sorted(zip(labels_0, values_0)))
                for label in labels_0:
                    if label not in labels_1:
                        labels_1.append(label)
                        values_1.append(0)
                labels_1, values_1 = zip(*sorted(zip(labels_1, values_1)))
                
                ax.bar(labels_1, values_1, color='#44a5c2', label='satisfied')
                ax.bar(labels_0, values_0, color='#024b7a', label='potentially unsatisfied', bottom=values_1)
                
                ax.legend()
                ax.margins(0.2)         
                ax.set_xticks(labels_0)
                
            else:
                x_data = x_data_0 if i <= 4 else x_data_1
                
                x_data_sample = x_data.sample(fraction=1.0, seed=3).toPandas()[col]
                
                ax.hist(x_data_sample, bins=20, color='#44a5c2' if i//cols == 0 else '#024b7a')
                
            ax.set_title(col)
            ax.set_xlabel('')
            ax.tick_params(labelsize=9)

        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.87)


    x_data_cat = x_data.select([ col for col in x_data.columns if x_data.schema[col].dataType == StringType() ])
    x_data_ord = x_data.select([ col for col in x_data.columns if x_data.schema[col].dataType == IntegerType() and x_data.select(col).distinct().count() <= 6 ])
    x_data_num = x_data.select([ col for col in x_data.columns if x_data.schema[col].dataType == IntegerType() and x_data.select(col).distinct().count() > 6 ])

    grid_plot_generator(x_data_cat, 1, 4, "Distribution of Nominal Columns", figsize=(25,5))
    grid_plot_generator(x_data_ord, 2, 7, "Distribution of Ordinal Columns", figsize=(25, 8))
    grid_plot_generator(x_data_num, 2, 4, "Distribution of Numerical Columns", figsize=(25,10), numerical=True)
    


def correlation_ratio(x_data, col1, col2):
    '''
    A measure of association between a categorical variable and a continuous variable.
    - Divide the continuous variable into N groups, based on the categories of the categorical variable.
    - Find the mean of the continuous variable in each group.
    - Compute a weighted variance of the means where the weights are the size of each group.
    - Divide the weighted variance by the variance of the continuous variable.
    
    It asks the question: If the category changes are the values of the continuous variable on average different?
    If this is zero then the average is the same over all categories so there is no association.
    
    Here its assumed that col1 is the categorical variable and col2 is the continuous variable.
    '''
    categories = np.array(x_data[col1])
    values = np.array(x_data[col2]).astype(int)
    group_variances = 0
    for category in set(categories):
        group = values[np.where(categories == category)[0]]
        group_variances += len(group)*(np.mean(group)-np.mean(values))**2
    total_variance = sum((values-np.mean(values))**2)

    return (group_variances / total_variance)**.5

def mix_correlation_matrix(x_data):
    '''
    plot a correlation matrix for the categorical and continuous features in the dataset
    '''
    cat_columns = [ col for col in x_data.columns if x_data.schema[col].dataType == StringType() ]
    cont_columns = [ col for col in x_data.columns if x_data.schema[col].dataType == IntegerType() ]
    
    x_data = x_data.sample(1.0).toPandas()
    
    corr = np.zeros((len(cat_columns), len(cont_columns)))
    for i in range(len(cat_columns)):
        for j in range(len(cont_columns)):
            corr[i, j] = correlation_ratio(x_data,  cat_columns[i], cont_columns[j])
            # if corr is nan then which two columns are being compared
            if np.isnan(corr[i, j]):
                print(cat_columns[i], cont_columns[j])
    
    corr = corr.T
    
 
    sns.set(font_scale=0.8, style="white")
    plt.figure(figsize=(4,18),  dpi=150, facecolor='#1d1d1d')
    g = sns.heatmap(corr, annot=True, cmap=sns.color_palette("flare", as_cmap=True), cbar=False, xticklabels=cat_columns, yticklabels=cont_columns, annot_kws={"color": "white"})
    for tick_label in g.axes.get_yticklabels():
        tick_label.set_color("white")
        tick_label.set_fontsize("9")
    for tick_label in g.axes.get_xticklabels():
        tick_label.set_color("white")
        tick_label.set_fontsize("9")
    plt.show()


def numerical_correlation_matrix(x_data):
    '''
    plot a correlation matrix for the numerical columns in the dataset
    '''
    
    num_columns = [ col for col in x_data.columns if x_data.schema[col].dataType == IntegerType() and x_data.select(col).distinct().count() > 6 ]
    
    x_data = x_data.sample(1.0).toPandas()
    
    corr = x_data[num_columns].corr()
    sns.set(font_scale=0.8, style="white")
    plt.figure(figsize=(10,10),  dpi=120, facecolor='#1d1d1d')
    g = sns.heatmap(corr, annot=True, cmap=sns.color_palette("flare", as_cmap=True), cbar=False, xticklabels=num_columns, yticklabels=num_columns, annot_kws={"color": "white"})
    for tick_label in g.axes.get_yticklabels():
        tick_label.set_color("white")
        tick_label.set_fontsize("9")
    for tick_label in g.axes.get_xticklabels():
        tick_label.set_color("white")
        tick_label.set_fontsize("9")
    plt.show()
    
    
def NaiveBayesTest(x_data, y_data, columns_vals):
    '''
    Given a dataset through x_data and y_data and a set of categorical or ordinal columns and their values as a dict, this
    checks if the Naive Bayes assumption holds over such columns.
    
    Notice that this automatically makes the trst over both classes. So column_values is restricted to columns of x.
    '''
    def multinomial_prob(x_data, columns_vals):
        '''
        Compute the probability of a set of categorical or ordinal columns using the definition of the probability.
        '''
        size = x_data.count()
        for col, val in columns_vals.items():
            x_data = x_data.filter(x_data[col] == val)
        return x_data.count()/size


    def univariate_prob(x_data, columns_vals):
        '''
        Compute the probability of a set of categorical or ordinal columns using the Naive Bayes assumption.
        '''
        size = x_data.count()
        prob = 1
        for col, val in columns_vals.items():
            x_data = x_data.filter(x_data[col] == val)
            prob *= x_data.count()/size
        return prob
    
    # Merge the two dataframes
    x_data = x_data.withColumn('id', monotonically_increasing_id())
    y_data = y_data.withColumn('id', monotonically_increasing_id())
    dataset = x_data.join(y_data, 'id').drop('id')
    
    x_data_0 = dataset.filter(dataset['satisfaction_index'] == 0)
    x_data_1 = dataset.filter(dataset['satisfaction_index'] == 1)
    
    # find the probabilitoes P(x1, x2, ...|c=0) and P(x1, x2, ...|c=1)
    prob_0 = round(multinomial_prob(x_data_0, columns_vals), 2)
    prob_1 = round(multinomial_prob(x_data_1, columns_vals), 2)
    
    # find them again use the naive bayes assumption
    prob_0_n = round(univariate_prob(x_data_0, columns_vals), 2)
    prob_1_n = round(univariate_prob(x_data_1, columns_vals), 2)
    
    
    # check if the probabilities using and not using the naive assumption for each class are close to each other
    if math.isclose(prob_0_n/prob_0, 1, abs_tol=0.05) and math.isclose(prob_1_n/prob_1, 1, abs_tol=0.05):
        print('Success, the Naive Bayes assumption is valid')
        
    else:
        analysis = f'''<font size=4> As expected, the Naive Bayes assumption does not hold. In particular, we have that
                    $$P(x_1, x_2, ...|C_1=0)={prob_0}$$ 
                    as computed numerically using the definition of the probability.
                    Meanwhile, applying the Naive Bayes assumption we have that
                    $$P(x_1, x_2, ...|C_1=0)=P(x_1|C_1=0)P(x_2|C_1=0)...={prob_0_n}$$ 
                    which is different from the correct probability.
                    \n\nLikewise, for the class $C_1=1$ we have that
                    $$P(x_1, x_2, ...|C_1=1)={prob_1}$$ 
                    but 
                    $$P(x_1, x_2, ...|C_1=1)=P(x_1|C_1=1)P(x_2|C_1=1)...={prob_1_n}$$ 
                    which is different from the correct probability.
                    </font>
                    '''
        display(Markdown(analysis))



def visualize_continuous_data(x_data, y_data, HighD=False):
    '''
    Plot all possible 4c2 pairs of continuous features in a scatter plot grid of 2x3. If HighD is True, 
    then plot all possible 4c3 pairs in a 1x4 grid of 3D scatter plots.
    '''
    cont_columns = [ col for col in x_data.columns if x_data.schema[col].dataType == IntegerType() and x_data.select(col).distinct().count() > 6 ]
    x_data_cont = x_data.select(cont_columns)
    
    x_data_cont = x_data_cont.sample(1.0).toPandas()
    y_data = y_data.sample(1.0).toPandas()

    x_data_cont_1 = x_data_cont[y_data['satisfaction_index'] == 1]
    x_data_cont_0 = x_data_cont[y_data['satisfaction_index'] == 0]
    
    if not HighD:
        # plot all possible 4c2 pairs of continuous features in a scatter plot grid of 2x3
        combinations = list(itertools.combinations(cont_columns, 2))
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))    
        axes = axes.flatten()
        for i, (col1, col2) in enumerate(combinations):
            axes[i].scatter(x_data_cont_1[col1], x_data_cont_1[col2], color='cyan', label='satisfied', s=2, alpha=0.7, marker='.')
            axes[i].scatter(x_data_cont_0[col1], x_data_cont_0[col2], color='pink', label='not satisfied', s=2, alpha=0.7, marker='.')
            axes[i].title.set_color('white')
            axes[i].set_xlabel(col1)
            axes[i].set_ylabel(col2)
            axes[i].legend()
        plt.show()    
        
    else:
        # Now will plot in 3D in this case we have 4c3 = 4 combinations
        # So this can be done in a 1x4 grid
        combinations = list(itertools.combinations(cont_columns, 3))
        fig = plt.figure(figsize=(17, 10))
        plt.subplots_adjust(left=0.05, right=0.95)
        for i, (col1, col2, col3) in enumerate(combinations):
            ax = fig.add_subplot(1, 4, i+1, projection='3d')
            ax.grid(False)                                                              # remove grid             
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False        # remove grey planes
            ax.scatter(x_data_cont_0[col1], x_data_cont_0[col2], x_data_cont_0[col3], color='pink', label='not satisfied', s=2, alpha=0.7, marker='.')
            ax.scatter(x_data_cont_1[col1], x_data_cont_1[col2], x_data_cont_1[col3], color='cyan', label='satisfied', s=2, alpha=0.7, marker='.')
      
            # reduce label and tick font size
            ax.set_xlabel(col1, fontsize=8)
            ax.set_ylabel(col2, fontsize=8)
            ax.set_zlabel(col3, fontsize=8)            
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.legend()
            
        plt.show()
        
        
        

def visualize_categorical_data(x_data, y_data):
    '''
    This was supposed to make a ballon plot of pie charts for each possible pair of categorical variables 
    (including ordinal ones).
    Due to limitations with Matpllotlib, equivalently this plots a heat map for each pair of categorical variables
    where the color of each cell is proportional to the number of observations with that pair of values and the number
    is the proportion of satisfaction for that pair of values.
    
    '''
    # get ordinal and categorical columns in x_data
    ordinal_cols = [col for col in x_data.columns if x_data.schema[col].dataType == IntegerType() and x_data.select(col).distinct().count() <= 6]
    categorical_cols = [col for col in x_data.columns if x_data.schema[col].dataType == StringType()]
    categorical_cols = categorical_cols + ordinal_cols

    x_data_cat = x_data.select(categorical_cols).sample(1.0).toPandas()
    y_data = y_data.sample(1.0).toPandas()

    # get all possible pairs of categorical columns
    cat_pairs = list(combinations(categorical_cols, 2))

    num_cols = 5
    num_rows = math.ceil(len(cat_pairs) / num_cols)

    # create a grid of plots with 5 columns and as many rows as needed
    # set dark background
    plt.style.use('dark_background')
    plt.rcParams['figure.dpi'] = 300
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5*num_rows))

    # loop over all pairs of columns and create a heatmap for each pair
    for i, (col1, col2) in enumerate(cat_pairs):
        
        row_idx = i // num_cols
        col_idx = i % num_cols
        
        # selecting the two categorical features
        x_data_pair = x_data_cat[[col1, col2]]

        # computing the cross table
        ct = pd.crosstab(x_data_pair[col1], x_data_pair[col2])

        # for each element in the cross table, compute the fraction of 1s in y_data corresponding to the two category values
        props = np.zeros((ct.shape[0], ct.shape[1]))
        y_data_n = np.array(y_data)
        
        # compute proportions (the value that shows up in the heatmap)
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                col1_val = ct.index[i]
                col2_val = ct.columns[j]
                # for each row in x_data_pair, get the indices of the rows where the two columns have the values col1_val and col2_val
                indices = np.where((x_data_pair[col1] == col1_val) & (x_data_pair[col2] == col2_val))[0]
                # use pandas to get rows of y_data corresponding to the indices
                y_data_pair = np.take(y_data_n, indices, axis=0)
                props[i,j] = round(np.sum(y_data_pair) / y_data_pair.shape[0], 2) if y_data_pair.shape[0] > 0 else 0
        
        # plot the heatmap on the appropriate axis
        ax = axes[row_idx, col_idx]
        im = ax.imshow(ct, cmap='YlGnBu')
        
        # Setting the ticks and labels
        ax.set_xticks(np.arange(len(ct.columns)))
        ax.set_yticks(np.arange(len(ct.index)))
        ax.set_xticklabels(ct.columns)
        ax.set_yticklabels(ct.index)
        ax.set_xlabel(col2)
        #ax.set_ylabel(col1)

        # Setting the text for each cell
        for i in range(len(ct.index)):
            for j in range(len(ct.columns)):
                text = ax.text(j, i, props[i,j], ha='center', va='center', color='deeppink')

    # remove empty plots
    for i in range(len(cat_pairs), num_rows*num_cols):
        row_idx = i // num_cols
        col_idx = i % num_cols
        ax = axes[row_idx, col_idx]
        ax.axis('off')
        

    # adjust the spacing between subplots and show the plot
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()



def convey_insights(bullets_arr):
    '''
    Give it a bullet points array, give you bullet points in markdown for insights.
    '''
    # make a markdown string with the bullets
    markdown_str = '<h3><font color="pink" size=5>Insights</font></h3> <font size=4>\n'
    
    for bullet in bullets_arr:
        markdown_str += '<font color="pink">✦</font> ' + bullet + '<br>'
    # display the markdown string
    markdown_str += '</font>'
    display(Markdown(markdown_str))