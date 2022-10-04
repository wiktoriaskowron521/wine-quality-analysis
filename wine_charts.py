import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import table
import numpy as np

#dataset: https://archive.ics.uci.edu/ml/datasets/wine+quality
df=pd.read_csv("C:\\Users\\Wiki\\Desktop\\E\\redwine.csv")

#statystyki, tabela ze statystykami
def statistic_table(df):
    #statystyki
    stats=round(df.describe(),2)
    stats.loc["std"]=round(stats.loc["std"],2)
    stats.loc["mean"]=round(stats.loc["mean"],2)
    stats.loc["cv"] = round(stats.loc["std"]/stats.loc["mean"],2)
    stats1=stats.transpose()
    
    #tabela ze statystykami
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False) 
    ax.set_frame_on(False)
    t = table(ax, stats1, loc='upper right', colWidths=[0.17]*len(stats1.columns))
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1.2, 1.2)
    plt.show()

#wykres słupkowy - współczynniki zmienności
def cv_chart(df):
    #statystyki
    stats=df.describe()
    stats.loc["std"]=round(stats.loc["std"],2)
    stats.loc["mean"]=round(stats.loc["mean"],2)
    stats.loc["cv"] = round(stats.loc["std"]/stats.loc["mean"],2)

    #wykres współczynnika zmienności
    data = stats.transpose()["cv"]
    data=data.sort_values()
    sns.set_style('darkgrid')
    sns.barplot(y=data.index, x=data.values,palette="Blues_d")
    plt.title('Wartość współczynnika zmienności dla zmiennych')
    plt.ylabel('Zmienne')
    plt.xlabel('Współczynnik zmienności')
    sns.despine()
    plt.show()

#macierz korelacji - heatmapa
def corr_chart(df):
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()
    
#normalizacja
#usuwam kolumny
df = df.drop(['density','pH'], 1)

#wykresy zależności
def corrs_chart(df):
    g = sns.pairplot(df, corner=True)
    g.map_lower(sns.kdeplot, levels=4, color=".2")
    for ax in g.axes.flatten():
        if ax:
            ax.set_xlabel(ax.get_xlabel(), rotation = 30)
            # rotate y axis labels
            ax.set_ylabel(ax.get_ylabel(), rotation = 0)
            # set y labels alignment
            ax.yaxis.get_label().set_horizontalalignment('right')
    plt.show()

corr_chart(df)


'''df['log_alcohol'] = np.log(df['alcohol'])
df['log_sulphates'] = np.log(df['sulphates'])
df['log_volatileacidity'] = np.log(df['volatile acidity'])

sns.boxplot(data=df[['log_alcohol','log_sulphates','log_volatileacidity']],orient="h", palette="rainbow")
sns.set_theme(style='darkgrid')
plt.title('Wykresy pudełkowe dla zmiennych objaśniających')
sns.despine()
plt.show()'''

#df.to_csv('winedata2.csv') 
stats=round(df.describe(),2)
stats.loc["std"]=round(stats.loc["std"],2)
stats.loc["mean"]=round(stats.loc["mean"],2)
stats.loc["cv"] = round(stats.loc["std"]/stats.loc["mean"],2)
stats1=stats.transpose()
print(stats1)
stats1.to_csv('stats.csv')
