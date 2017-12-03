import pandas as pd
from matplotlib import pyplot as plt

def load_dice_score(name_file, name_score='AveraDiceScore'):
    df = pd.read_csv(name_file)

    test = df['ScribbleId'][0].split('_')
    df['image'] = df['ScribbleId'].map(lambda x:x.split('_')[0])
    if test[1] == 'd':
        df['d'] = df['ScribbleId'].map(lambda x:int(x.split('_')[2]))
    if test[3] == 'n':
        df['n'] = df['ScribbleId'].map(lambda x:int(x.split('_')[4]))
    if test[5] == 'k':
        df['k'] = df['ScribbleId'].map(lambda x:int(x.split('_')[6]))
    
    # Average Score on the k draws
    cc = [c for c in df.columns]
    cc.remove('k')
    cc.remove(name_score)
    
    df = df.groupby(by=cc, as_index=False)[name_score].mean()
    return df

def plot_influence(df, name_score='AveraDiceScore'):
    print("Correlations:\n")
    print(df.corr()[name_score])
    
    df_n = df.groupby(['n'], as_index=False)[name_score].mean()
    df_n.plot(x='n', y=name_score)
    plt.title('Influcence of n')
    
    df_d = df.groupby(['d'], as_index=False)[name_score].mean()
    df_d.plot(x='d', y=name_score)
    plt.title('Influcence of d')

def image_to_num(name):
    if name == 'girafe':
        return 1
    elif name == 'wolf':
        return 2
    elif name == 'tiger':
        return 3
    elif name == 'kangourou':
        return 4
    else:
        return 5