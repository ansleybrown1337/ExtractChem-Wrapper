import seaborn as sns
import numpy as np
import math
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, RepeatedKFold
from statsmodels.tools.eval_measures import rmse
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from scipy.optimize import curve_fit
import GOF_stats as gof
#sns.set_palette(sns.dark_palette("mediumaquamarine", reverse=True)) #Set correct color pallete for continuous data
#sns.set_style('ticks', {'font.family':'serif', 'font.serif':'Times New Roman'})
###############################################################rounding function
def myround(x, base=5):
    return int(math.ceil(x/base)*base)
def bootstrap(data, n=10000, func=np.median, p=0.05):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    xbar_init = np.mean(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()
    """
    Return 2-sided symmetric confidence interval specified
    by p.
    """
    u_pval = (1+p)/2.
    l_pval = (1-u_pval)
    l_indx = int(np.floor(n*l_pval))
    u_indx = int(np.floor(n*u_pval))
    print('statistic: ', np.median(data))
    return(simulations[l_indx],simulations[u_indx])
    #return(data.apply(ci(p=0.05)))
def clean_outliers(df):
    #returns df without outliers based on relative diff of obs and pred values
    #using 5 standard deviations from the mean
    return df[(np.abs(stats.zscore(df.Rel_diff_ECpw)) < 5)]
#############################################################categorize function
def func(x, interval=150):
    breaks = [interval, interval*2, interval*3, interval*4]
    if 0 < x < interval:
        return f'0 to <{breaks[0]}'
    elif interval < x < interval*2:
        return f'{breaks[0]} to <{breaks[1]}'
    elif interval*2 < x < interval*3:
        return f'{breaks[1]} to <{breaks[2]}'
    elif interval*3 < x < interval*4:
        return f'{breaks[2]} to <{breaks[3]}'
    return f'>{breaks[3]}'
############plot stuff be sure to use the AJBatch df merged with the ExCh out df
#making category columns for observed and modelled values
def categories(df):
    df['cl_cat']=df['Cl'].apply((lambda x: func(x, interval=5)))
    df['so4_cat']=df['SO4'].apply((lambda x: func(x, interval=25)))
    df['cl_cat_mod']=df['Chloride_result_solution'].apply((lambda x: func(x, interval=5)))
    df['so4_cat_mod']=df['Sulphate_result_solution'].apply((lambda x: func(x, interval=25)))
    df['caco3_cat']=df['CaCO3'].apply((lambda x: func(x, interval=60)))
    df['gyp_cat']=df['Gypsum'].apply((lambda x: func(x, interval=15)))

def ion_levels(df):
    cl_hue=sorted(df.cl_cat.unique())
    so4_hue=sorted(df.so4_cat.unique())
    caco3_hue=sorted(df.caco3_cat.unique())
    gyp_hue=sorted(df.gyp_cat.unique())

    return(cl_hue, so4_hue, caco3_hue, gyp_hue)

def ion_plots(df, final_type, change, directory, save=False):
    if change==False:
        ycol=['Na','Cl','SO4','Ca','Mg']
    else:
        ycol=['change_Na','change_Cl','change_SO4','change_Ca','change_Mg']
    if final_type=='ECe':
        xcol = 'ECe'
        xcol_mod = 'Final EC'
        xlab = '$EC_{e, obs}$'
        xlab_mod = '$EC_{e, mod}$'
        ylab = ['$Na_{e}$ ($mmol_c$ $L^{-1}$)',
                '$Cl_{e}$ ($mmol_c$ $L^{-1}$)',
                '$SO_{4, e}$ ($mmol_c$ $L^{-1}$)',
                '$Ca_{e}$ ($mmol_c$ $L^{-1}$)',
                '$Mg_{e}$ ($mmol_c$ $L^{-1}$)']
        ylab_mod = ['$Na_{e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Cl_{e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$SO_{4 e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Ca_{e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Mg_{e, mod}$ ($mmol_c$ $L^{-1}$)']

    elif final_type=='ECpw':
        xcol = 'ECpw'
        xcol_mod = 'Final EC'
        xlab = '$EC_{pw, obs}$'
        xlab_mod = '$EC_{pw, mod}$'
        ylab = ['$Na_{pw}$ ($mmol_c$ $L^{-1}$)',
                '$Cl_{pw}$ ($mmol_c$ $L^{-1}$)',
                '$SO_{4, pw}$ ($mmol_c$ $L^{-1}$)',
                '$Ca_{pw}$ ($mmol_c$ $L^{-1}$)',
                '$Mg_{pw}$ ($mmol_c$ $L^{-1}$)']
        ylab_mod = ['$Na_{pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Cl_{pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$SO_{4 pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Ca_{pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Mg_{pw, mod}$ ($mmol_c$ $L^{-1}$)']

    na_plots(df, ycol[0], xcol, xcol_mod, xlab, xlab_mod, ylab[0], ylab_mod[0], ion_levels(df)[0], save, directory)
    cl_plots(df, ycol[1], xcol, xcol_mod, xlab, xlab_mod, ylab[1], ylab_mod[1], ion_levels(df)[1], save, directory)
    so4_plots(df, ycol[2], xcol, xcol_mod, xlab, xlab_mod, ylab[2], ylab_mod[2], ion_levels(df)[0], save, directory)
    ca_plots(df, ycol[3], xcol, xcol_mod, xlab, xlab_mod, ylab[3], ylab_mod[3], ion_levels(df)[0], save, directory)
    mg_plots(df, ycol[4], xcol, xcol_mod, xlab, xlab_mod, ylab[4], ylab_mod[4], ion_levels(df)[0], save, directory)

#Sodium plots
def na_plots(df, ycol, xcol, xcol_mod, xlab, xlab_mod, ylab, ylab_mod, hue_order, save, directory):
    #f, ax = plt.subplots(ncols=2)
    Na_obs=sns.relplot(x=xcol, y=ycol, data=df, hue='cl_cat', hue_order=hue_order)
    Na_obs.set(xlabel=xlab, ylabel=ylab, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['Na']),max(df['Sodium_result_solution'])))))
    Na_obs._legend.texts[0].set_text('')
    Na_obs._legend.set_title('   $Cl_e$\n ($mmol_c$ $L^{-1}$)')
    Na_obs._legend._legend_box.sep = -5  # move title down slightly
    Na_mod=sns.relplot(x=xcol_mod, y='Sodium_result_solution', data=df, hue='cl_cat_mod', hue_order=hue_order)
    Na_mod.set(xlabel=xlab_mod, ylabel=ylab_mod, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['Na']),max(df['Sodium_result_solution'])))))
    Na_mod._legend.texts[0].set_text('')
    Na_mod._legend.set_title('   $Cl_{e, mod}$\n ($mmol_c$ $L^{-1}$)')
    Na_mod._legend._legend_box.sep = -5  # move title down slightly
    if save==True:
        Na_obs.savefig(directory+'\\'+'Na_obs v EC_obs.png')
        Na_mod.savefig(directory+'\\'+'Na_mod v EC_mod.png')
#chloride Plot
def cl_plots(df, ycol, xcol, xcol_mod, xlab, xlab_mod, ylab, ylab_mod, hue_order, save, directory):
    cl_obs=sns.relplot(x=xcol, y=ycol, data=df, hue='so4_cat', hue_order=hue_order)
    cl_obs.set(xlabel=xlab, ylabel=ylab, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['Cl']),max(df['Chloride_result_solution'])))))
    cl_obs._legend.texts[0].set_text('')
    cl_obs._legend.set_title('   $SO_{4 e}$\n ($mmol_c$ $L^{-1}$)')
    cl_obs._legend._legend_box.sep = -5  # move title down slightly
    cl_mod=sns.relplot(x=xcol_mod, y='Chloride_result_solution', data=df, hue='so4_cat_mod', hue_order=hue_order)
    cl_mod.set(xlabel=xlab_mod, ylabel=ylab_mod, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['Cl']),max(df['Chloride_result_solution'])))))
    cl_mod._legend.texts[0].set_text('')
    cl_mod._legend.set_title('   $SO_{4 e, mod}$\n ($mmol_c$ $L^{-1}$)')
    cl_mod._legend._legend_box.sep = -5  # move title down slightly
    if save==True:
        cl_obs.savefig(directory+'\\'+'Cl_obs v EC_obs.png')
        cl_mod.savefig(directory+'\\'+'Cl_mod v EC_mod.png')
#sulfate plot
def so4_plots(df, ycol, xcol, xcol_mod, xlab, xlab_mod, ylab, ylab_mod, hue_order, save, directory):
    so4_obs=sns.relplot(x=xcol, y=ycol, data=df, hue='cl_cat', hue_order=hue_order)
    so4_obs.set(xlabel=xlab, ylabel=ylab, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['SO4']),max(df['Sulphate_result_solution'])))))
    so4_obs._legend.texts[0].set_text('')
    so4_obs._legend.set_title('   $Cl_e$\n ($mmol_c$ $L^{-1}$)')
    so4_obs._legend._legend_box.sep = -5  # move title down slightly
    so4_mod=sns.relplot(x=xcol_mod, y='Sulphate_result_solution', data=df, hue='cl_cat_mod', hue_order=hue_order)
    so4_mod.set(xlabel=xlab_mod, ylabel=ylab_mod, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['SO4']),max(df['Sulphate_result_solution'])))))
    so4_mod._legend.texts[0].set_text('')
    so4_mod._legend.set_title('   $Cl_{e, mod}$\n ($mmol_c$ $L^{-1}$)')
    so4_mod._legend._legend_box.sep = -5  # move title down slightly
    if save==True:
        so4_obs.savefig(directory+'\\'+'SO4_obs v EC_obs.png')
        so4_mod.savefig(directory+'\\'+'SO4_mod v EC_mod.png')
#Calcium plot
def ca_plots(df, ycol, xcol, xcol_mod, xlab, xlab_mod, ylab, ylab_mod, hue_order, save, directory):
    ca_obs=sns.relplot(x=xcol, y=ycol, data=df, hue='cl_cat', hue_order=hue_order)
    ca_obs.set(xlabel=xlab, ylabel=ylab, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['Ca']),max(df['Calcium_result_solution'])))))
    ca_obs._legend.texts[0].set_text('')
    ca_obs._legend.set_title('   $Cl_e$\n ($mmol_c$ $L^{-1}$)')
    ca_obs._legend._legend_box.sep = -5  # move title down slightly
    ca_mod=sns.relplot(x=xcol_mod, y='Calcium_result_solution', data=df, hue='cl_cat_mod', hue_order=hue_order)
    ca_mod.set(xlabel=xlab_mod, ylabel=ylab_mod, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['Ca']),max(df['Calcium_result_solution'])))))
    ca_mod._legend.texts[0].set_text('')
    ca_mod._legend.set_title('   $Cl_{e, mod}$\n ($mmol_c$ $L^{-1}$)')
    ca_mod._legend._legend_box.sep = -5  # move title down slightly
    if save==True:
        ca_obs.savefig(directory+'\\'+'Ca_obs v EC_obs.png')
        ca_mod.savefig(directory+'\\'+'Ca_mod v EC_mod.png')
#Magnesium plot
def mg_plots(df, ycol, xcol, xcol_mod, xlab, xlab_mod, ylab, ylab_mod, hue_order, save, directory):
    mg_obs=sns.relplot(x=xcol, y=ycol, data=df, hue='cl_cat', hue_order=hue_order)
    mg_obs.set(xlabel=xlab, ylabel=ylab, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['Mg']),max(df['Magnesium_result_solution'])))))
    mg_obs._legend.texts[0].set_text('')
    mg_obs._legend.set_title('   $Cl_e$\n ($mmol_c$ $L^{-1}$)')
    mg_obs._legend._legend_box.sep = -5  # move title down slightly
    mg_mod=sns.relplot(x=xcol_mod, y='Magnesium_result_solution', data=df, hue='cl_cat_mod', hue_order=hue_order)
    mg_mod.set(xlabel=xlab_mod, ylabel=ylab_mod, xlim=(0,myround(max(max(df[xcol]),max(df[xcol_mod])))), ylim=(0,myround(max(max(df['Mg']),max(df['Magnesium_result_solution'])))))
    mg_mod._legend.texts[0].set_text('')
    mg_mod._legend.set_title('   $Cl_{e, mod}$\n ($mmol_c$ $L^{-1}$)')
    mg_mod._legend._legend_box.sep = -5  # move title down slightly
    if save==True:
        mg_obs.savefig(directory+'\\'+'Mg_obs v EC_obs.png')
        mg_mod.savefig(directory+'\\'+'Mg_mod v EC_mod.png')
#HCO3 1:1 plot for calibrating co2 partial pressure
def bicarb_plot(df,
    xcol = 'HCO3',
    ycol = 'Bicarbonate_result_solution',
    xlab = '$HCO_{3 pw, obs}$',
    ylab = '$HCO_{3 pw, mod}$',
    color='CO2',
    colorlab='$CO_2$ (Atm.)'):
    ax=sns.scatterplot(x=xcol, y=ycol, hue=color, data=df, legend='brief')
    ax.set(xlabel=xlab, ylabel=ylab) #plt.getp(ax) for available options
    ax.text(20, 5, 'IOA: '+str(round(gof.index_agreement(df[xcol].values, df[ycol].values), 3)))
    ax.legend(frameon=False).texts[0].set_text(colorlab) #'\n     $EC_{e}$\n $mmol_c$ $L^{-1}$')
    #ax.legend(frameon=False).texts[0].set_text('\n   $SO_{4, obs}$\n $mmol_c$ $L^{-1}$')
    #1:1 line generation:
    X_plot = np.linspace(0, myround(max(max(df[xcol]),max(df[ycol]))), 100) #x-coordinate (0,30)
    Y_plot = X_plot #y-coordinate (0,30)
    plt.plot(X_plot, Y_plot, color='black', linestyle='--')
    plt.show()
    sns.despine()
#for linear model stats
#slope, intercept, r_value, p_value, std_err = stats.linregress(ssat_all['ECpw'], ssat_all['Final EC'])
#slope_spe, intercept_spe, r_value_spe, p_value_spe, std_err_spe = stats.linregress(df['ECe'], df['Final EC'])

#################################################################Bernstein Stuff
#Bernstein Corrected ECe,g
def bern(df, water_change=False, initial_type='Paste'):
    #df['bern_ECe'] = df['ECe']-((2.2*df['PasteGWC'])/df['ssatGWC'])-2.2 #WRONG DONT USE; here for reference of old incorrect method
    df['bern_ECpw'] = (df['ECe']-2.2)/(df['ssatGWC']/df['PasteGWC']) + 2.2
    df['bern_ECe'] = (df['ECpw']-2.2)/(df['PasteGWC']/df['ssatGWC']) + 2.2
    df['deltaEC'] = 2.2-(2.2*df['ssatGWC'])/df['PasteGWC']
    df['bern_ECeg'] = df['ECe']-df['deltaEC'] #<<CORRECT
    df['Paste_linearDilution'] = df['ECpw']*(df['ssatGWC']/df['PasteGWC'])
    df['ssat_linearDilution'] = df['ECe']*(df['PasteGWC']/df['ssatGWC'])
    def func2(x):
        if x < 0:
            return 0
        else:
            return x
    df['bern_ECe']=df['bern_ECe'].apply(func2)

    #Modified Bernstein ECe,g
    df['percent_gyp']=df['Gypsum']*0.086069 #converts meq/100g to g/100g, %
    df['Mod_bern_ECe'] = df['ECe']-((2.2*df['percent_gyp']/100*df['PasteGWC'])/df['ssatGWC'])-2.2*df['percent_gyp']/100

    #Relative EC change plots
    df['EC_diff'] = ((df['ECe'] - df['Final EC'])/df['ECe'])*100
    df['bern_diff'] = df['ECe'] - df['bern_ECe']
    df['linear_diff'] = df['ECe'] - df['Paste_linearDilution']

    if water_change == True and initial_type == 'Paste':
        df['model_diff'] = df['ECpw'] - df['Final EC']
        diff_list = [df['deltaEC'], df['bern_diff'], df['linear_diff'],
                     df['model_diff']]
    elif water_change == True and initial_type == 'ssat':
        df['model_diff'] = df['ECe'] - df['Final EC']
        diff_list = [df['deltaEC'], df['bern_diff'], df['linear_diff'],
                     df['model_diff']]
    elif water_change == False and initial_type == 'Paste':
        df['model_diff'] = df['ECe'] - df['Final EC']
        diff_list = [df['deltaEC'], df['bern_diff'], df['linear_diff'],
                     df['model_diff']]
    elif water_change == False and initial_type == 'ssat':
        df['model_diff'] = df['ECpw'] - df['Final EC']
        diff_list = [df['deltaEC'], df['bern_diff'], df['linear_diff'],
                     df['model_diff']]
    return [x.describe() for x in diff_list]

#Relative Difference Plots
def EC_rel_diff_plots(df, type='Paste', gypsum=True):
    if type=='Paste':
        xcol='ECe'
        ycol='Rel_diff_ECe'
        xlab='$EC_{e}$'
        ylab='Relative Difference (%)\nBetween Measured $EC_e$ and\nModeled $EC_e$'
    else:
        xcol='ECpw'
        ycol='Rel_diff_ECpw'
        xlab='$EC_{pw}$'
        ylab='Relative Difference (%)\nBetween Measured $EC_{pw}$ and\nModeled $EC_{pw}$'
    ax=sns.scatterplot(x=xcol, y=ycol, hue='gyp_cat', data=df, hue_order=ion_levels(df)[3], legend='brief')
    ax.set(xlabel=xlab, ylabel=ylab, xlim=(0,myround(max(df['ECe']))), ylim=(-100, 500)) #plt.getp(ax) for available options
    ax.legend(frameon=False).texts[0].set_text('\n      Gypsum\n $mmol_c$ $100g^{-1}$')
    ax.axhline(0,ls='-', color='black')
    ax.axhline(df[ycol].median(),ls='--', color='red')
    #ax.axhline(-14.60,ls='--', color='blue') #lower CI for median from bootstrap in R
    #ax.axhline(25.58,ls='--', color='blue') #upper CI for median from bootstrap in R
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='r', lw=1, ls='--')],
                  labels = ['Median Difference ('+str(round(df[ycol].median(),1))+'%)'], frameon=False, loc='center right'))
    #ax.add_artist(Legend(ax, [Line2D([0], [0], color='b', lw=1, ls='--')],
    #              labels = ['95% Confidence Interval'] , frameon=False, loc=(12.5/25, 250/600)))
    if gypsum==True:
        ax.set_title('With Gypsum')
    else:
        ax.set_title('Without Gypsum')
    sns.despine()
    plt.tight_layout()

def one_to_one(df,
    xcol = 'ECe',
    ycol = 'Final EC',
    xlab = '$EC_{e, obs}$',
    ylab = '$EC_{e, mod}$',
    color='Gypsum',
    errorbar=True):
    #Gypsum
    if color=='Gypsum':
        ax=sns.scatterplot(x=xcol, y=ycol, hue='gyp_cat', data=df, hue_order=ion_levels(df)[3], legend='brief')
        ax.set(xlabel=xlab, ylabel=ylab) #plt.getp(ax) for available options
        #ax.text(max(df[xcol])*0.7, max(df[ycol])*0.2, 'IOA: '+str(round(gof.index_agreement(df[xcol].values, df[ycol].values), 3)))
        ax.legend(loc='upper left',frameon=False).texts[0].set_text('\n      Gypsum\n $mmol_c$ $100g^{-1}$')
        #1:1 line generation:
        X_plot = np.linspace(0, myround(max(max(df[xcol]),max(df[ycol]))), 100) #x-coordinate (0,30)
        Y_plot = X_plot #y-coordinate (0,30)
        plt.plot(X_plot, Y_plot, color='black', linestyle='--')
        plt.show()
        sns.despine()
        plt.tight_layout()
    #CaCO3
    elif color=='CaCo3':
        ax=sns.scatterplot(x=xcol, y=ycol, hue='caco3_cat', data=df, hue_order=ion_levels(df)[2])
        ax.set(xlabel=xlab, ylabel=ylab) #plt.getp(ax) for available options
        #ax.text(max(df[xcol])*0.7, max(df[ycol])*0.2, 'IOA: '+str(round(gof.index_agreement(df[xcol].values, df[ycol].values), 3)))
        ax.legend(loc='upper left',frameon=False).texts[0].set_text('\n      Calcite\n $mmol_c$ $100g^{-1}$')
        #1:1 line generation:
        X_plot = np.linspace(0, myround(max(max(df[xcol]),max(df[ycol]))), 100) #x-coordinate (0,30)
        Y_plot = X_plot #y-coordinate (0,30)
        plt.plot(X_plot, Y_plot, color='black', linestyle='--')
        plt.show()
        sns.despine()
        plt.tight_layout()
    #by Year
    elif color=='Year':
        ax=sns.scatterplot(x=xcol, y=ycol, hue='Year', data=df, legend='brief')
        ax.set(xlabel=xlab, ylabel=ylab) #plt.getp(ax) for available options
        #ax.text(max(df[xcol])*0.7, max(df[ycol])*0.2, 'IOA: '+str(round(gof.index_agreement(df[xcol].values, df[ycol].values), 3)))
        ax.legend(frameon=False).texts[0].set_text('Year Collected')
        #1:1 line generation:
        X_plot = np.linspace(0, myround(max(max(df[xcol]),max(df[ycol]))), 100) #x-coordinate (0,30)
        Y_plot = X_plot #y-coordinate (0,30)
        plt.plot(X_plot, Y_plot, color='black', linestyle='--')
        plt.show()
        sns.despine()
    #by Depth
    #elif color=='Depth':
    else:
        ax=sns.scatterplot(x=xcol, y=ycol, hue='Depth', data=df, legend='brief')
        ax.set(xlabel=xlab, ylabel=ylab) #plt.getp(ax) for available options
        #ax.text(max(df[xcol])*0.7, max(df[ycol])*0.2, 'IOA: '+str(round(gof.index_agreement(df[xcol].values, df[ycol].values), 3)))
        ax.legend(frameon=False).texts[0].set_text(color)
        #1:1 line generation:
        X_plot = np.linspace(0, myround(max(max(df[xcol]),max(df[ycol]))), 100) #x-coordinate (0,30)
        Y_plot = X_plot #y-coordinate (0,30)
        plt.plot(X_plot, Y_plot, color='black', linestyle='--')
        plt.show()
        sns.despine()
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='black', lw=1, ls='--')],
                  labels = ['1:1 Line'], frameon=False, loc='upper left',
                  bbox_to_anchor=(0.5, 0.25)))
    if errorbar == True:
        plt.errorbar(df[xcol], df[ycol], xerr=df[xcol]*(.005), yerr=df[ycol]*(.05),
                     fmt='none', ecolor='lightgray', capsize=0, zorder=-32)

def ion_one_to_one(df, final_type):
    xcol=['Na','Cl','SO4','Ca','Mg', 'NO3']
    ycol=['Sodium_result_solution','Chloride_result_solution',
          'Sulphate_result_solution','Calcium_result_solution',
          'Magnesium_result_solution','Nitrate_result_solution',]
    if final_type=='Paste':
        xlab = ['$Na_{e}$ ($mmol_c$ $L^{-1}$)',
                '$Cl_{e}$ ($mmol_c$ $L^{-1}$)',
                '$SO_{4, e}$ ($mmol_c$ $L^{-1}$)',
                '$Ca_{e}$ ($mmol_c$ $L^{-1}$)',
                '$Mg_{e}$ ($mmol_c$ $L^{-1}$)',
                '$NO_{3e}$ ($mmol_c$ $L^{-1}$)']
        ylab_mod = ['$Na_{e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Cl_{e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$SO_{4 e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Ca_{e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Mg_{e, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$NO_{3e, mod}$ ($mmol_c$ $L^{-1}$)']
    else:
        xlab = ['$Na_{pw}$ ($mmol_c$ $L^{-1}$)',
                '$Cl_{pw}$ ($mmol_c$ $L^{-1}$)',
                '$SO_{4, pw}$ ($mmol_c$ $L^{-1}$)',
                '$Ca_{pw}$ ($mmol_c$ $L^{-1}$)',
                '$Mg_{pw}$ ($mmol_c$ $L^{-1}$)',
                '$NO_{3pw}$ ($mmol_c$ $L^{-1}$)']
        ylab_mod = ['$Na_{pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Cl_{pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$SO_{4 pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Ca_{pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$Mg_{pw, mod}$ ($mmol_c$ $L^{-1}$)',
                    '$NO_{3pw, mod}$ ($mmol_c$ $L^{-1}$)']
    for idx, val in enumerate(ycol):
        one_to_one(df=df, xcol = xcol[idx], ycol = val, xlab = xlab[idx],
                   ylab = ylab_mod[idx], color='Gypsum')
####################################################################LINEAR MODEL
##Linear fitting
def WLS_fit(
    # Weighted least squares no-intercept model
    df,
    df2,
    xcol = 'Final EC',
    ycol = 'bern_ECeg',
    xlab = 'Selective Dilution $EC_{e, g}$, dS $m^{-1}$',
    ylab = 'Callaghan (2016) $EC_{e, g}$, dS $m^{-1}$'):
    #df2 = df2[df2['ECe']>=3.08]
    ax=sns.scatterplot(x=xcol, y=ycol, hue='gyp_cat', data=df, hue_order=ion_levels(df)[3], legend='brief')
    ax.set(xlabel=xlab, ylabel=ylab, xlim=(-1,myround(max(max(df[xcol]),max(df[ycol])))), ylim=(-1, myround(max(max(df[xcol]),max(df[ycol]))))) #plt.getp(ax) for available options
    ax.legend(frameon=False).texts[0].set_text('\n      Gypsum\n $mmol_c$ $100g^{-1}$')
    #ax.add_artist(Legend(ax, [Line2D([0], [0], color='r', lw=1, ls='--')],
    #              labels = ['Fitted Curve\n'+'$R^2$: '+str(round(gof.correlation(df2[ycol].values,df2[xcol].values)**2, 3))], frameon=False, loc='center left'))
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='r', lw=1, ls='-')],
                  labels = ['OLS Regression'], frameon=False, loc='lower right', bbox_to_anchor=(1, 0.1)))
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='b', lw=1, ls='-')],
                  labels = ['WLS Regression'], frameon=False, loc='lower right', bbox_to_anchor=(1, 0.05)))
    t = np.linspace(0, max(df2[xcol]), len(df2[xcol]))
    # OLS and WLS regression analysis
    # OLS model
    mod_ols=sm.OLS(df2[ycol], df2[xcol]).fit()
    #print(mod_ols.summary())
    #print('Slope 95% CI')
    #print(mod_ols.conf_int())
    pred_val = mod_ols.fittedvalues.copy()
    residual0 = mod_ols.resid
    std_resid = (residual0-np.mean(residual0))/np.std(residual0)
    print(sp.stats.levene(df2[xcol], residual0)) #test for homogeneity

    # OLS model to quantify variance
    residual1 = np.sqrt(mod_ols.resid**2)
    mod_ols2 = sm.OLS(residual1, pred_val).fit()
    pred_val2 = mod_ols2.fittedvalues.copy()
    residual2 = mod_ols2.resid

    # Identifying a weighting scheme
    # wts = 1/(pred_val2)
    # alternative weight calculations
    # wts = 1/np.var(residual0)
    # wts = 1/sp.stats.iqr(df[xcol]-df[ycol])

    wts = 1/df2[xcol]

    # WLS model
    mod_wls = sm.WLS(df2[ycol], df2[xcol], weights=wts).fit()
    print(mod_wls.summary())
    print('Slope 95% CI')
    print(mod_wls.conf_int())
    pred_val3 = mod_wls.fittedvalues.copy()
    residual3 = mod_wls.resid
    std_resid3 = (residual3-np.mean(residual3))/np.std(residual3)


    # 1:1 Line
    X_plot = np.linspace(0, myround(max(max(df[xcol]),max(df[ycol]))), 100) #x-coordinate (0,30)
    Y_plot = X_plot #y-coordinate (0,30)
    plt.plot(X_plot, Y_plot, color='black', linestyle='--')

    # OLS CI plotting
    plt.plot(df2[xcol], pred_val, '-', color='r')
    plt.fill_between(t, t*mod_ols.conf_int().values[0][0], t*mod_ols.conf_int().values[0][1],
                     color = 'red', alpha = 0.15)
    # WLS CI plotting
    plt.plot(df2[xcol], pred_val3, '-', color='b')
    plt.fill_between(t, t*mod_wls.conf_int().values[0][0], t*mod_wls.conf_int().values[0][1],
                     color = 'blue', alpha = 0.15)

    sns.despine()
    plt.tight_layout()

    # Standardized Residual Plots
    fig, ax2 = plt.subplots()
    ax2.scatter(y=std_resid3, x=pred_val3)
    ax2.scatter(y=std_resid, x=pred_val)
    ax2.set(xlabel='Fitted Value', ylabel='Standardized Residuals')

    fig, ax3 = plt.subplots()
    ax3.scatter(y=residual1, x=pred_val)
    ax3.set(xlabel='Fitted Value', ylabel='sqrt(residuals$^2$)')

    #Plotting errorbars
    #Hach manual says error is 0.5% of reading not range...
    #https://www.hach.com/intellical-cdc401-laboratory-4-poles-graphite-conductivity-cell-3-m-cable/product-details?id=7640489881
    plt.errorbar(df[xcol], df[ycol], xerr=df[xcol]*.005, fmt='none', ecolor='lightgray', capsize=0, zorder=-32)
        # 100 points between 0 and max, 3.08dS/m is ECe thresh for equilibrated samples

    #histogram of residuals
    #fig, ax4 = plt.subplots()
    #ax4.hist(x=residual0, bins=54)

    #print out residuals for OLS v WLS
    #print(pd.concat([residual, residual3], axis=1))

def no_int_fit(
    # no-intercept model; k-folds validation
    df,
    threshold = 3.08,
    xcol = 'Final EC',
    ycol = 'bern_ECeg',
    xlab = 'Measured $EC_{pw}$ (dS $m^{-1}$)',
    ylab = 'Simulated $EC_{pw}$ (dS $m^{-1}$)',
    eq_x = 'Measured $EC_{pw}$',
    eq_y = 'Simulated $EC_{pw}$',
    summary_plots = False,
    errorbar = True):
    df2 = df[df['ECe']>=threshold]
    ax=sns.scatterplot(x=xcol, y=ycol, hue='gyp_cat', data=df, hue_order=ion_levels(df)[3], legend='brief')
    ax.set(xlabel=xlab, ylabel=ylab, xlim=(-1,myround(max(max(df[xcol]),max(df[ycol])))), ylim=(-1, myround(max(max(df[xcol]),max(df[ycol]))))) #plt.getp(ax) for available options
    ax.legend(frameon=False).texts[0].set_text('\n      Gypsum\n $mmol_c$ $100g^{-1}$')
    t = np.linspace(0, max(df2[xcol]), len(df2[xcol]))

    #sklearn linear regression no-intercept model
    linreg = LinearRegression(fit_intercept=False)
    X_cal = df2[xcol].values.reshape(-1,1)
    Y_cal = df2[ycol].values.reshape(-1,1)
    linreg.fit(X_cal, Y_cal)
    # Calculate our y hat (how our model performs against the test data held off)
    y_hat = linreg.predict(X_cal)
    # See our Root Mean Squared Error:
    cal_mse = mean_squared_error(Y_cal, y_hat)
    cal_rmse = np.sqrt(cal_mse)
    # See our Mean Absolute Error
    cal_mae = mean_absolute_error(Y_cal, y_hat)
    # See our R^2
    r2 = r2_score(Y_cal, y_hat)
    print(f'the calibration RMSE is: {cal_rmse}')
    print(f'the calibration MAE is: {cal_mae}')
    print(f'the calibration R^2 is: {r2}')
    #K-folds cross-validation
    foldnum = 10
    n_repeats = 100
    cv = RepeatedKFold(n_splits=foldnum, n_repeats=n_repeats, random_state=2652124)
    cv_5_rmse = -1*cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='neg_root_mean_squared_error')
    mean_rmse = np.mean(cv_5_rmse)
    cv_5_mae = -1*cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='neg_mean_absolute_error')
    mean_mae = np.mean(cv_5_mae)
    cv_5_r2 = cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='r2')
    mean_r2 = np.mean(cv_5_r2)
    print(f'the average k-fold RMSE ({foldnum} folds, {n_repeats} repititions) is: {mean_rmse}')
    #print(f'inidivual values: {cv_5_rmse}')
    print(f'the k-fold MAE ({foldnum} folds, {n_repeats} repititions) is: {mean_mae}')
    #print(f'inidivual values: {cv_5_mae}')
    print(f'the k-fold R^2 ({foldnum} folds, {n_repeats} repititions) is: {mean_r2}')
    #print(f'inidivual values: {cv_5_r2}')

    # OLS model
    X_ols = df2[xcol]
    Y_ols = df2[ycol]
    mod_ols=sm.OLS(Y_ols, X_ols).fit()
    print(mod_ols.summary())
    print('Slope 95% CI')
    print(mod_ols.conf_int())
    pred_val = mod_ols.fittedvalues.copy()
    residual0 = mod_ols.resid
    std_resid = (residual0-np.mean(residual0))/np.std(residual0)
    print(sp.stats.levene(X_ols, residual0)) #test for homogeneity
    slope = mod_ols.params[0]
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='black', lw=1, ls='--')],
                  labels = ['1:1 Line'], frameon=False, loc='upper left', bbox_to_anchor=(0.35, 0.25)))
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='r', lw=1, ls='-')],
                  labels = [eq_y + ' = ' + str(round(slope,2)) + eq_x],
                  # + '\n'
                    #        '$R^2$: ' + str(round(gof.correlation(df2[ycol].values,df2[xcol].values)**2, 3))],
                             frameon=False, loc='upper left', bbox_to_anchor=(0.35, 0.2)))
    ax.text(myround(max(max(X_ols),max(Y_ols)))*0.38, myround(max(max(X_ols),max(Y_ols)))*0.05, '$R^2$: ' + str(round(gof.correlation(Y_ols.values, X_ols.values)**2, 2)))

    # 1:1 Line
    X_plot = np.linspace(0, myround(max(max(X_ols),max(Y_ols))), 100) #x-coordinate (0,30)
    Y_plot = X_plot #y-coordinate (0,30)
    plt.plot(X_plot, Y_plot, color='black', linestyle='--')

    # OLS CI plotting
    plt.plot(X_ols, pred_val, '-', color='r')
    ci_range = np.linspace(min(df2[xcol]), max(df2[xcol]), len(df2[xcol]))
    plt.fill_between(ci_range, ci_range*mod_ols.conf_int().values[0][0], ci_range*mod_ols.conf_int().values[0][1],
                     color = 'black', alpha = 0.15)

    if errorbar == True:
        #Plotting errorbars
            #X-Axis:
                #Hach manual says error is 0.5% of reading not range...
                #https://www.hach.com/intellical-cdc401-laboratory-4-poles-graphite-conductivity-cell-3-m-cable/product-details?id=7640489881
            #Y-axis:
                #McNeal, 1970 EC calculation +-5%
        plt.errorbar(X_ols, Y_ols, xerr=X_ols*(.005), yerr=Y_ols*(.05), fmt='none', ecolor='lightgray', capsize=0, zorder=-32)

    sns.despine()
    plt.tight_layout()

    if summary_plots == True:
        # Standardized Residual Plots
        # OLS model to quantify variance
        residual1 = np.sqrt(mod_ols.resid**2)
        mod_ols2 = sm.OLS(residual1, pred_val).fit()
        pred_val2 = mod_ols2.fittedvalues.copy()
        residual2 = mod_ols2.resid

        fig, ax2 = plt.subplots()
        ax2.scatter(y=std_resid, x=pred_val)
        ax2.set(xlabel='Fitted Value', ylabel='Standardized Residuals')

        fig, ax3 = plt.subplots()
        ax3.scatter(y=residual1, x=pred_val)
        ax3.set(xlabel='Fitted Value', ylabel='sqrt(residuals$^2$)')

def int_fit(
    # intercept model; k-folds validation
    df,
    threshold = 3.08,
    xcol = 'ECe',
    ycol = 'Final EC', #or bern_ECeg
    xlab = 'GC Dataset $EC_e$ (dS $m^{-1}$)',
    ylab = 'EGM $EC_{eg}$ (dS $m^{-1}$)',
    eq_x = '$EC_{e}$',
    eq_y = '$EC_{eg}$',
    summary_plots = False,
    errorbar = True):
    df2 = df[df['ECe']>=threshold]
    ax=sns.scatterplot(x=xcol, y=ycol, hue='gyp_cat', data=df, hue_order=ion_levels(df)[3], legend='brief')
    ax.set(xlabel=xlab, ylabel=ylab, xlim=(-1,myround(max(max(df[xcol]),max(df[ycol])))), ylim=(-1, myround(max(max(df[xcol]),max(df[ycol]))))) #plt.getp(ax) for available options
    ax.legend(frameon=False).texts[0].set_text('\n      Gypsum\n $mmol_c$ $100g^{-1}$')
    t = np.linspace(0, max(df2[xcol]), len(df2[xcol]))

    #sklearn linear regression intercept model
    linreg = LinearRegression(fit_intercept=True)
    X_cal = df2[xcol].values.reshape(-1,1)
    Y_cal = df2[ycol].values.reshape(-1,1)
    linreg.fit(X_cal, Y_cal)
    # Calculate our y hat (how our model performs against the test data held off)
    y_hat = linreg.predict(X_cal)
    shift = X_cal - y_hat
    avg_shift, med_shift, min_shift, max_shift = np.mean(shift), np.median(shift), np.min(shift), np.max(shift)
    print(f'The average ECe shift is: {avg_shift}')
    print(f'The median ECe shift is: {med_shift}')
    print(f'The range of ECe shifts is: {min_shift} to {max_shift}')
    # See our Root Mean Squared Error:
    cal_mse = mean_squared_error(Y_cal, y_hat)
    cal_rmse = np.sqrt(cal_mse)
    # See our Mean Absolute Error
    cal_mae = mean_absolute_error(Y_cal, y_hat)
    # See our R^2
    r2 = r2_score(Y_cal, y_hat)
    print(f'the calibration RMSE is: {cal_rmse}')
    print(f'the calibration MAE is: {cal_mae}')
    print(f'the calibration R^2 is: {r2}')
    #K-folds cross-validation
    foldnum = 10
    n_repeats = 100
    cv = RepeatedKFold(n_splits=foldnum, n_repeats=n_repeats, random_state=2652124)
    cv_5_rmse = -1*cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='neg_root_mean_squared_error')
    mean_rmse = np.mean(cv_5_rmse)
    cv_5_mae = -1*cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='neg_mean_absolute_error')
    mean_mae = np.mean(cv_5_mae)
    cv_5_r2 = cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='r2')
    mean_r2 = np.mean(cv_5_r2)
    print(f'the average k-fold RMSE ({foldnum} folds, {n_repeats} repititions) is: {mean_rmse}')
    #print(f'inidivual values: {cv_5_rmse}')
    print(f'the k-fold MAE ({foldnum} folds, {n_repeats} repititions) is: {mean_mae}')
    #print(f'inidivual values: {cv_5_mae}')
    print(f'the k-fold R^2 ({foldnum} folds, {n_repeats} repititions) is: {mean_r2}')
    #print(f'inidivual values: {cv_5_r2}')

    # OLS model
    X_ols = sm.add_constant(df2[xcol])
    Y_ols = df2[ycol]
    mod_ols = sm.OLS(Y_ols, X_ols).fit()
    print(mod_ols.summary())
    print('95% CI')
    print(mod_ols.conf_int())
    pred_val = mod_ols.fittedvalues.copy()
    residual0 = mod_ols.resid
    std_resid = (residual0-np.mean(residual0))/np.std(residual0)
    print(sp.stats.levene(df2[xcol], residual0)) #test for homogeneity
    slope = mod_ols.params[1]
    intercept = mod_ols.params[0]
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='black', lw=1, ls='--')],
                  labels = ['1:1 Line'], frameon=False, loc='upper left', bbox_to_anchor=(0.5, 0.25)))
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='r', lw=1, ls='-')],
                  labels = [eq_y + ' = ' + str(round(slope,2)) +' '+ eq_x + '\n+ ' + str(round(intercept,2))],
                             frameon=False, loc='upper left', bbox_to_anchor=(0.5, 0.2)))
    ax.text(myround(max(max(df[xcol]),max(df[ycol])))*0.25, myround(max(max(df[xcol]),max(df[ycol])))*0.05, '$R^2$: ' + str(round(gof.correlation(df2[ycol].values,df2[xcol].values)**2, 2)))

    # 1:1 Line
    X_plot = np.linspace(0, myround(max(max(df[xcol]),max(df[ycol]))), 100) #x-coordinate (0,30)
    Y_plot = X_plot #y-coordinate (0,30)
    plt.plot(X_plot, Y_plot, color='black', linestyle='--')

    # OLS CI plotting
    plt.plot(df2[xcol], pred_val, '-', color='r')
    mean_x = df2[xcol].mean()
    n = df2[xcol].count()
    dof = n - mod_ols.df_model - 1
    t_stat = sp.stats.t.ppf(1-0.025, df=dof)
    s_err = np.sum(np.power(residual0, 2))
    x_pred = np.linspace(df2[xcol].min(), df2[xcol].max(), df2[xcol].count())
    conf = t_stat * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((df2[xcol]-mean_x),2) / ((np.sum(np.power(df2[xcol],2))) - n*(np.power(mean_x,2))))))
    lower = pred_val - abs(conf)
    upper = pred_val + abs(conf)
    x, u, l = sorted(df2[xcol]), sorted(upper), sorted(lower)
    plt.fill_between(x, l, u, color = 'black', alpha = 0.15, interpolate=True)

    if errorbar == True:
        #Plotting errorbars
            #X-Axis:
                #Hach manual says error is 0.5% of reading not range...
                #https://www.hach.com/intellical-cdc401-laboratory-4-poles-graphite-conductivity-cell-3-m-cable/product-details?id=7640489881
            #Y-axis:
                #McNeal, 1970 EC calculation +-5%
        plt.errorbar(df[xcol], df[ycol], xerr=df[xcol]*.005, yerr=df[ycol]*(.05), fmt='none', ecolor='lightgray', capsize=0, zorder=-32)

    sns.despine()
    plt.tight_layout()

    if summary_plots == True:
        # Standardized Residual Plots
        # OLS model to quantify variance
        residual1 = np.sqrt(mod_ols.resid**2)
        mod_ols2 = sm.OLS(residual1, pred_val).fit()
        pred_val2 = mod_ols2.fittedvalues.copy()
        residual2 = mod_ols2.resid
        fig, ax2 = plt.subplots()
        ax2.scatter(y=std_resid, x=pred_val)
        ax2.set(xlabel='Fitted Value', ylabel='Standardized Residuals')

        fig, ax3 = plt.subplots()
        ax3.scatter(y=residual1, x=pred_val)
        ax3.set(xlabel='Fitted Value', ylabel='sqrt(residuals$^2$)')

def fixed_int_fit(
    # intercept model; k-folds validation
    df,
    threshold = 3.08,
    xcol = 'ECe',
    ycol = 'Final EC',
    xlab = '$EC_{e, obs}$, dS $m^{-1}$',
    ylab = 'Callaghan (2016) $EC_{e, g}$, dS $m^{-1}$',
    summary_plots = False,
    errorbar = True):
    df2 = df[df['ECe']>=threshold]
    ax=sns.scatterplot(x=xcol, y=ycol, hue='gyp_cat', data=df, hue_order=ion_levels(df)[3], legend='brief')
    ax.set(xlabel=xlab, ylabel=ylab, xlim=(-1,myround(max(max(df[xcol]),max(df[ycol])))), ylim=(-1, myround(max(max(df[xcol]),max(df[ycol]))))) #plt.getp(ax) for available options
    ax.legend(frameon=False).texts[0].set_text('\n      Gypsum\n $mmol_c$ $100g^{-1}$')
    t = np.linspace(0, max(df2[xcol]), len(df2[xcol]))

    #sklearn linear regression intercept model
    linreg = LinearRegression(fit_intercept=True)
    X_cal = df2[xcol].values.reshape(-1,1)
    Y_cal = df2[ycol].values.reshape(-1,1)
    linreg.fit(X_cal, Y_cal)
    # Calculate our y hat (how our model performs against the test data held off)
    y_hat = linreg.predict(X_cal)
    shift = X_cal - y_hat
    avg_shift, med_shift, min_shift, max_shift = np.mean(shift), np.median(shift), np.min(shift), np.max(shift)
    print(f'The average ECe shift is: {avg_shift}')
    print(f'The median ECe shift is: {med_shift}')
    print(f'The range of ECe shifts is: {min_shift} to {max_shift}')
    # See our Root Mean Squared Error:
    cal_mse = mean_squared_error(Y_cal, y_hat)
    cal_rmse = np.sqrt(cal_mse)
    # See our Mean Absolute Error
    cal_mae = mean_absolute_error(Y_cal, y_hat)
    # See our R^2
    r2 = r2_score(Y_cal, y_hat)
    print(f'the calibration RMSE is: {cal_rmse}')
    print(f'the calibration MAE is: {cal_mae}')
    print(f'the calibration R^2 is: {r2}')
    #K-folds cross-validation
    foldnum = 10
    n_repeats = 100
    cv = RepeatedKFold(n_splits=foldnum, n_repeats=n_repeats, random_state=2652124)
    cv_5_rmse = -1*cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='neg_root_mean_squared_error')
    mean_rmse = np.mean(cv_5_rmse)
    cv_5_mae = -1*cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='neg_mean_absolute_error')
    mean_mae = np.mean(cv_5_mae)
    cv_5_r2 = cross_val_score(linreg, X_cal, Y_cal, cv=cv,
     scoring='r2')
    mean_r2 = np.mean(cv_5_r2)
    print(f'the average k-fold RMSE ({foldnum} folds, {n_repeats} repititions) is: {mean_rmse}')
    #print(f'inidivual values: {cv_5_rmse}')
    print(f'the k-fold MAE ({foldnum} folds, {n_repeats} repititions) is: {mean_mae}')
    #print(f'inidivual values: {cv_5_mae}')
    print(f'the k-fold R^2 ({foldnum} folds, {n_repeats} repititions) is: {mean_r2}')
    #print(f'inidivual values: {cv_5_r2}')

    # OLS model
    X_ols = sm.add_constant(df2[xcol])
    Y_ols = df2[ycol]
    df_ols = df2[[xcol, ycol]]
    df_ols.columns=['ECe', 'ECeg']

    print(df_ols)
    mod_ols = smf.ols(formula='ECeg - c * ECe ~ 1', data=df_ols).fit()
    print(mod_ols.summary())
    print('95% CI')
    print(mod_ols.conf_int())
    pred_val = mod_ols.fittedvalues.copy()
    residual0 = mod_ols.resid
    std_resid = (residual0-np.mean(residual0))/np.std(residual0)
    print(sp.stats.levene(df2[xcol], residual0)) #test for homogeneity
    intercept = mod_ols.params[0]
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='black', lw=1, ls='--')],
                  labels = ['1:1 Line'], frameon=False, loc='upper left', bbox_to_anchor=(0.5, 0.25)))
    ax.add_artist(Legend(ax, [Line2D([0], [0], color='r', lw=1, ls='-')],
                  labels = ['$EC_{e, g}$ = ' + ' $EC_{e, obs}$' + ' + ' + str(round(intercept,4))],
                             frameon=False, loc='upper left', bbox_to_anchor=(0.5, 0.2)))
    ax.text(myround(max(max(df[xcol]),max(df[ycol])))*0.58,
            myround(max(max(df[xcol]),max(df[ycol])))*0.05,
            '$R^2$: ' + str(round(gof.correlation(df2[ycol].values,
                                                  df2[xcol].values)**2, 3)))

    # 1:1 Line
    X_plot = np.linspace(0, myround(max(max(df[xcol]),max(df[ycol]))), 100) #x-coordinate (0,30)
    Y_plot = X_plot #y-coordinate (0,30)
    plt.plot(X_plot, Y_plot, color='black', linestyle='--')

    # OLS CI plotting
    plt.plot(df2[xcol], pred_val, '-', color='r')
    mean_x = df2[xcol].mean()
    n = df2[xcol].count()
    dof = n - mod_ols.df_model - 1
    t_stat = sp.stats.t.ppf(1-0.025, df=dof)
    s_err = np.sum(np.power(residual0, 2))
    x_pred = np.linspace(df2[xcol].min(), df2[xcol].max(), df2[xcol].count())
    conf = t_stat * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((df2[xcol]-mean_x),2) / ((np.sum(np.power(df2[xcol],2))) - n*(np.power(mean_x,2))))))
    lower = pred_val - abs(conf)
    upper = pred_val + abs(conf)
    x, u, l = sorted(df2[xcol]), sorted(upper), sorted(lower)
    plt.fill_between(x, l, u, color = 'black', alpha = 0.15, interpolate=True)

    if errorbar == True:
        #Plotting errorbars
            #X-Axis:
                #Hach manual says error is 0.5% of reading not range...
                #https://www.hach.com/intellical-cdc401-laboratory-4-poles-graphite-conductivity-cell-3-m-cable/product-details?id=7640489881
            #Y-axis:
                #McNeal, 1970 EC calculation +-5%
        plt.errorbar(df[xcol], df[ycol], xerr=df[xcol]*.005, yerr=df[ycol]*(.05), fmt='none', ecolor='lightgray', capsize=0, zorder=-32)

    sns.despine()
    plt.tight_layout()

    if summary_plots == True:
        # Standardized Residual Plots
        # OLS model to quantify variance
        residual1 = np.sqrt(mod_ols.resid**2)
        mod_ols2 = sm.OLS(residual1, pred_val).fit()
        pred_val2 = mod_ols2.fittedvalues.copy()
        residual2 = mod_ols2.resid
        fig, ax2 = plt.subplots()
        ax2.scatter(y=std_resid, x=pred_val)
        ax2.set(xlabel='Fitted Value', ylabel='Standardized Residuals')

        fig, ax3 = plt.subplots()
        ax3.scatter(y=residual1, x=pred_val)
        ax3.set(xlabel='Fitted Value', ylabel='sqrt(residuals$^2$)')


####################################################################logfit graph
#obsolete, no longer in use, but here for record
'''
#df['diff_predicted'] = -0.479830314485866*np.log(df['ECe'])+1.04517144071876 #found in excel
#popt, pcov = curve_fit(lambda fx,a,b,c: a*(fx**-b)+c,  df['ECe'].drop([6,51]),  df['Rel_diff_ECe'].drop([6,51]))
#popt, pcov = curve_fit(lambda fx,a,b,c: a*(fx**-b)+c,  df['ECe'],  df['Rel_diff_ECe'])
#df['diff_predicted'] = popt[0]*(df['ECe']**-popt[1])+popt[2]
#df['ECe_log_pred'] =  df['ECe']*(1+df['diff_predicted']/100)
#df['Linear'] = 0.491283281*df['ECe']+0.611544701
##With Gypsum ECe v. Relative %
#ax=sns.scatterplot(x='ECe', y='Rel_diff_ECe', hue='gyp_cat', data=df.drop([6,51]), hue_order=gyp_hue, legend='brief')
#ax.set(xlabel='$EC_{e}$', ylabel='Relative Difference (%)\nBetween Measured $EC_e$ and\nModeled $EC_{eg}$',
#       xlim=(0,25), ylim=(-100, 500)) #plt.getp(ax) for available options
#ax.legend(frameon=False).texts[0].set_text('\n      Gypsum\n $mmol_c$ $100g^{-1}$')
#ax.axhline(0,ls='-', color='black')
#ax.add_artist(Legend(ax, [Line2D([0], [0], color='r', lw=1, ls='--')],
#              labels = ['Fitted Curve'], frameon=False, loc='center right'))
#t = np.linspace(min(df['ECe']), max(df['ECe']), 100)    # 100 points between min and max
#fit = np.polyfit(df['Rel_diff_ECe'].drop([6,51])/100, np.log(df['ECe'].drop([6,51])), 1)#, w=np.sqrt(abs(df['Rel_diff_ECe'].drop([6,51]))))
#y = (fit[0]*np.log(t)+fit[1])*100        # allocate y with float elements
#popt, pcov = curve_fit(lambda fx,a,b,c: a*(fx**-b)+c,  df['ECe'].drop([6,51]),  df['Rel_diff_ECe'].drop([6,51])/100)
#power_y = popt[0]*(t**-popt[1])+popt[2]
#plt.plot(t, power_y*100, '--', color='r')
#sns.despine()
#plt.tight_layout()
#print(gof.rmse(df['SSAT to Paste'].values, df['ECe_log_pred'].values))
#RMSE table

ycol = ['Callaghan', 'bern_ECe', 'Linear']
for i in ycol:
    print(i+':')
    print(gof.rmse(df['SSAT to Paste Final no Gyp'].values, df[i].values))


#RMSPE calculation for linear calibration in R
rmspe = c(0)
for(j in 1:length(x)) {
    X = x[-j]
    Y = y[-j]
    mdl = lm(Y ~ X) #<<< this is where I'd add code to average model intercepts and such
    r2 = summary(mdl)$r.squared
    df = data.frame(X = x[j], Y = y[j])
    y_pred <- predict(mdl, df)
    rmspe <- sqrt(rmspe + (y[j] - y_pred)^2)
  }

rmspe

fdd2=fdd[fdd['Paste_ECe']>0.451 and fdd['Paste_GWC']>0.317 and fdd['Field_GWC']>0.451]
'''
