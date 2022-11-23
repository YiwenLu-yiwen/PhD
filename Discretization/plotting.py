import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

def read_file(files_list, exp1path):
    dic = {}
    for each in files_list:
        df = pd.read_csv(exp1path + each)
        dic[each[5:-4]] = df
    return dic

def grabData(dic, typ='accuracy'):
    indx_lst = list(map(int, list(dic.keys())))
    indx_lst = sorted(indx_lst)
    label_lst = list(dic['20'].iloc[:, 0].values)
    total_list = []
    for key in indx_lst:
        values = dic[str(key)]
        total_list.append(values[typ].values.tolist())
    if len(total_list) == 5:
        a,b,c,d,e = total_list
        values_lst = list(map(list, list(zip(a,b,c,d,e))))
    if len(total_list) == 3:
        a,b,c = total_list
        values_lst = list(map(list, list(zip(a,b,c))))
    if len(total_list) == 4 : # error bar 17/11
        a = list(map(float,total_list[0][0][1:-1].split(',')))
        b = list(map(float,total_list[0][1][1:-1].split(',')))

        c = list(map(float,total_list[1][0][1:-1].split(',')))
        d = list(map(float,total_list[1][1][1:-1].split(',')))

        e = list(map(float,total_list[2][0][1:-1].split(',')))
        f = list(map(float,total_list[2][1][1:-1].split(',')))

        g = [float(total_list[3][0]), float(total_list[3][0])]
        h = [float(total_list[3][1]), float(total_list[3][1])]
        values_lst = [(a,c,e,g), (b,d,f,h)]
    return values_lst, indx_lst, label_lst


def plotDf(files_list, exp1path, ax1, ax2, ax3, name=None, legend=True, title=True, xalbel_name=True, fig=False):
    # fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, sharex=True, figsize=(16,5))
    dic = read_file(files_list, exp1path)
    if title:
        ax1.set_title('Recall')
        ax2.set_title('Precision')
        ax3.set_title('F1')

    values_lst, indx_lst, label_lst = grabData(dic=dic, typ='recall')
    # rp_value, kd_value, classification_value, honest_value, mi_value, linear_value, joint_value = \
    #     values_lst

    # rp_value, kd_value, classification_value, honest_value, joint_value = \
    #     values_lst
    # ax1.plot(indx_lst, rp_value, 'b-', marker='o', label='RP')
    # ax1.plot(indx_lst, kd_value, 'g-', marker='*', label='KD')
    # ax1.plot(indx_lst, classification_value, 'y', marker='v', label='Clf')
    # ax1.plot(indx_lst, honest_value, 'k', marker='x', label='Honest')
    # ax1.plot(indx_lst, mi_value, 'r', marker="^", label='MI')
    # ax1.plot(indx_lst, linear_value, 'm', marker="s", label='Logist')
    stage_value, joint_value = values_lst

    stage_mean_value = list(map(np.mean, stage_value))
    joint_mean_value = list(map(np.mean, joint_value))
    stage_std_error = np.array(list(map(np.std, stage_value)))/math.sqrt(len(stage_value[0]))
    joint_std_error = np.array(list(map(np.std, joint_value)))/math.sqrt(len(stage_value[0]))

    ax1.errorbar(indx_lst, stage_mean_value, marker="s", yerr = stage_std_error, label='Stage')
    ax1.errorbar(indx_lst, joint_mean_value, marker="P", yerr = joint_std_error, label='Joint')
    ax1.grid()
    if legend:
        ax1.legend()
    ax1.set_xscale('log')


    values_lst, indx_lst, label_lst = grabData(dic=dic, typ='precision')
# rp_value, kd_value, classification_value, honest_value, mi_value, linear_value, joint_value = \
    #     values_lst

    # rp_value, kd_value, classification_value, honest_value, joint_value = \
    #     values_lst
    stage_value, joint_value = values_lst

    stage_mean_value = list(map(np.mean, stage_value))
    joint_mean_value = list(map(np.mean, joint_value))
    stage_std_error = np.array(list(map(np.std, stage_value)))/math.sqrt(len(stage_value[0]))
    joint_std_error = np.array(list(map(np.std, joint_value)))/math.sqrt(len(stage_value[0]))

    ax2.errorbar(indx_lst, stage_mean_value, marker="s", yerr = stage_std_error, label='Stage')
    ax2.errorbar(indx_lst, joint_mean_value, marker="P", yerr = joint_std_error, label='Joint')

    # ax2.plot(indx_lst, rp_value, 'b-', marker='o', label='RP')
    # ax2.plot(indx_lst, kd_value, 'g-', marker='*', label='KD')
    # ax2.plot(indx_lst, classification_value, 'y', marker='v', label='Clf')
    # ax2.plot(indx_lst, honest_value, 'k', marker='x', label='Honest')
    # # ax2.plot(indx_lst, mi_value, 'r', marker="^", label='MI')
    # # ax2.plot(indx_lst, linear_value, 'm', marker="s", label='Logist')
    # ax2.plot(indx_lst, joint_value, 'c', marker="P", label='Joint')
    ax2.grid()


    values_lst, indx_lst, label_lst = grabData(dic=dic, typ='f1')
    # rp_value, kd_value, classification_value, honest_value, mi_value, linear_value, joint_value = \
    #     values_lst

    stage_value, joint_value = values_lst
    stage_mean_value = list(map(np.mean, stage_value))
    joint_mean_value = list(map(np.mean, joint_value))
    stage_std_error = np.array(list(map(np.std, stage_value)))/math.sqrt(len(stage_value[0]))
    joint_std_error = np.array(list(map(np.std, joint_value)))/math.sqrt(len(stage_value[0]))

    ax3.errorbar(indx_lst, stage_mean_value, marker="s", yerr = stage_std_error, label='Stage')
    ax3.errorbar(indx_lst, joint_mean_value, marker="P", yerr = joint_std_error, label='Joint')


    # rp_value, kd_value, classification_value, honest_value, joint_value = \
    #     values_lst
    # ax3.plot(indx_lst, rp_value, 'b-', marker='o', label='RP')
    # ax3.plot(indx_lst, kd_value, 'g-', marker='*', label='KD')
    # ax3.plot(indx_lst, classification_value, 'y', marker='v', label='Clf')
    # ax3.plot(indx_lst, honest_value, 'k', marker='x', label='Honest')
    # # ax3.plot(indx_lst, mi_value, 'r', marker="^", label='MI')
    # # ax3.plot(indx_lst, linear_value, 'm', marker="s", label='Logist')
    # ax3.plot(indx_lst, joint_value, 'c', marker="P", label='Joint')
    ax3.grid()

    if xalbel_name:
        fig.supxlabel('sample size')
    # plt.savefig(name + '.pdf', bbox_inches='tight', pad_inches=0)

def plot_2Discretization(data, best_subsetData_list, dim_list, best_value_list,title_name=None, ax=None):
    ax = plt.gca() if ax is None else ax
    xx1, xx2 = [], []
    _class = []
    for each_dic in best_subsetData_list:
        xx1 += list(each_dic['values'][:, 0])
        xx2 += list(each_dic['values'][:, 1])
        _class  += list(each_dic['values'][:, -1])
    # scatterplot
    plotShown = ax.scatter(x=xx1, y=xx2, c=_class)#, cmap="summer")

    best_value_dic = {0:[], 1:[]}
    for i in range(len(dim_list)):
        best_value_dic[dim_list[i]].append(best_value_list[i])
    # best_value_dic
    for key in best_value_dic:
        if key == 0:
            for each in best_value_dic[key]:
                ax.axvline(x=each)
        elif key == 1:
            for each in best_value_dic[key]:
                ax.axhline(y=each)
    ax.set_title(title_name)
    ax.set_xlabel('$X_0$')
    ax.set_ylabel('$X_1$')

def plot_2Discretization_Step(data, best_subsetData_list, dim_list, best_value_list, best_fmi_list, title_name=None, ncols=3, figsize=(12, 10), sharey='row'):
    indx = 0
    nrow = math.ceil(len(dim_list)/ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrow, figsize=figsize, sharey=sharey, tight_layout=True)
    for j in range(nrow):
        for k in range(ncols):
            step_dim_list = dim_list[:indx]
            step_value_list = best_value_list[:indx]
            step_fmi = best_fmi_list[indx]
            previous_fmi = best_fmi_list[indx-1] if indx !=0 else 0
            ax = axs[j, k] if nrow !=1 else axs[k]
            plot_2Discretization(data, best_subsetData_list, step_dim_list, step_value_list, title_name=None, ax=ax)
            if dim_list[indx]==1:
                ax.axhline(y=best_value_list[indx], color='r')
            else:
                ax.axvline(x=best_value_list[indx], color='r')

            text = str(round(step_fmi, 4)) + '\n+(' + str(round(step_fmi-previous_fmi, 4)) + ")"
            ax.text(1, 0.95, text, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            indx += 1
            if indx >= len(best_fmi_list):
                break
    fig.suptitle(title_name)
    fig.supxlabel(r'$X_0$')
    fig.supylabel(r'$X_1$')