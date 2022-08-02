import pandas as pd
import matplotlib.pyplot as plt

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
    return values_lst, indx_lst, label_lst


def plotDf(files_list, exp1path, ax1, ax2, ax3, name=None, legend=False, title=False, xalbel_name=False, fig=False):
    # fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, sharex=True, figsize=(16,5))
    dic = read_file(files_list, exp1path)
    if title:
        ax1.set_title('Recall')
        ax2.set_title('Precision')
        ax3.set_title('F1')

    values_lst, indx_lst, label_lst = grabData(dic=dic, typ='recall')
    rp_value, kd_value, classification_value, honest_value, mi_value, linear_value = \
        values_lst
    ax1.plot(indx_lst, rp_value, 'b-', marker='o', label='RP')
    ax1.plot(indx_lst, kd_value, 'g-', marker='*', label='KD')
    ax1.plot(indx_lst, classification_value, 'y', marker='v', label='Clf')
    ax1.plot(indx_lst, honest_value, 'k', marker='x', label='Honest')
    ax1.plot(indx_lst, mi_value, 'r', marker="^", label='MI')
    ax1.plot(indx_lst, linear_value, 'm', marker="s", label='Logist')
    ax1.grid()
    if legend:
        ax1.legend()
    ax1.set_xscale('log')


    values_lst, indx_lst, label_lst = grabData(dic=dic, typ='precision')
    rp_value, kd_value, classification_value, honest_value, mi_value, linear_value = \
        values_lst
    ax2.plot(indx_lst, rp_value, 'b-', marker='o', label='RP')
    ax2.plot(indx_lst, kd_value, 'g-', marker='*', label='KD')
    ax2.plot(indx_lst, classification_value, 'y', marker='v', label='Clf')
    ax2.plot(indx_lst, honest_value, 'k', marker='x', label='Honest')
    ax2.plot(indx_lst, mi_value, 'r', marker="^", label='MI')
    ax2.plot(indx_lst, linear_value, 'm', marker="s", label='Logist')
    ax2.grid()


    values_lst, indx_lst, label_lst = grabData(dic=dic, typ='f1')
    rp_value, kd_value, classification_value, honest_value, mi_value, linear_value = \
        values_lst
    ax3.plot(indx_lst, rp_value, 'b-', marker='o', label='RP')
    ax3.plot(indx_lst, kd_value, 'g-', marker='*', label='KD')
    ax3.plot(indx_lst, classification_value, 'y', marker='v', label='Clf')
    ax3.plot(indx_lst, honest_value, 'k', marker='x', label='Honest')
    ax3.plot(indx_lst, mi_value, 'r', marker="^", label='MI')
    ax3.plot(indx_lst, linear_value, 'm', marker="s", label='Logist')
    ax3.grid()

    if xalbel_name:
        fig.supxlabel('sample size')
    # plt.savefig(name + '.pdf', bbox_inches='tight', pad_inches=0)