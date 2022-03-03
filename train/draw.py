import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def draw_comparison(tr, val, label, best_x, title='', x_label='Epochs', y_label='Loss', name='plot.png', log_scale=False):
    fig, ax = plt.subplots()
    if log_scale:
        ax.set_yscale("log")
    ax.plot(tr, color="red", label=label[0])
    ax.plot(val, color="blue", label=label[1])
    plt.axvline(x=best_x, linestyle="--")
    plt.text(best_x, val[best_x], "i="+str(best_x), rotation=90)
    plt.legend(loc="best")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def draw_comparison_metrics(tr_m, val_m, label, best_x, title='', x_label='Epochs', name='plot.png', log_scale=False):

    tr_values = {x: [] for x in tr_m[0].keys() if "matrix" not in x}
    val_values = {x: [] for x in tr_m[0].keys() if "matrix" not in x}
    for tr_eval, val_eval in zip(tr_m, val_m):
        for i in tr_values.keys():
            tr_values[i].append(tr_eval[i])
            val_values[i].append(val_eval[i])

    for i in tr_values.keys():
        fig, ax = plt.subplots(figsize=(12, 7))
        if log_scale:
            ax.set_yscale("log")
        ax.plot(tr_values[i], color="red", label=label[0])
        ax.plot(val_values[i], color="blue", label=label[1])
        ax.set_xlabel(x_label)
        ax.set_ylabel(i)
        ax.set_title(title)
        plt.axvline(x=best_x, linestyle="--")
        plt.text(best_x, val_values[i][best_x], "i="+str(best_x), rotation=90)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(i + "_" + name)
        plt.close()
