from matplotlib import pyplot as pp
import numpy as np

#data = [[float(n) for n in line.split()] for line in open("final_results").readlines()]

def plot_data():
    colors = ["blue", "green", "gray", "cyan"]

    ind = np.arange(4)

    gao_vals = [.76, .74, .67, .72]
    vn_vals = [.77, .75, .68, .73]
    syn_vals = [.77, .75, .68, .73]
    both_vals = [.79, .76, .69, .74]

    axes = pp.gca()
    axes.set_ylim([.65, .8])
    rects = []
    rects.append(pp.bar(ind, gao_vals, width=.2, capsize=5))
    rects.append(pp.bar([i+.2 for i in ind], vn_vals, width=.2, capsize=5))
    rects.append(pp.bar([i+.4 for i in ind], syn_vals, width=.2, capsize=5))
    rects.append(pp.bar([i+.6 for i in ind], both_vals, width=.2, capsize=5))

    pp.xticks([i+.3 for i in ind], ("MOH-X (10 fold)", "TroFi (10 Fold)", "VUA (All)", "VUA (Verbs)"), fontsize=14)
    pp.ylabel("F1", fontsize=14)

#    sig_rects = [rects[3][0], rects[1][0], rects[2][0]]

#    for r in sig_rects:
#        pp.text(r.get_x() + r.get_width()/2. - .05, 1.*r.get_height(), "*", fontsize=24)

    pp.legend(labels=["Gao et al Repr.", "+VN Data", "+Syn Data", "+Both"], loc="best")
    pp.grid()
    pp.show()


plot_data()
