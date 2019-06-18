from matplotlib import pyplot as pp
import numpy as np


moh_svm = [.5761, .0448]
trofi_svm = [.7751, .0014]
vua_all_svm = [.5285, .0072]
vua_verb_svm = [.5527, .0107]

moh_svm_deps = [.6486, .0439]
trofi_svm_deps = [.7946, .0014]
vua_all_svm_deps = [.5303, .0074]
vua_verb_svm_deps = [.5767, .0104]

moh_normal = [.6521, .0447]
moh_vn = [.6816, .0537]
moh_syn = [.705, .0488]
moh_both = [.6835, .0509]

trofi_normal = [.6554, .0209]
trofi_vn = [.6709, .0195]
trofi_syn = [.6708, .0196]
trofi_both = [.6847, .0177]

gao_fivetwo = [(.734, .002), (.696, .003)]
vn_fivetwo = [(.736, .003), (.698, .007)]
syn_fivetwo = [(.736, .004), (.699, .006)]
both_fivetwo = [(.733, .004), (.696, .004)]

gao_ten = [(.7308, .0029), (.6829, .0074)]
vn_ten = [(.7379, .0027), (.6933, .0054)]
syn_ten = [(.7385, .0046), (.692, .0075)]
both_ten = [(.7375, .0031), (.6926, .0056)]

gao_bs = [(.732, .006), (.691, .009)]
vn_bs = [(.74, .005), (.701, .009)]
syn_bs = [(.739, .005), (.699, .009)]
both_bs = [(.739, .005), (.705, .009)]

gao_cls_ten = [[.6669, .0022]]
vn_cls_ten = [[.6719, .0012]]
syn_cls_ten = [[.6780, .0016]]
both_cls_ten = [[.6805, .0009]]

gao_cls_bs = [[.6400, .0099]]
vn_cls_bs = [[.6816, .0086]]
syn_cls_bs = [[.6974, .0085]]
both_cls_bs = [[.6600, .0098]]

gao_cls_fivetwo = [[0,0]]
vn_cls_fivetwo = [[0,0]]
syn_cls_fivetwo = [[0,0]]
both_cls_fivetwo = [[0,0]]


def plot_svm():
    baseline_vals = [moh_svm[0], trofi_svm[0], vua_all_svm[0], vua_verb_svm[0]]
    dep_vals = [moh_svm_deps[0], trofi_svm_deps[0], vua_all_svm_deps[0], vua_verb_svm_deps[0]]
    baseline_errs = [moh_svm[1], trofi_svm[1], vua_all_svm[1], vua_verb_svm[1]]
    dep_errs = [moh_svm_deps[1], trofi_svm_deps[1], vua_all_svm_deps[1], vua_verb_svm_deps[1]]
    
    ind = np.arange(4)
    
    axes = pp.gca()
    axes.set_ylim([.5, .82])
    rects = []
    rects.append(pp.bar(ind, baseline_vals, width=.2, capsize=5, edgecolor="black", linewidth=.4))
    rects.append(pp.bar([i+.2 for i in ind], dep_vals, width=.2, capsize=5, edgecolor="black", linewidth=.4))

    pp.xticks([i+.3 for i in ind], ("MOH-X", "Trofi", "VUA All", "VUA Verbs"), fontsize=14)
    pp.ylabel("F1", fontsize=14)

    pp.errorbar(ind, baseline_vals, baseline_errs, linestyle="None", color="black", capsize=3, linewidth=.6)
    pp.errorbar([i+.2 for i in ind], dep_vals, dep_errs, linestyle="None", color="black", capsize=3, linewidth=.6)

    
    pp.legend(labels=["Baseline", "+Dependency Features"], loc="best")
    axes.yaxis.grid()
    pp.show()


def plot_data():
    colors = ["blue", "green", "gray", "cyan"]

    ind = np.arange(5)
    # MOH-X, Trofi, VUA all, VUA verbs
    gao_vals = [moh_normal[0], trofi_normal[0], gao_cls_ten[0][0], gao_ten[1][0], gao_ten[0][0]]
    vn_vals = [moh_vn[0], trofi_vn[0], vn_cls_ten[0][0], vn_ten[1][0], vn_ten[0][0]]
    syn_vals = [moh_syn[0], trofi_syn[0], syn_cls_ten[0][0], syn_ten[1][0], syn_ten[0][0]]
    both_vals = [moh_both[0], trofi_both[0], both_cls_ten[0][0], both_ten[1][0], both_ten[0][0]]

    gao_errs = [moh_normal[1], trofi_normal[1], gao_cls_ten[0][1], gao_ten[1][1], gao_ten[0][1]]
    vn_errs = [moh_vn[1], trofi_vn[1], vn_cls_ten[0][1], vn_ten[1][1], vn_ten[0][1]]
    syn_errs = [moh_syn[1], trofi_syn[1], syn_cls_ten[0][1], syn_ten[1][1], syn_ten[0][1]]
    both_errs = [moh_both[1], trofi_both[1], both_cls_ten[0][1], both_ten[1][1], both_ten[0][1]]
    
    axes = pp.gca()
    axes.set_ylim([.6, .755])
    rects = []
    rects.append(pp.bar(ind, gao_vals, width=.2, capsize=5, edgecolor="black", linewidth=.4))
    rects.append(pp.bar([i+.2 for i in ind], vn_vals, width=.2, capsize=5, edgecolor="black", linewidth=.4))
    rects.append(pp.bar([i+.4 for i in ind], syn_vals, width=.2, capsize=5, edgecolor="black", linewidth=.4))
    rects.append(pp.bar([i+.6 for i in ind], both_vals, width=.2, capsize=5, edgecolor="black", linewidth=.4))

    pp.xticks([i+.3 for i in ind], ("MOH-X", "Trofi", "VUA CLS", "VUA SEQ (Verbs)", "VUA SEQ (All)"), fontsize=14)
    pp.ylabel("F1", fontsize=14)

    pp.errorbar(ind, gao_vals, gao_errs, linestyle="None", color="black", capsize=3, linewidth=.6)
    pp.errorbar([i+.2 for i in ind], vn_vals, vn_errs, linestyle="None", color="black", capsize=3, linewidth=.6)
    pp.errorbar([i+.4 for i in ind], syn_vals, syn_errs, linestyle="None", color="black", capsize=3, linewidth=.6)
    pp.errorbar([i+.6 for i in ind], both_vals, both_errs, linestyle="None", color="black", capsize=3, linewidth=.6)

    
    pp.legend(labels=["Gao et al Repr.", "+VN Data", "+Syn Data", "+Both"], loc="best")
    axes.yaxis.grid()
    pp.show()


#plot_all(cls=True)
#plot_all(verbs=True)
#plot_all()
plot_svm()
