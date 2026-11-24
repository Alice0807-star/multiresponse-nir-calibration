"""Round 3: complete curves with comparable scales, plus a selection index."""
import argparse
import string
from matplotlib.backends.backend_pdf import PdfPages
from plot_common import *


def limits(source,target=None,tail=False):
    records=[r for r in CURVES if r["source"]==source and (target is None or r["target"]==target)]
    values=[v["mean_r2"] for r in records for key in ["A_trials","union_trials"] for v in r[key]
            if not tail or v["components"]>=8]
    lo,hi=min(values),max(values); padding=.05*(hi-lo)
    return lo-padding,hi+padding


def curve_legend(fig,source):
    cap=12 if source=="cgl" else 11
    handles=[Line2D([],[],color=COL["A"],lw=1.4,label="A input: target subset"),
             Line2D([],[],color=COL["B"],lw=1.4,label="B/C input: same union"),
             Rectangle((0,0),1,1,fc="#E6EEF8",ec="none",label=f"Expanded-only counts ({cap+1}–15)"),
             *[Line2D([],[],color=GREY,ls=STYLES[s],lw=1.1,label=f"Seed {s}") for s in SEEDS]]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.995),ncol=3,
               frameon=False,columnspacing=1.5,handlelength=2.3,labelspacing=.7,fontsize=7.8)


def curve_panel(ax,records,source,ylim,tail=False):
    cap=12 if source=="cgl" else 11
    # The shaded integers, not another dashed line, identify the extra candidates.
    ax.axvspan(cap+.5,15.45,color="#E6EEF8",zorder=0)
    for rec in sorted(records,key=lambda r:r["seed"]):
        for key,color in [("A_trials",COL["A"]),("union_trials",COL["B"])]:
            points=[v for v in rec[key] if not tail or v["components"]>=8]
            ax.plot([v["components"] for v in points],[v["mean_r2"] for v in points],
                    color=color,ls=STYLES[rec["seed"]],lw=1.15,alpha=.78,zorder=2)
    ax.set_xlim(7.6 if tail else .6,15.45); ax.set_ylim(*ylim)
    ax.set_xticks([8,10,12,15] if tail else [1,5,10,15])
    ticks=MaxNLocator(nbins=3).tick_values(*ylim)
    ax.set_yticks([v for v in ticks if ylim[0]<=v<=ylim[1]])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if ylim[0]>.9 else "%.2f"))
    ax.tick_params(labelsize=8)
    clean(ax,"y")


def figureS1(final):
    fig=figure_mm(219)
    axs=fig.subplots(4,2,gridspec_kw=dict(left=.14,right=.97,top=.865,bottom=.105,
                                         hspace=.65,wspace=.40))
    curve_legend(fig,"cgl")
    full_lims=limits("cgl");tail_lims=limits("cgl",tail=True)
    fig.text(.14,.908,"CGL · original partition · all three seeds, without averaging curves",fontsize=9,weight="bold")
    for target in range(4):
        records=[r for r in CURVES if r["source"]=="cgl" and r["target"]==target]
        for col,tail in enumerate([False,True]):
            ax=axs[target,col]
            curve_panel(ax,records,"cgl",tail_lims if tail else full_lims,tail)
            panel(ax,string.ascii_lowercase[target*2+col],
                  TARGETS["cgl"][target]+(" · 8–15 detail" if tail else " · full range"),label_x=-.15)
            ax.set_ylabel("Validation R²",fontsize=8.5)
            if target==3:ax.set_xlabel("PLS component count",fontsize=8.5)
    fig.text(.14,.045,"Vertical scale is common within each column. Actual selected counts are indexed in Figure S3.",fontsize=7.8,color=GREY)
    fig.text(.14,.02,"Validation curves describe the candidate range; they are not outer-test performance or convergence evidence.",fontsize=7.5,color=GREY)
    export(fig,"Figure_S1",final);plt.close(fig)


def corn_page(target,final):
    name="Figure_S2_"+TARGETS["corn"][target].lower()
    fig=figure_mm(239)
    axs=fig.subplots(5,2,gridspec_kw=dict(left=.14,right=.97,top=.855,bottom=.10,
                                         hspace=.63,wspace=.40))
    curve_legend(fig,"corn")
    fig.text(.14,.922,"Corn · "+TARGETS["corn"][target]+" · original five-fold partition",fontsize=10,weight="bold")
    fig.text(.14,.897,"One target per sheet; the five folds share the same vertical range in each window.",fontsize=7.8,color=GREY)
    full_lims=limits("corn",target);tail_lims=limits("corn",target,tail=True)
    for fold in range(1,6):
        records=[r for r in CURVES if r["source"]=="corn" and r["target"]==target and r["fold"]==f"fold_{fold}"]
        for col,tail in enumerate([False,True]):
            ax=axs[fold-1,col]
            curve_panel(ax,records,"corn",tail_lims if tail else full_lims,tail)
            panel(ax,string.ascii_lowercase[(fold-1)*2+col],
                  f"Fold {fold}"+(" · 8–15 detail" if tail else " · full range"),label_x=-.15)
            ax.set_ylabel("Validation R²",fontsize=8.5)
            if fold==5:ax.set_xlabel("PLS component count",fontsize=8.5)
    fig.text(.14,.04,"All three seeds and all negative scores are retained. Actual selected counts: Figure S3.",fontsize=7.8,color=GREY)
    fig.text(.14,.015,"Full and detail windows have different scales; candidate scores are validation R², not test R².",fontsize=7.5,color=GREY)
    export(fig,name,final)
    return fig


def figureS2(final):
    figs=[]
    with PdfPages(OUT/"Figure_S2_four_targets.pdf") as pdf:
        for target in range(4):
            fig=corn_page(target,final);pdf.savefig(fig);figs.append(fig)
    for fig in figs:plt.close(fig)


def selection_matrix(ax,source,target=None):
    if source=="cgl":
        rows=[("official",j) for j in range(4)]; names=TARGETS[source]
    else:
        rows=[(f"fold_{f}",target) for f in range(1,6)];names=[f"Fold {f}" for f in range(1,6)]
    for y,(fold,j) in enumerate(rows):
        for group,seed in enumerate(SEEDS):
            rec=next(r for r in CURVES if r["source"]==source and r["fold"]==fold and r["target"]==j and r["seed"]==seed)
            for k,arm in enumerate("ABC"):
                x=group*3+k; n=rec[arm+"_components"]
                at_boundary=(arm=="B" and n==rec["B_upper"]) or (arm=="C" and n==15)
                ax.add_patch(Rectangle((x-.5,y-.5),1,1,fc="#CCE1ED" if at_boundary else "#F1F3F5",
                                      ec="white",lw=1))
                ax.text(x,y,str(n),ha="center",va="center",fontsize=8.5,
                        weight="bold" if at_boundary else "normal",color=INK)
    ax.set_xlim(-.5,8.5);ax.set_ylim(len(rows)-.5,-.5)
    ax.set_xticks(range(9),list("ABC")*3);ax.set_yticks(range(len(rows)),names)
    ax.tick_params(length=0,pad=5,labelsize=8)
    for group,seed in enumerate(SEEDS):
        text(ax,(group*3+1.5)/9,-.245,f"Seed {seed}",ha="center",fontsize=8)
    for x in [2.5,5.5]:ax.axvline(x,color="white",lw=3)
    for s in ax.spines.values():s.set_visible(False)


def figureS3(final):
    fig=figure_mm(222)
    axs=fig.subplots(3,2,gridspec_kw=dict(left=.15,right=.97,top=.88,bottom=.115,
                                         hspace=.95,wspace=.55))
    fig.text(.15,.96,"Actual selected PLS component counts",fontsize=11,weight="bold")
    fig.text(.15,.925,"Shaded bold cells: B selects its limit q, or C selects 15. Numbers are counts, not prediction scores.",
             fontsize=7.8,color=GREY)
    selection_matrix(axs[0,0],"cgl");panel(axs[0,0],"a","CGL · official partition",label_x=-.14)
    b=axs[0,1];b.set_axis_off();panel(b,"b","Boundary selections",label_x=-.14)
    for x,lab in [(0,"Source"),(.40,"B = q"),(.65,"C = 15"),(.88,"C > q")]:
        text(b,x,.80,lab,fontsize=8.2,weight="bold",ha="left" if x==0 else "center")
    boundary=[]
    for y,source,label in [(.56,"cgl","CGL"),(.34,"corn","Corn")]:
        recs=[r for r in CURVES if r["source"]==source]
        counts=[sum(r["B_components"]==r["B_upper"] for r in recs),
                sum(r["C_components"]==15 for r in recs),
                sum(r["C_components"]>r["B_upper"] for r in recs)]
        text(b,0,y,label,fontsize=8.5)
        for x,count in zip([.40,.65,.88],counts):text(b,x,y,f"{count}/{len(recs)}",fontsize=8.5,ha="center")
        boundary.append(dict(source=source,target_pairs=len(recs),B_at_cap=counts[0],C_at_15=counts[1],C_exceeds_B_cap=counts[2]))
    text(b,0,.05,"q = 12 for CGL; q = 11 for Corn.\nDescriptive, dependent target-pair counts.",fontsize=7.8,
         color=GREY,linespacing=1.5)
    for ax,target,letter in zip([axs[1,0],axs[1,1],axs[2,0],axs[2,1]],range(4),"cdef"):
        selection_matrix(ax,"corn",target)
        panel(ax,letter,"Corn · "+TARGETS["corn"][target],label_x=-.14)
    rows=[]
    for r in CURVES:
        for arm in "ABC":
            rows.append(dict(source=r["source"],fold=r["fold"],target=r["target_name"],seed=r["seed"],
                             arm=arm,selected_components=r[arm+"_components"],
                             allowed_max=r["B_upper"] if arm in "AB" else 15))
    fig.text(.15,.025,"All 216 recorded choices are shown. Seeds and folds are not independent material sources.",fontsize=8,color=GREY)
    write_csv("Figure_S3_selected_components",rows);write_csv("Figure_S3_boundary_counts",boundary)
    export(fig,"Figure_S3",final);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--draft",action="store_true")
    args=p.parse_args();integrity()
    figureS1(not args.draft);figureS2(not args.draft);figureS3(not args.draft)
    manifest=[]
    for source in ["cgl","corn"]:
        for target in ([None] if source=="cgl" else range(4)):
            records=[r for r in CURVES if r["source"]==source and (target is None or r["target"]==target)]
            manifest.append(dict(source=source,target="all" if target is None else TARGETS[source][target],
               paired_records=len(records),curves=len(records)*2,
               distinct_scores=sum(len(r["A_trials"])+len(r["union_trials"]) for r in records),
               full_ylim=list(limits(source,target)),tail_ylim=list(limits(source,target,tail=True))))
    (QA/"curve_coverage_and_limits.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":main()
