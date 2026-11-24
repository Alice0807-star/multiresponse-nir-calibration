"""Round 1: paired differences, partition matrix, and target-level error costs."""
import argparse
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from plot_common import *


def paired_effects():
    rows=[]
    for cohort,label in zip(COHORTS,COHORT_LABELS):
        record=SUMMARY["cohorts"][cohort]
        for union,random in [("B","D_low"),("C","D_wide")]:
            for draw in record["D_draws"][random]:
                rows.append(dict(cohort=cohort,label=label,union_method=union,random_method=random,
                    workflow=draw["replicate"],union_macro_r2=record["methods"][union]["mean_r2"],
                    random_macro_r2=draw["mean_r2"],
                    delta_macro_r2=record["methods"][union]["mean_r2"]-draw["mean_r2"]))
    return rows


def figure2(final):
    rows=paired_effects(); summaries=[]
    fig=figure_mm(151)
    gs=fig.add_gridspec(3,2,left=.26,right=.73,bottom=.12,top=.89,
                        height_ratios=[6,0.55,1.45],hspace=.4,wspace=.8)
    ax=fig.add_subplot(gs[0,:])
    # The main panel puts all 80 paired differences on the same physical scale.
    for i,(cohort,label) in enumerate(zip(COHORTS,COHORT_LABELS)):
        for k,(union,random,marker) in enumerate([("B","D_low","s"),("C","D_wide","o")]):
            y=i*2+k
            block=[r for r in rows if r["cohort"]==cohort and r["union_method"]==union]
            vals=np.array([r["delta_macro_r2"] for r in block])
            mean=float(vals.mean()); positive=int((vals>0).sum()); equal=int((vals==0).sum()); negative=int((vals<0).sum())
            ax.scatter(vals,y+np.linspace(-.11,.11,10),s=22,marker=marker,
                       facecolor="white",edgecolor=COL[union],lw=.9,zorder=3)
            ax.plot(mean,y,ls="none",marker="|",ms=12,mew=1.6,color=INK,zorder=4)
            ax.text(1.055,y,f"{mean:+.6f}",transform=ax.get_yaxis_transform(),fontsize=8,
                    va="center",clip_on=False)
            ax.text(1.335,y,f"{positive}/{equal}/{negative}",transform=ax.get_yaxis_transform(),fontsize=8,
                    va="center",clip_on=False)
            summaries.append(dict(cohort=cohort,contrast=union+" minus "+random,
                         mean_delta_macro_r2=mean,positive=positive,equal=equal,negative=negative,
                         min_delta=float(vals.min()),max_delta=float(vals.max())))
        if i<3:
            ax.axhline(i*2+1.5,color=GRID,lw=.65)
    ax.axvline(0,color=GREY,lw=.95)
    ax.set_yticks(np.arange(8),[f"{label} · {method}" for label in COHORT_LABELS for method in ["B − Dlow","C − Dwide"]])
    ax.tick_params(axis="y",length=0,pad=8,labelsize=8)
    ax.set_ylim(7.55,-.55); ax.set_xlim(-.0105,.038)
    ax.set_xticks([-.01,0,.01,.02,.03]); ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel("Difference in macro-average R² (union − random)",labelpad=8,fontsize=8.5)
    clean(ax); ax.spines["left"].set_visible(False)
    panel(ax,"a","Paired differences on one scale",label_x=-.10)
    text(ax,1.055,1.035,"Mean Δ",fontsize=8,weight="bold")
    text(ax,1.335,1.035,"+ / = / −",fontsize=8,weight="bold")
    handles=[Line2D([],[],marker="s",color=COL["B"],mfc="white",ls="none",label="B − Dlow"),
             Line2D([],[],marker="o",color=COL["C"],mfc="white",ls="none",label="C − Dwide"),
             Line2D([],[],marker="|",color=INK,mew=1.5,ms=10,ls="none",label="Mean of 10 differences")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.51,.993),ncol=3,
               frameon=False,columnspacing=1.7,handletextpad=.5)
    # Two explicitly labelled magnified windows keep small CGL effects legible.
    # These repeat the CGL points in panel a; they do not add observations.
    zgs=fig.add_gridspec(1,2,left=.15,right=.97,bottom=.105,top=.245,wspace=.43)
    for z,union,random,letter,marker in [(0,"B","D_low","b","s"),(1,"C","D_wide","c","o")]:
        zz=fig.add_subplot(zgs[0,z])
        vals=np.array([r["delta_macro_r2"] for r in rows if r["cohort"]=="cgl/original" and r["union_method"]==union])*1e4
        zz.scatter(vals,.5+np.linspace(-.10,.10,10),s=23,marker=marker,
                   facecolor="white",edgecolor=COL[union],lw=.9,zorder=3)
        zz.plot(vals.mean(),.5,marker="|",ls="none",color=INK,ms=12,mew=1.6,zorder=4)
        zz.axvline(0,color=GREY,lw=.9); zz.set_xlim(-.2,3.5); zz.set_ylim(0,1)
        zz.set_yticks([]); zz.set_xticks([0,1,2,3])
        zz.set_xlabel("Difference in macro R² ($\\times10^{-4}$)",fontsize=9)
        clean(zz); zz.spines["left"].set_visible(False)
        panel(zz,letter,f"CGL: {union} − {'Dlow' if z==0 else 'Dwide'} · zoom",label_x=-.12)
    fig.text(.15,.019,"Original settings: metric means across three seeds. New Corn partitions: seed 17. Counts are descriptive.",
             fontsize=7.5,color=GREY)
    write_csv("Figure_2_paired_differences",rows); write_csv("Figure_2_effect_summary",summaries)
    export(fig,"Figure_2",final)
    plt.close(fig)


def figure3(final):
    records=[]; data=[]
    definitions=[("B − A","B","A"),("C − B","C","B"),("C − mean(Dwide)","C","D_wide")]
    for contrast,num,den in definitions:
        vals=[]
        for row in SUMMARY["split_sensitivity_fixed_seed"]:
            val=row[num]-row[den]; vals.append(val)
            records.append(dict(cohort=row["cohort"],selection_seed=17,contrast=contrast,delta_macro_r2=val))
        data.append(vals)
    data=np.array(data)
    fig=figure_mm(91)
    ax=fig.add_subplot(111)
    fig.subplots_adjust(left=.26,right=.92,bottom=.25,top=.82)
    cmap=LinearSegmentedColormap.from_list("signed_difference",["#B87845","#FAFAF8","#315C9B"])
    # Native vector cells keep the entire numerical matrix editable in SVG/PDF.
    im=ax.pcolormesh(np.arange(4)-.5,np.arange(4)-.5,data,cmap=cmap,
                    norm=TwoSlopeNorm(vmin=-.03,vcenter=0,vmax=.03),
                    shading="flat",edgecolors="white",linewidth=2,rasterized=False)
    ax.set_xlim(-.5,2.5);ax.set_ylim(2.5,-.5)
    for (j,k),v in np.ndenumerate(data):
        ax.text(k,j,f"{v:+.5f}",ha="center",va="center",fontsize=11,
                color="white" if v>.019 else INK)
    ax.set_xticks(range(3),["Original","31001","31002"])
    ax.set_yticks(range(3),[r[0] for r in definitions]); ax.tick_params(length=0,pad=9)
    ax.set_xlabel("Five-fold partition scheme",labelpad=9)
    ax.set_title("Corn partition sensitivity · selection seed 17",loc="left",pad=15,weight="bold")
    ax.set_xticks(np.arange(-.5,3,1),minor=True); ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which="minor",color="white",lw=2); ax.tick_params(which="minor",length=0)
    for s in ax.spines.values(): s.set_visible(False)
    cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.08,ticks=[-.03,0,.03])
    cb.ax.set_ylabel("Δ macro-average R²",fontsize=8,labelpad=7); cb.ax.tick_params(labelsize=8)
    if cb.solids is not None:cb.solids.set_rasterized(False)
    cb.outline.set_visible(False)
    fig.text(.16,.045,"The same 80 samples are repartitioned; these are not independent material sources.",fontsize=8,color=GREY)
    write_csv("Figure_3_partition_differences",records)
    export(fig,"Figure_3",final)
    plt.close(fig)


def figure4(final):
    fig=figure_mm(116)
    axs=fig.subplots(1,2,gridspec_kw=dict(left=.145,right=.92,bottom=.23,top=.79,wspace=.88))
    records=[]; absolute=[]
    for ax,source,letter in zip(axs,["cgl","corn"],"ab"):
        methods=SUMMARY["cohorts"][source+"/original"]["methods"]
        for arm,scores in methods.items():
            for target,value in zip(TARGETS[source],scores["rmse"]):
                absolute.append(dict(source=source,partition="original",target=target,method=arm,aggregated_rmse=value))
        for num,den,off,marker,color,label in [("B","A",-.25,"s",COL["B"],"B / A"),
              ("C","A",0,"o",COL["C"],"C / A"),("C","D_wide",.25,"^",COL["D_wide"],"C / Dwide")]:
            vals=np.asarray(methods[num]["rmse"])/np.asarray(methods[den]["rmse"])
            if not np.all(vals>0): raise ValueError("Nonpositive RMSE ratio cannot be shown on a log axis")
            ax.scatter(vals,np.arange(4)+off,s=28,marker=marker,color=color,label=label,zorder=3)
            for j,val in enumerate(vals):
                ax.text(1.055,j+off,f"{val:.3f}",transform=ax.get_yaxis_transform(),
                        va="center",fontsize=8,color=INK,clip_on=False)
                records.append(dict(source=source,partition="original",target=TARGETS[source][j],
                     numerator=num,denominator=den,rmse_numerator=methods[num]["rmse"][j],
                     rmse_denominator=methods[den]["rmse"][j],ratio=val))
        ax.axvline(1,color=GREY,lw=.9,ls="--")
        ax.set_xscale("log"); ax.set_xlim(.35,5.5)
        ax.set_xticks([.5,1,2,4],["0.5","1","2","4"]); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_yticks(range(4),TARGETS[source]); ax.set_ylim(3.6,-.6)
        ax.set_xlabel("RMSE(numerator) / RMSE(denominator)\nLogarithmic scale",fontsize=8,labelpad=9)
        clean(ax); panel(ax,letter,"CGL" if source=="cgl" else "Corn",label_x=-.14)
        text(ax,1.055,1.035,"Ratio",fontsize=8,weight="bold")
    h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc="upper center",bbox_to_anchor=(.54,.995),ncol=3,frameon=False,columnspacing=2.6)
    fig.text(.145,.885,"Original partitions · RMSE metrics averaged across three seeds",fontsize=9,weight="bold")
    fig.text(.145,.055,"Ratios < 1 favour the numerator; ratios > 1 favour the denominator.",fontsize=8,color=GREY)
    fig.text(.145,.02,"Dwide additionally averages 10 workflows. Ratios are calculated after averaging RMSE.",fontsize=7.5,color=GREY)
    write_csv("Figure_4_rmse_ratios",records); write_csv("Figure_4_absolute_rmse",absolute)
    export(fig,"Figure_4",final)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--draft",action="store_true")
    args=p.parse_args(); integrity()
    for fn in [figure2,figure3,figure4]: fn(not args.draft)


if __name__=="__main__": main()
