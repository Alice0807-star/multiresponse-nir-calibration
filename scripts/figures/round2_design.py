"""Round 2: original editable design schematic using observed set membership."""
import argparse
from collections import Counter
from plot_common import *


def figure1(final):
    fig=figure_mm(168)
    gs=fig.add_gridspec(2,5,left=.04,right=.98,top=.95,bottom=.075,
                       width_ratios=[1,1,1,1,1],height_ratios=[2.5,1.65],hspace=.30,wspace=.37)
    a=fig.add_subplot(gs[0,:2]); b=fig.add_subplot(gs[0,2:]); c=fig.add_subplot(gs[1,:])
    for ax,letter,title in [(a,"a","Construct the union"),(b,"b","Arms and controlled comparisons"),
                            (c,"c","Selection and held-out evaluation")]:
        ax.set_axis_off(); ax.set_xlim(0,1); ax.set_ylim(0,1)
        text(ax,0,1.05,letter,fontsize=11,weight="bold")
        text(ax,.085,1.05,title,fontsize=9.3,weight="bold")
        ax.set_gid("panel_"+letter)
    counts=Counter(v for row in MEMBERS["own"] for v in row)
    shared={v for v,n in counts.items() if n>1}
    assert len(shared)==6
    text(a,0,.95,"CGL original · seed 17",fontsize=8,color=GREY)
    left,right=.18,.98
    w=(right-left)/42
    yy=[.79,.66,.53,.40]
    # Shading tracks the same actual column across the four input subsets.
    for i,v in enumerate(MEMBERS["union"]):
        if v in shared:
            a.add_patch(Rectangle((left+i*w,.39),w*.88,.49,transform=a.transAxes,
                        fc="#E2F0EF",ec="none",gid=f"shared_column_track_{v}"))
    for j,(row,y) in enumerate(zip(MEMBERS["own"],yy)):
        text(a,0,y+.03,f"$S_{j+1}$",fontsize=9,va="center")
        text(a,.98,y+.075,TARGETS["cgl"][j]+" · 12",fontsize=7.8,ha="right")
        for i,v in enumerate(MEMBERS["union"]):
            face=COL["B"] if v in row and v in shared else COL["A"] if v in row else "#F2F4F6"
            a.add_patch(Rectangle((left+i*w,y),w*.84,.052,transform=a.transAxes,
                        fc=face,ec="none",gid=f"observed_subset_{j+1}_column_{v}"))
    arrow(a,(.57,.35),(.57,.28))
    text(a,.03,.31,"Merge; deduplicate",fontsize=7.8)
    text(a,0,.22,"$U$",fontsize=9,va="center")
    for i,v in enumerate(MEMBERS["union"]):
        a.add_patch(Rectangle((left+i*w,.19),w*.84,.055,transform=a.transAxes,
                    fc=COL["B"] if v in shared else COL["A"],ec="none",gid=f"observed_union_column_{v}"))
    text(a,.98,.14,"42 distinct columns; 6 shared",fontsize=7.8,ha="right")
    a.add_patch(Rectangle((0,.065),.026,.025,transform=a.transAxes,fc=COL["B"],ec="none"))
    text(a,.05,.067,"Shared columns",fontsize=7.5)
    text(a,0,-.005,"Ordered within U, not a wavelength axis.",fontsize=7.5,color=GREY)

    # A compact arm table anchors the contrasts in actual, different settings.
    headings=[(.0,"Arm"),(.25,"Reads"),(.73,"Components")]
    for x,value in headings:
        text(b,x,.935,value,fontsize=8.4,weight="bold")
    configs=[("A","$S_j$","1–q",COL["A"]),
             ("B","$U$","1–q",COL["B"]),
             ("C","$U$","1–15",COL["C"]),
             ("Dlow","$D_j$ (draw r)","1–q",COL["D_low"]),
             ("Dwide","Same $D_j$","1–15",COL["D_wide"])]
    for i,(name,reads,allowed,color) in enumerate(configs):
        y=.805-i*.105
        b.add_patch(Rectangle((0,y-.015),.195,.082,transform=b.transAxes,fc=color,ec="none",gid="arm_"+name))
        text(b,.0975,y+.024,name,fontsize=8.7,weight="bold",color="white",ha="center",va="center")
        text(b,.25,y+.024,reads,fontsize=8.6,va="center")
        text(b,.73,y+.024,allowed,fontsize=8.4,va="center")
    text(b,0,.335,"q: CGL=12, Corn=11; feasible counts only.",fontsize=7.7)
    text(b,0,.27,"D: retain Sj + random columns; match |U| inputs.",fontsize=7.7)
    for y,comparison,question in [(.16,"A–B","Input visibility"),(.085,"B–C","Component range"),
                                 (.01,"B–Dlow / C–Dwide","Source of added columns")]:
        text(b,0,y,comparison,fontsize=8,weight="bold")
        text(b,.48,y,question,fontsize=8)

    # Shared V is deliberately more prominent than a footnote.
    c.add_patch(Rectangle((0,.735),.405,.235,transform=c.transAxes,
                fc="#FFF1DC",ec="#D2A05D",lw=.8,gid="shared_validation_notice"))
    text(c,.2025,.885,"Shared validation V",fontsize=9,weight="bold",ha="center")
    text(c,.2025,.785,"Both selection steps",fontsize=8,ha="center")
    arrow(c,(.085,.735),(.085,.645),color="#A4783F")
    arrow(c,(.29,.735),(.29,.645),color="#A4783F")
    nodes=[(0,.17,"Select Sj","Fit on F;\nsearch on V"),
           (.20,.18,"Choose components","PLS1: fit F;\nchoose on V"),
           (.41,.15,"Refit F + V","Final\nmodels"),
           (.59,.22,"Freeze all new","Selections and\nrefitted models"),
           (.84,.16,"Score on T","Held-out\npredictions")]
    for i,(x,width,title,body) in enumerate(nodes):
        c.add_patch(Rectangle((x,.24),width,.39,transform=c.transAxes,
                    fc="#E3F0EE" if i==3 else PALE,ec="none",gid=f"evaluation_stage_{i+1}"))
        text(c,x+width/2,.53,title,fontsize=8.4,weight="bold",ha="center",va="center")
        text(c,x+width/2,.355,body,fontsize=7.8,ha="center",va="center",linespacing=1.4)
        if i<4:
            arrow(c,(x+width+.005,.44),(nodes[i+1][0]-.006,.44))
    text(c,0,.105,"All new units frozen before any test prediction/scoring.",fontsize=8.3,weight="bold")
    text(c,0,-.02,"F: fitting; V: validation; T: outer test.",fontsize=7.8,color=GREY)
    fig.text(.04,.025,"Original A–C: records reused; new Corn partitions reselect subsets.",
             fontsize=7.5,color=GREY)
    write_csv("Figure_1_observed_membership",[
       dict(union_position=i+1,column_index=v,subset_1=int(v in MEMBERS["own"][0]),
            subset_2=int(v in MEMBERS["own"][1]),subset_3=int(v in MEMBERS["own"][2]),
            subset_4=int(v in MEMBERS["own"][3]),membership_count=counts[v])
       for i,v in enumerate(MEMBERS["union"])])
    export(fig,"Figure_1",final)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--draft",action="store_true")
    args=p.parse_args();integrity();figure1(not args.draft)


if __name__=="__main__":main()
