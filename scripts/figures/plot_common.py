"""Shared publication styling and frozen-record I/O; no training operations."""
from pathlib import Path
import csv
import json
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, NullLocator

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"evidence"/"figure_data"
OUT=ROOT/"generated"/"figures"; PREVIEW=ROOT/"generated"/"figure_previews"; QA=ROOT/"generated"/"figure_qa"
GENERATED_DATA=ROOT/"generated"/"figure_data"
for folder in [OUT,PREVIEW,QA,GENERATED_DATA]:folder.mkdir(parents=True,exist_ok=True)
# Author-side layout QA is optional and not redistributed as project code.
# All published frozen figures were already visually and numerically audited.
import os
require_matplotlib_panel_alignment=None
if os.environ.get("NATURE_FIGURE_QA_DIR"):
    sys.path.insert(0,os.environ["NATURE_FIGURE_QA_DIR"])
    from audit_panel_alignment import require_matplotlib_panel_alignment

SUMMARY=json.loads((DATA/"summary.json").read_text(encoding="utf-8"))
CURVES=json.loads((DATA/"candidate_curves.json").read_text(encoding="utf-8"))
MEMBERS=json.loads((DATA/"figure1_membership.json").read_text(encoding="utf-8"))
COL=dict(A="#535860",B="#16827E",C="#315C9B",D_low="#747C87",D_wide="#B8793E")
INK="#242A32"; GREY="#68737F"; GRID="#E4E8EC"; PALE="#F0F3F5"
SEEDS=(17,29,43)
STYLES={17:"-",29:(0,(4,2)),43:(0,(1,1.5))}
TARGETS={"cgl":["Casein","Glucose","Lactate","Moisture"],"corn":["Moisture","Oil","Protein","Starch"]}
COHORTS=["cgl/original","corn/original","corn/repeat_31001","corn/repeat_31002"]
COHORT_LABELS=["CGL original","Corn original","Corn 31001","Corn 31002"]

plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "font.size":9,"axes.labelsize":9,"axes.titlesize":9.5,"xtick.labelsize":8,"ytick.labelsize":8,
 "legend.fontsize":8,"text.color":INK,"axes.labelcolor":INK,"axes.edgecolor":"#98A2AD",
 "xtick.color":INK,"ytick.color":INK,"axes.linewidth":.7,"lines.linewidth":1.2,
 "lines.markersize":4.5,"xtick.major.width":.7,"ytick.major.width":.7,
 "xtick.major.size":3,"ytick.major.size":3,"svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,
 "savefig.facecolor":"white","figure.facecolor":"white","mathtext.fontset":"dejavusans",
 "axes.formatter.useoffset":False})


def figure_mm(height,width=180):
    return plt.figure(figsize=(width/25.4,height/25.4))


def panel(ax,letter,title=None,pad=9,label_x=-.12):
    ax.set_gid("panel_"+letter)
    trans=ax.transAxes+ScaledTranslation(0,pad/72,ax.figure.dpi_scale_trans)
    ax.text(label_x,1,letter,transform=trans,fontsize=11,fontweight="bold",va="baseline",
            clip_on=False,gid="panel_label_"+letter)
    if title:
        ax.set_title(title,loc="left",pad=pad,fontweight="bold")


def clean(ax,grid="x"):
    for side in ["top","right"]:
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid,color=GRID,lw=.6)


def export(fig,name,final=True,alignment_options=None):
    fig.canvas.draw()
    comparable=[a for a in fig.axes if a.get_visible() and not a.get_label().startswith("<colorbar")]
    if len(comparable)>1 and require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(fig,json_out=QA/(name+".alignment.json"),
             overlay_svg=QA/(name+".alignment.svg"),strict=True,**(alignment_options or {}))
    fig.savefig(OUT/(name+".pdf"))
    fig.savefig(OUT/(name+".svg"))
    p=OUT/(name+".svg")
    p.write_text(re.sub(r"<!DOCTYPE[^>]*>","",p.read_text(encoding="utf-8")),encoding="utf-8")
    fig.savefig(PREVIEW/(name+".png"),dpi=300)
    if final:
        fig.savefig(OUT/(name+".tif"),dpi=800,pil_kwargs={"compression":"tiff_lzw"})
    print("EXPORTED",name,flush=True)


def write_csv(name,rows):
    with (GENERATED_DATA/(name+".csv")).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def text(ax,x,y,value,**kwargs):
    return ax.text(x,y,value,transform=ax.transAxes,**kwargs)


def arrow(ax,start,end,color=GREY):
    p=FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=9,lw=1,color=color,
                      transform=ax.transAxes,shrinkA=0,shrinkB=0)
    ax.add_patch(p); return p


def integrity():
    assert len(CURVES)==72
    assert sum(len(r["A_trials"])+len(r["union_trials"]) for r in CURVES)==1884
    assert sorted(set(v for s in MEMBERS["own"] for v in s))==MEMBERS["union"]
    assert len(MEMBERS["union"])==42
