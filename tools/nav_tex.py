"""Write report/ch_bi/tab_navbench.tex from the navigation benchmark and closed-loop results."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    ("blind", "Blind waypoint follower", "Kör ara nokta takipçisi"),
    ("reactive_gt", "Previous: reactive steering on a GT costmap", "Önceki: GT maliyet haritasında reaktif yönlendirme"),
    ("mppi_none", "LiDAR map, no moving-object handling + MPPI", "LiDAR haritası, hareketli nesne işlemi yok + MPPI"),
    ("mppi_raycast", "LiDAR map + raycast clearing + MPPI", "LiDAR haritası + ışınla temizleme + MPPI"),
    ("mppi_doppler", "LiDAR map + Doppler + MPPI", "LiDAR haritası + Doppler + MPPI"),
    ("mppi_doppler_raycast", "LiDAR map + Doppler + guarded raycast + MPPI (ours)",
     "LiDAR haritası + Doppler + korumalı ışın + MPPI (bu çalışma)"),
]


def open_loop_rows(scene="nav_clutter"):
    res = json.loads((ROOT / "results" / "nav_bench" / f"{scene}_summary.json").read_text())
    by = defaultdict(list)
    for r in res:
        by[r["method"]].append(r)
    rows = []
    for key, en, tr in NAMES:
        rs = by.get(key)
        if not rs:
            continue
        succ = [r for r in rs if r["completed"]]
        t = f"{np.mean([r['duration'] for r in succ]):.0f}" if succ else "--"
        rows.append((en, tr, f"{len(succ)}/{len(rs)}", f"{np.mean([r['collisions'] for r in rs]):.1f}", t,
                     f"{sum(r['people_contacts'] for r in rs)} / {sum(r['robot_caused_contacts'] for r in rs)}",
                     f"{min(r['min_clear_people'] for r in rs):.2f}"))
    return rows


def closed_loop_rows():
    out = []
    for scene in ("nav_clutter", "outdoor"):
        for est, dyn, en in (("3d", "none", "FMCW-LIO 3D pose, no moving-object handling"),
                             ("4d_leg", "doppler+raycast", "4D + legs pose, Doppler + guarded raycast (ours)")):
            rs = []
            for f in sorted((ROOT / "results" / "nav_closed").glob(f"{scene}_s*_{est}_{dyn}.log")):
                m = re.search(r"^\{.*\"est\".*\}$", f.read_text(), re.M)
                if m:
                    rs.append(json.loads(m.group(0)))
            if not rs:
                continue
            succ = [r for r in rs if r["completed"]]
            out.append((scene, en, f"{len(succ)}/{len(rs)}", sum(r["fell"] for r in rs),
                        f"{np.mean([r['duration'] for r in succ]):.0f}" if succ else "--",
                        f"{max(r['est_err_max'] for r in rs):.2f}", sum(r["collisions"] for r in rs)))
    return out


def main():
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Navigation benchmark, open loop for localization (true pose), 5 seeds. Contacts: all / robot-caused.",
         r"\trcap{/ Navigasyon karşılaştırması, lokalizasyon açısından açık döngü (gerçek poz), 5 tohum. Temas: hepsi / robot kaynaklı.}}\label{tab:navbench}",
         r"\begin{tabular}{@{}p{6.2cm}ccccc@{}}\toprule",
         r"Method {\color{trbar}/ Yöntem} & Success & Collisions & Time [s] & Person contacts & Min dist. [m]\\\midrule"]
    for en, tr, *v in open_loop_rows():
        L.append(f"{en} \\trc{{{tr}}} & " + " & ".join(v) + r"\\")
    L += [r"\bottomrule\end{tabular}\end{table}", ""]
    cl = closed_loop_rows()
    if cl:
        L += [r"\begin{table}[h]\centering\footnotesize",
              r"\caption{Closed loop: the robot walks on its own pose estimate AND its own map. Contacts are on/off contact",
              r"events with structure (the stuck classic robot leaning on a road barrier counts many).",
              r"\trcap{/ Kapalı döngü: robot kendi poz tahmini VE kendi haritasıyla yürüyor. Temaslar yapıyla temas başlangıç olaylarıdır",
              r"(takılan klasik robotun yol bariyerine yaslanması çok sayıda olay üretir).}}\label{tab:navclosed}",
              r"\begin{tabular}{@{}lp{5.4cm}ccccc@{}}\toprule",
              r"Scene & Configuration & Done & Falls & Time [s] & Max pose err. [m] & Contacts\\\midrule"]
        for scene, en, *v in cl:
            sc = scene.replace("_", r"\_")
            L.append(r"\texttt{" + sc + "} & " + en + " & " + " & ".join(str(x) for x in v) + r"\\")
        L += [r"\bottomrule\end{tabular}\end{table}"]
    out = ROOT / "report" / "ch_bi" / "tab_navbench.tex"
    out.write_text("\n".join(L) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
