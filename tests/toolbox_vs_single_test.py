from sklearn.datasets import make_blobs
from pyclustkit.eval.cvi import *

x, y = make_blobs(n_samples=1000)
cvit = CVIToolbox(x, y)


from pyclustkit.eval.core import process_adg
from pyclustkit.eval.core._adg_operations import get_subgraph

# Define the graph
processes = process_adg
processes = get_subgraph(processes,'max_intra_cluster_distances')


# ball_hall: passed
a1 = cvit.ball_hall()
a2 = ball_hall(x,y)
print(a1, a2)

# banfeld raftery: passed
a1 = cvit.banfeld_raftery()
a2 = banfeld_raftery(x,y)
print(a1, a2)

# c_index:
a1 = cvit.c_index()
a2 = c_index(x,y)
print(a1, a2)

# cdbw : passed
a1 = cvit.cdbw()
a2 = cdbw(x,y)
print(a1, a2)

# det_ratio:
a1 = cvit.det_ratio()
a2 = det_ratio(x, y)
print(a1, a2)

# dunn: passed
a1 = cvit.dunn()
a2 = dunn(x, y)
print(a1, a2)

# gamma: passed
a1 = cvit.gamma()
a2 = gamma(x, y)
print(a1, a2)

# gdi21 : passed
a1 = cvit.gdi21()
a2 = gdi21(x, y)
print(a1, a2)

# gdi31 : passed
a1 = cvit.gdi31()
a2 = gdi31(x, y)
print(a1, a2)

# gdi41 : passed
a1 = cvit.gdi41()
a2 = gdi41(x,y)
print(a1, a2)

# gdi51 : passed
a1 = cvit.gdi51()
a2 = gdi51(x,y)
print(a1, a2)


# gdi61 : passed
a1 = cvit.gdi61()
a2 = gdi61(x,y)
print(a1, a2)

# gdi12 : passed
a1 = cvit.gdi12()
a2 = gdi12(x,y)
print(a1, a2)

# gdi22 : passed
a1 = cvit.gdi22()
a2 = gdi22(x,y)
print(a1, a2)

# gdi32 : passed
a1 = cvit.gdi32()
a2 = gdi32(x,y)
print(a1, a2)

# gdi42 : passed
a1 = cvit.gdi42()
a2 = gdi42(x,y)
print(a1, a2)

# gdi52 : passed
a1 = cvit.gdi52()
a2 = gdi52(x,y)
print(a1, a2)

# gdi62 : passed
a1 = cvit.gdi62()
a2 = gdi62(x,y)
print(a1, a2)

# gdi13 : passed
a1 = cvit.gdi13()
a2 = gdi13(x, y)
print(a1, a2)

# gdi23 : passed
a1 = cvit.gdi23()
a2 = gdi23(x, y)
print(a1, a2)

# gdi33 : passed
a1 = cvit.gdi33()
a2 = gdi33(x, y)
print(a1, a2)

# gdi43 : passed
a1 = cvit.gdi43()
a2 = gdi43(x, y)
print(a1, a2)

# gdi53 : passed
a1 = cvit.gdi53()
a2 = gdi53(x, y)
print(a1, a2)

# gdi63 : passed
a1 = cvit.gdi63()
a2 = gdi63(x, y)
print(a1, a2)

# ksq_detw: passed
a1 = cvit.ksq_detw()
a2 = ksq_detw(x,y)
print(a1,a2)

# log_det_ratio: passed
a1 = cvit.log_det_ratio()
a2 = log_det_ratio(x, y)
print(a1, a2)

# log_ss_ratio: passed
a1 = cvit.log_ss_ratio()
a2 = log_ss_ratio(x, y)
print(a1, a2)

# mcclain_rao: passed
a1 = cvit.mcclain_rao()
a2 = mcclain_rao(x, y)
print(a1, a2)

# pbm: passed
a1 = cvit.pbm()
a2 = pbm(x, y)
print(a1, a2)

# point_biserial: passed
a1 = cvit.point_biserial()
a2 = point_biserial(x, y)
print(a1, a2)

# ratkowsky_lance:
a1 = cvit.ratkowsky_lance()
a2 = ratkowsky_lance(x,y)
print(a1, a2)

# ray-turi: passed
a1 = cvit.ray_turi()
a2 = ray_turi(x, y)
print(a1, a2)


# Friedman Rubin: passed
a1 = cvit.rubin()
a2 = friedman_rudin_2(x, y)
print(a1, a2)

# scott_symons: passed
a1 = cvit.scott_symons()
a2 = scott_symons(x,y)
print(a1, a2)

# sd_dis: passed
a1 = cvit.sd_dis()
a2 = sd_dis(x, y)
print(a1, a2)


# sd_scat: passed
a1 = cvit.sd_scat()
a2 = sd_scat(x, y)
print(a1, a2)

# sdbw: passed
a1 = cvit.sdbw()
a2 = s_dbw(x, y)
print(a1, a2)

# tau: passed
a1 = cvit.tau()
a2 = tau(x, y)
print(a1, a2)


# trace_w: passed
a1 = cvit.trace_w()
a2 = trace_w(x,y)
print(a1, a2)

# trace_wib: passed
a1 = cvit.trace_wib()
a2 = trace_wib(x, y)
print(a1, a2)

# Wemmert Gancarski : passed
a1 = cvit.wemmert_gancarski()
a2 = wemmert_gancarski(x,y)
print(a1,a2)

# Xie Beni: passed
a1 = cvit.xie_beni()
a2 = xie_beni(x,y)
print(a1, a2)
