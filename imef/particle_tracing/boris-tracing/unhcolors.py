from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# colorscales for color consistency
seq_standard = "YlGnBu"
div_standard = "RdGy"
# pcolor = 'GnBu'

# UNH logo hexcode
nh_blue = "#233E8A"
white = ["#FFFFFF"]

# UNH-themed sequential colors
nh_blues = ["#152451", "#233E8A", "#3B62CE", "#7C97DE", "#BDCBEF", "#DFE5F7", "#EFF2FB"]
nh_grays = ["#333333", "#575757", "#858585", "#ADADAD", "#D6D6D6", "#EBEBEB", "#F5F5F5"]
nh_blues_r = nh_blues[::-1]
nh_grays_r = nh_grays[::-1]

# UNH-themed divergent colors
nh_cscale = nh_blues + white + nh_grays_r
nh_cscale_r = nh_cscale[::-1]

nh_cmap_div = LinearSegmentedColormap.from_list("cdiv", nh_cscale, N=500)
nh_cmap_div_r = LinearSegmentedColormap.from_list("cdiv_r", nh_cscale_r, N=500)
nh_cmap_seq = LinearSegmentedColormap.from_list("cseq", nh_blues, N=500)
nh_cmap_seq_r = LinearSegmentedColormap.from_list("cseq_r", nh_blues_r, N=500)

# colorscale for sequential plots
cseq = nh_cmap_seq
cseq_r = nh_cmap_seq_r
# colorscale for divergent plots (negative and positive values)
cdiv = nh_cmap_div
