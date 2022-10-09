# using
using Dates
using Logging
using CSV

include("src/utils.jl")
include("src/SI_NNLS_restart.jl")

DATASET_INFO = Dict([
    ("mnist", (60000, 780, "data/mnist")),
    ("E2006test", (3308, 150360, "data/E2006test")),
    ("E2006train", (16087 , 150360, "data/E2006train")),
    ("news20", (19996, 1355191, "data/news20.binary")),
    ("real-sim", (72309, 20958, "data/real-sim")),
])

# load data
dataset = "real-sim"
# dataset = "news20"
# dataset = "E2006train"

n, d, path = DATASET_INFO[dataset]
C, b =  read_libsvm_into_sparse(path, n, d)

rows, columns, values = findnz(C)


C, C_b, non_zero_col_norm = reformulation_sparsev4(C, b)
@info "reformulated!"
m, n = size(C)
x0_ = zeros(n)
C_x0_ = C * x0_

outputdir = "./results"
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS-sss")

println("timestamp = $(timestamp)")
println("Completed initialization.")
total_time = 360.0
num_restart = 2000
ϵ = 1e-7

     begin
        x0 = x0_[:]
        C_x0 = C_x0_[:]
        K = 100000
        freq = 20 
        restart_ratio = 0.7
        bs = 1

        @info "SI_NNLS (bs=$bs) without Restart Running on $(dataset) dataset."
        @info "--------------------------------------------------"
        file_path = "$(outputdir)/$(dataset)_SINNLS_nonrestart_$(bs)_$(timestamp).csv"
        SI_NNLS_restart_v3(C, b, C_b, x0, C_x0, bs, K, total_time, num_restart, freq, restart_ratio, file_path, ϵ)
    end

