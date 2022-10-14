using LinearAlgebra
using SparseArrays
using IterativeSolvers
using KrylovKit
mutable struct Data
    features::Array{Float64,2}
    values::Vector{Float64}
end

struct Results
    iterations::Vector{Float64}
    times::Vector{Float64}
    metric::Vector{Float64}
    fvalue::Vector{Float64}

    function Results()
        new(
            Array{Int64}([]),
            Array{Float64}([]),
            Array{Float64}([]),
            Array{Float64}([]),
        )
    end
end


"""Append execution measures to Results."""
function logresult!(r::Results, currentiter, elapsedtime, metric, fvalue)

    push!(r.iterations, currentiter)
    push!(r.times, elapsedtime)
    push!(r.metric, metric)
    push!(r.fvalue, fvalue)
    return
end


"""Export results into CSV formatted file."""
function exportresultstoCSV(results::Results, outputfile::String)

    CSV.write(
        outputfile,
        (
            iterations = results.iterations,
            times = results.times,
            metricLPs = results.metric,
            fvaluegaps = results.fvalue,
        ),
    )
end



#we do not consider sparsity
function read_libsvm(path::String, n::Int64, d::Int64)::Data


    features = zeros(n, d)
    values = zeros(n)
    data_str = readlines(path)
    feature_value_str = split(data_str[2], " ")

    for i = 1:n
        feature_value_str = split(data_str[i], " ")
        value = parse(Float64, feature_value_str[1])
        values[i] = value

        for j = 3:(length(feature_value_str)-1)
            idx_feature_pair = split(feature_value_str[j], ':')
            idx = parse(Int64, idx_feature_pair[1])
            feature = parse(Float64, idx_feature_pair[2])
            features[i, idx] = feature
        end
    end
    Data(features, values)
end

function data_normalize!(features::Array{Float64,2})
    n = size(features, 1)
    for i = 1:n
        features[i, :] = features[i, :] / norm(features[i, :])
    end
end

function reformulation(A, b)
    t1 = time()
    col_norm = norm.(eachcol(A))
    non_zero_col_idx = Vector{Int64}()
    non_zero_col_norm = Vector{Float64}()
    for i = 1:length(col_norm)
        if abs(col_norm[i]) > 1e-10
            push!(non_zero_col_idx, i)
            push!(non_zero_col_norm, col_norm[i])
        end
    end
    Â = A[:, non_zero_col_idx]
    A_T_b = (b' * Â)'
    m, n = size(Â)
    t2 = time()
    @info "td1: $(t2 - t1)"
    Ã = spzeros(m, n)
    for i = 1:size(Â, 2)
        Ã[:, i] = Â[:, i] ./ A_T_b[i]
    end
    t3 = time()
    @info "td1: $(t3 - t2)"
    return Â, Ã, A_T_b, non_zero_col_norm
end

function reformulation_sparse(A::SparseMatrixCSC, b::Vector{Float64})
    t1 = time()
    col_norm = norm.(eachcol(A))
    non_zero_col_idx = Vector{Int64}()
    non_zero_col_norm = Vector{Float64}()
    for i = 1:length(col_norm)
        if abs(col_norm[i]) > 1e-10
            push!(non_zero_col_idx, i)
            push!(non_zero_col_norm, col_norm[i])
        end
    end
    Â = A[:, non_zero_col_idx]
    t2 = time()
    @info "td1: $(t2 - t1)"

    A_T_b = (b' * Â)'
    m, n = size(Â)

    Ã = spzeros(m, n)
    rows = rowvals(Â)
    vals = nonzeros(Â)
    for j = 1:n
        for i in nzrange(Â, j)
            row = rows[i]
            val = vals[i]
            Ã[row, j] = val / A_T_b[j]
            # Ã[row, j] = val
        end
    end
    t3 = time()
    @info "td1: $(t3 - t2)"
    return Â, Ã, A_T_b, non_zero_col_norm
end

function reformulation_sparse(A::SparseMatrixCSC, b::Vector{Float64})
    col_norm = norm.(eachcol(A))
    non_zero_col_idx = Vector{Int64}()
    non_zero_col_norm = Vector{Float64}()
    for i = 1:length(col_norm)
        if abs(col_norm[i]) > 1e-10
            push!(non_zero_col_idx, i)
            push!(non_zero_col_norm, col_norm[i])
        end
    end
    Â = A[:, non_zero_col_idx]
    @info "b length: $(length(b))"
    @info "Â size: $(size(Â))"
    A_T_b = (b'*Â)[:]

    return Â, A_T_b, non_zero_col_norm
end


function reformulation_sparsev4(A::SparseMatrixCSC, b::Vector{Float64})
    col_norm = norm.(eachcol(A))
    non_zero_col_idx = Vector{Int64}()
    non_zero_col_norm = Vector{Float64}()
    for i = 1:length(col_norm)
        if abs(col_norm[i]) > 1e-10
            push!(non_zero_col_idx, i)
            push!(non_zero_col_norm, col_norm[i])
        end
    end
    Â = A[:, non_zero_col_idx]
    @info "b length 1: $(length(b))"
    @info "Â size 1: $(size(Â))"
    A_T_b = (b'*Â)[:]
    non_zero_col_idx = Vector{Int64}()
    for i = 1:length(A_T_b)
        if A_T_b[i] > 0
            push!(non_zero_col_idx, i)
        end
    end
    Â = Â[:, non_zero_col_idx]
    @info "b length 2: $(length(b))"
    @info "Â size 2: $(size(Â))"
    A_T_b = A_T_b[non_zero_col_idx]
    non_zero_col_norm = non_zero_col_norm[non_zero_col_idx]
    return Â, A_T_b, non_zero_col_norm
end


function get_blocks(d, block_size, is_random = false)

    perm = Vector{Vector{Int64}}()
    blocks = d ÷ block_size
    for i = 1:blocks
        push!(perm, [(1+(i-1)*block_size):i*block_size;])
    end
    if blocks * block_size < d
        push!(perm, [(blocks*block_size+1):d;])
    end

    if is_random
        perm[:] = perm[randperm(length(perm))]
    end
    return perm
end


function read_libsvm_into_sparse(
    filepath::String,
    num_dataset::Int,
    dim_dataset::Int,
)

    train_indices = Array{Int}([])
    feature_indices = Array{Int}([])
    values = Array{Float64}([])
    labels = Array{Float64}([])
    open(filepath) do f

        line = 1
        while !eof(f)
            s = readline(f)
            split_line = split(s, " ", keepempty = false)

            label = nothing
            for v in split_line
                if isnothing(label)
                    label = parse(Float64, v)
                    push!(labels, label)
                    continue
                end
                _index, _value = split(v, ":")
                index = parse(Int, _index)
                value = parse(Float64, _value)
                push!(train_indices, line)
                push!(feature_indices, index)
                push!(values, value)
            end
            line += 1
        end
    end

    C = sparse(train_indices, feature_indices, values, num_dataset, dim_dataset)
    return C, labels
end


function compute_blocks_rows_slice(C::SparseMatrixCSC, blocksize::Int)
    d, n = size(C)

    blocks = Array{UnitRange{Int}}([])
    row_idxs = Array{Vector{Int}}([])
    if n % blocksize == 0
        len_b = n ÷ blocksize
    else
        len_b = n ÷ blocksize + 1
    end
    for i = 1:len_b
        if i == len_b
            push!(blocks, (1+(i-1)*blocksize):n)
        else
            push!(blocks, (1+(i-1)*blocksize):(i*blocksize))
        end

        row_set = Set{Int}()
        for j in blocks[i]
            loc = C.colptr[j]:(C.colptr[j+1]-1)
            union!(row_set, C.rowval[loc])
        end
        row_vec = collect(row_set)
        push!(row_idxs, row_vec)
    end

    sliced_Cs = Array{SparseMatrixCSC{Float64,Int}}([])
    for j = 1:length(row_idxs)
        push!(sliced_Cs, C[row_idxs[j], blocks[j]])
    end

    blocks, row_idxs, sliced_Cs
end


function compute_blocks_rows_slice_dense(C, blocksize::Int)
    d, n = size(C)

    blocks = Array{UnitRange{Int}}([])
    if n % blocksize == 0
        len_b = n ÷ blocksize
    else
        len_b = n ÷ blocksize + 1
    end
    for i = 1:len_b
        if i == len_b
            push!(blocks, (1+(i-1)*blocksize):n)
        else
            push!(blocks, (1+(i-1)*blocksize):(i*blocksize))
        end
    end
    sliced_Cs = Array{SparseMatrixCSC{Float64,Int}}([])
    for j = 1:len_b
        push!(sliced_Cs, C[:, blocks[j]])
    end

    blocks, sliced_Cs
end

function compute_Lips(C, blocks, row_idxs)
    etas = zeros(length(blocks))
    blocksize = length(blocks[1])
    if blocksize >= 5
        for i = 1:length(blocks)
            sub_C = C[row_idxs[i], blocks[i]]
            c, _, _, _ = svdsolve(sub_C)
            etas[i] = 1.0 / c[1]^2
        end
    elseif blocksize == 1
        col_norm = norm.(eachcol(C))
        ubs = 1.0 ./ (col_norm .^ 2)
        for i = 1:length(blocks)
            etas[i] = 1.0 / norm(C[row_idxs[i], blocks[i]])^2
        end
    else
        @info "not supported for block size for 2,3,4"
    end
    etas
end

function compute_Lips_dense(sliced_Cs, blocksize)
    etas = zeros(length(sliced_Cs))
    if blocksize >= 5
        for i = 1:length(sliced_Cs)
            c, _ = svdl(sliced_Cs[i], nsv = 1)
            etas[i] = 1.0 / c[1]^2
        end
    elseif blocksize == 1
        for i = 1:length(sliced_Cs)
            etas[i] = 1.0 / norm(sliced_Cs[i])^2
        end
    else
        @info "not supported for block size for 2,3,4"
    end
    etas
end

function compute_eta(C)
    c, _ = svdl(C, nsv = 1)
    return 1.0 / c[1]^2
end

function first_order_opt(C, b, x, C_x, C_b)
    tmp = (C_x'*C)[:]
    val = norm(x - max.(x - tmp + C_b, 0.0))
    return val
end

function first_order_opt(C, b, x, C_x, C_b, col_norm_square)
    tmp = (C_x'*C)[:]
    val = norm(x - max.(x - (tmp - C_b) ./ col_norm_square, 0.0))
    return val
end

function first_order_optv2(C, b, x, C_x, C_b)
    val = 0.0
    n = length(x)
    for i = 1:n
        if abs(x[i]) < 1e-14
            val += max(C_b[i] - C[:, i]' * C_x, 0.0)^2
        elseif x[i] > 0
            val += abs(C_b[i] - C[:, i]' * C_x)^2
        end
    end
    sqrt(val)
end
