using IterativeSolvers

function SI_NNLS_restart_v3(C::SparseMatrixCSC, b::Vector{Float64}, C_b::Vector{Float64}, x0::Vector{Float64}, C_x0::Vector{Float64}, blocksize::Int64, K::Int64, total_time::Float64,
                                        num_restart::Int64, freq::Int64, restart_ratio::Float64, file_path::String, ϵ)
    t1 = time()
    results = Results()
    col_norm_square = norm.(eachcol(C)).^2

    init_metric = first_order_optv3(C, b, x0, C_x0, C_b, col_norm_square)
    init_epoch = 0
    init_time = 0.0

    blocks, row_idxs, sliced_Cs = compute_blocks_rows_slice(C, blocksize)
    @info "slice time: $(time() - t1)"
    ηs = compute_Lips(C, blocks, row_idxs)

    ubs = 1.0 ./ col_norm_square
    for i = 1:num_restart
        # x0, C_x0, init_metric, td = SI_NNLS(C, x0, C_x0, K, freq, init_metric, γ, blocks, row_idxs, extra_term, ηs, ubs, sliced_Cs)
        x0, C_x0, init_metric, init_epoch, init_time = SI_NNLS_v3(C, b, C_b, x0, C_x0, blocks, row_idxs, sliced_Cs, ηs, K, total_time, freq, init_metric, results, init_epoch, init_time, restart_ratio, ϵ, col_norm_square)
        @info "restart epoch: $i"
        if init_time >= total_time || init_metric < ϵ
            break
        end
    end
    exportresultstoCSV(results, file_path)
end


function SI_NNLS_v3(C::SparseMatrixCSC, b::Vector{Float64}, C_b::Vector{Float64}, x0::Vector{Float64}, C_x0::Vector{Float64}, blocks::Array{UnitRange{Int}}, row_idxs::Array{Vector{Int}}, sliced_Cs, ηs, K::Int64, total_time::Float64, freq::Int64, init_metric::Float64, results::Results, init_epoch::Int64, init_time::Float64, restart_ratio, ϵ, col_norm_square)

    t0 = time()
    m, n = size(C)

    num_blks = length(blocks)
    K *= num_blks
    prev_a, prev_A = 0.0, 0.0
    C_b = (b' * C)[:]
    upper_bounds = C_b ./ col_norm_square

    a = 1.0/(sqrt(2)*num_blks^1.5)
    A = a
    later_a = a / (num_blks - 1)
    later_A = A + later_a

    @info "num_blks: $num_blks"
    p = zeros(n)
    r = zeros(n)
    s = zeros(m)
    t = zeros(m)
    q = C_x0

    x = deepcopy(x0)

    idx_seq = 1:num_blks
    prev_jk, jk = 0, 0
    for k = 1:K
        prev_jk = jk
        jk = rand(idx_seq)
        j = blocks[jk]
        row_j = row_idxs[jk]
        sliced_C = sliced_Cs[jk]
        tt0 = time()
        if k == 1
            product = (q[row_j]' * sliced_C)'
        elseif k == 2
            product = ((q[row_j] + prev_a/a * t[row_j])' * sliced_C)'
            t[row_idxs[prev_jk]] .= 0
        else
            ratio = prev_a^2 / (a * (prev_A - prev_a))
            product = ((q[row_j] + (1-ratio)/prev_A*s[row_j] + (num_blks-1)*ratio*t[row_j])' * sliced_C)'
            # after this, set r to zero
            t[row_idxs[prev_jk]] .= 0
        end
        p[j] = p[j] .+ num_blks * a * (product .- C_b[j])
        prev_xj = x[j]
        x[j] = max.(0.0,  min.(x0[j] - (ηs[jk]) * p[j],  upper_bounds[j]))
        t[row_j] = sliced_C * (x[j] - prev_xj)
        q[row_j] = q[row_j] + t[row_j]
        if k >= 2
            r[j] = r[j] + ((num_blks-1) * a - prev_A) * (x[j]-prev_xj)
            s[row_j] = s[row_j] + ((num_blks-1)*a - prev_A) * t[row_j]
            # r[j] = r[j] + ( - prev_A) * (x[j]-prev_xj)
        end
        prev_a, prev_A = a, A
        a, A = later_a, later_A
        later_a = min(num_blks/(num_blks-1)*later_a, sqrt(later_A)/(2*num_blks))
        later_A = later_A + later_a
        if k % (freq * num_blks) == 0
            x̃ = x + 1.0/prev_A * r
            C_x̃ = C * x̃
            metric = first_order_optv3(C, b, x̃, C_x̃, C_b, col_norm_square)
            func_value = 0.5 * norm(C_x̃ - b)^2
            td = time() - t0
            @info "k: $(k÷num_blks), time: $(td+init_time), metric: $metric,  func_value: $func_value"

            logresult!(results, k + init_epoch, td+init_time, metric, func_value)
            if metric <= restart_ratio * init_metric || td + init_time > total_time || metric < ϵ || k == K
                return x̃, C_x̃, metric, k + init_epoch, td + init_time
            end
        end
    end
end
