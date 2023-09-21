function init_params(params)

    @unpack N,D,rank_partition,t0,dt,tF,case,c,ρ,u_in,Re,ν = params

    @assert length(rank_partition) == D

    if case == "Airfoil" || case == "LidDriven"  || case == "Cylinder"
        params[:ν] = c*ρ*u_in/Re
        println("Recompute ν = $(params[:ν])")
    end

    #Time Step vector creation
    time_step = t0+dt:dt:tF 
    merge!(params,Dict(:time_step=>time_step))


    if params[:restart]
        @unpack restart_file, t_endramp, t0 = params
        restart_path = joinpath(@__DIR__, "../../restarts", restart_file)
        restart_df = DataFrame(CSV.File(restart_path))
        
        initial_rescale_factor = 1.0
        
        if t_endramp>t0
            initial_rescale_factor = t_endramp/t0
        end
        params = merge!(params, Dict(:restart_df => restart_df, :initial_rescale_factor=>initial_rescale_factor))
    end


end