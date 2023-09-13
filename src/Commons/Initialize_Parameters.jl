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


end