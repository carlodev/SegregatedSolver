include("BoundaryConditions.jl")

function run_cylinder(params,distribute)
    @unpack rank_partition, D, N, t0, dt, tF, ν, θ, M = params

    parts  = distribute(LinearIndices((prod(rank_partition),)))

    mesh_file_path = joinpath(@__DIR__, "../../models", params[:mesh_file])
    
    model = GmshDiscreteModel(parts, mesh_file_path)
    #add_SEM_tag!(model)
    writevtk(model,"Cylinder")
    

    # hf_gen!(params)
    


    u_diri_tags, u_diri_values, p_diri_tags, p_diri_values, u0, force_tags = bc_cylinder(params) 
    merge!(params, Dict(:u0 => u0, :model => model))

    V, U, P, Q, Y, X = creation_fe_spaces(params, u_diri_tags, u_diri_values, p_diri_tags, p_diri_values)
    

    degree = 4*params[:order]
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)

    forces_params = Dict(:force_tags => force_tags,:ρ => params[:ρ], :degree => degree, :model => model)
    new_dict = Dict(:U => U,
    :P => P,
    :X => X,
    :Y => Y,
    :Ω => Ω,
    :dΩ => dΩ,
    :degree => degree,
    :force_params => forces_params)
    merge!(params, new_dict)

    trials = [U, P]
    tests = [V, Q]
  
    merge!(params, Dict(:trials => trials, :tests => tests))
  
    solve_case(params)




end

