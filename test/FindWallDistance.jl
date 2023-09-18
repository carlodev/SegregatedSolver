using SegregatedSolver
using Gridap,GridapDistributed,PartitionedArrays, GridapGmsh,GridapPETSc,Parameters
using DifferentialEquations


walltag = "airfoil"
nparts = 1



function main_wd_test(nparts,distribute)

    parts  = distribute(LinearIndices((nparts,)))
    model = GmshDiscreteModel(parts,joinpath("models","DU89_2D_A1_v2.msh"))
    params= Dict(:u_in=>1.0, :Re=>250e3, :c =>1.0, :D =>2,:model=>model)
    merge!(params, Dict(:parts=>parts))
    SegregatedSolver.initial_velocity_wall_distance(params, walltag)

end

with_debug() do distribute
    main_wd_test(nparts,distribute)
end

#mpiexecjl --project=. -n 4 julia test/FindWallDistance.jl
