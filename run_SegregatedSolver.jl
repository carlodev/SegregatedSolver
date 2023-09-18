using SegregatedSolver
using Parameters,PartitionedArrays

 petsc_options = " -vel_ksp_type gmres -vel_pc_type gamg -vel_ksp_rtol 1.e-10 -vel_ksp_converged_reason \
                   -pres_ksp_type cg -pres_pc_type gamg  -pres_ksp_rtol 1.e-6 -pres_ksp_converged_reason \
                   -ksp_atol 0.0"

params = Dict(
      :N => 50,
      :D => 2, #Dimension
      :order => 1, 
      :t0 => 0.0,
      :dt => 0.002,
      :tF => 2.5,
      :case => "Airfoil",
      :θ => 0.5,
      :u_in=> 1.0,
      :M=> 5, #internal iterations
      :backend => with_debug,  #or with_mpi() with_debug()
      :rank_partition=>(2, 2),
      :ν => 0.001,
      :petsc_options => petsc_options,
      :method=>:VMS,
      :Cᵢ => [4, 36],
      :benchmark=>false,
      :t_endramp=> 0.0,
      :mesh_file => "DU89_2D_A1_v2.msh",
      :TI => 0.001,
      :ρ=>1.0,
      :Re=> 250_000,
      :c=> 0.1,
      :restart=> true,
      :restart_file=>"InitialDU89_AoA1.csv",     
)



SegregatedSolver.main(params)


#mpiexecjl --project=. -n 4 julia run_SegregatedSolver.jl
