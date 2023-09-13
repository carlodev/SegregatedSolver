using SegregatedSolver
using Parameters,PartitionedArrays

 petsc_options = " -vel_ksp_type gmres -vel_pc_type gamg    -vel_ksp_rtol 1.e-10 -vel_ksp_atol 0.0 -vel_ksp_converged_reason \
                   -pres_ksp_type preonly -pres_pc_type lu  -pres_ksp_atol 1.e-10 -pres_ksp_converged_reason"

params = Dict(
      :N => 50,
      :D => 2, #Dimension
      :order => 1, 
      :t0 => 0.0,
      :dt => 0.01,
      :tF => 2.5,
      :case => "TaylorGreen",
      :θ => 0.5,
      :u_in=> 10.0,
      :M=> 5, #internal iterations
      :backend => with_mpi,  #or with_mpi() with_debug()
      :rank_partition=>(2, 2),
      :ν => 0.001,
      :petsc_options => petsc_options,
      :method=>:VMS,
      :Cᵢ => [4, 36],
      :benchmark=>false,
      :t_endramp=> 0.0,
      :mesh_file => "Cylinder_2D.msh",
      :TI => 0.001,
      :ρ=>1.0,
      :Re=> 1_000,
      :c=> 0.1,
      :restart=> false,
)


SegregatedSolver.main(params)


#mpiexecjl --project=. -n 4 julia run_SegregatedSolver.jl
