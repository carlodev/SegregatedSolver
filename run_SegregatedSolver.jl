using SegregatedSolver
using Parameters,PartitionedArrays

petsc_options = "-vel_ksp_type gmres -vel_pc_type gamg -vel_ksp_rtol 1.e-12  \
                  -pres_ksp_type gmres -pres_pc_type gamg -pres_ksp_atol 1.e-5 \
                  -ksp_converged_reason -log_view"

params = Dict(
      :N => 50,
      :D => 2, #Dimension
      :order => 1, 
      :t0 => 0.0,
      :dt => 0.005,
      :tF => 2.5,
      :case => "Cylinder",
      :θ => 0.5,
      :u_in=> 10.0,
      :M=> 2, #internal iterations
      :backend => with_mpi,  #or with_mpi() with_debug()
      :rank_partition=>(2, 2),
      :ν => 0.001,
      :petsc_options => "-ksp_converged_reason",
      :method=>:SUPG,
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
