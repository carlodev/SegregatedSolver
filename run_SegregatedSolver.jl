using SegregatedSolver
using Parameters,PartitionedArrays

params = Dict(
      :N => 50,
      :D => 2, #Dimension
      :order => 1, 
      :t0 => 0.0,
      :dt => 0.01,
      :tF => 10.0,
      :case => "Airfoil",
      :θ => 0.5,
      :u_in=>1.0,
      :M=> 6, #internal iterations
      :backend => with_mpi,  #or with_mpi() with_debug()
      :rank_partition=>(2,2),
      :ν => 0.001,
      :petsc_options => "",
      :method=>:VMS,
      :Cᵢ => [4, 36],
      :benchmark=>false,
      :t_endramp=> 5.0,
      :mesh_file => "DU89_2D_A1_F.msh",
      :TI => 0.001,
      :ρ=>1.0,
      :Re=> 100_000,
      :c=>1.0,
      :restart=> false,
)


SegregatedSolver.main(params)


#mpiexecjl --project=. -n 4 julia run_SegregatedSolver.jl
