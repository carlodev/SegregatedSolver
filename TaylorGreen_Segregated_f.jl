using Revise
using Gridap
using GridapDistributed
using GridapGmsh
using GridapPETSc
using LinearAlgebra
using PartitionedArrays
using MPI
using Parameters
using SparseArrays

using FillArrays

using GridapDistributed: Algebra
using Gridap:FESpaces
using Gridap.Arrays
using Gridap.CellData

include(joinpath("src","Commons","CommonsProcedures.jl"))
include(joinpath("src","Commons","AddNewTags.jl"))
include(joinpath("src","Commons","StabParams.jl"))
include(joinpath("src","Commons","LinearUtilities.jl"))
include(joinpath("src","Commons","StabilizedEquations.jl"))
include(joinpath("src","Commons","SolversOptions.jl"))
include(joinpath("src","Commons","MatrixCreation.jl"))

#TaylorGreen
include(joinpath("src","TaylorGreen","TaylorGreen.jl"))
rank_partition = (2, 2)


function main(rank_partition,distribute)
  parts  = distribute(LinearIndices((prod(rank_partition),)))


    if typeof(parts) <: MPIArray
      comm = MPI.COMM_WORLD
      #To avoid multiple printing of the same line in parallel
      if MPI.Comm_rank(comm) != 0
        redirect_stderr(devnull)
        redirect_stdout(devnull)
      end
    end
  
    params = Dict(
      :N => 50,
      :D => 2, #Dimension
      :order => 1, 
      :t0 => 0.0,
      :dt => 0.01,
      :tF => 0.5,
      :case => "TaylorGreen",
      :θ => 0.5, 
      :restart=>false)

    diameter = 0.5 #0.5 [m] vortex dimension
  
    Vs = 1 #1[m/s]swirling speed
    Ua = 0.3 #0.3 [m/s]convective velocity in x
    Va = 0.2 #0.2 [m/s]convective velocity in y
    params[:ν] = 0.001 #0.001 m2/s 
  
    #Domain and mesh definition
    domain = (-diameter, diameter, -diameter, diameter)
    partition = (params[:N], params[:N])
    model = CartesianDiscreteModel(parts, rank_partition,domain,partition; isperiodic=(true, true))
  
    # hf_gen!(params)
    velocity, pa, ωa = analytical_solution(diameter, Vs, Ua, Va, params[:ν])
    merge!(params, Dict(:model => model))
    V, Q, U, P, Y, X, model = CreateTGSpaces(model, params, pa) #We update model with the new label of the center point
  
  
    println("spaces created")
  
  
  
    degree = 4 * params[:order]
    Ω = Triangulation(model)
    dΩ = Measure(Ω, degree)
    new_dict = Dict(:U => U,
      :P => P,
      :X => X,
      :Y => Y,
      :Ω => Ω,
      :dΩ => dΩ,
      :degree => degree,
      :force_params => nothing,
      :p0 => pa, :u0=>velocity)
  
    merge!(params, new_dict)
  
    @unpack D, N, t0, dt, tF, ν, θ = params
  
   
  
    #Assemble Matrices
  
  
    time_step = dt:dt:tF
  
  
  
    #SEGREGATED 
      
    trials = [U,P]
    tests = [V,Q]
  
    h = h_param(Ω, D)
    G, GG, gg = G_params(Ω,params)
        
    M = 6
    petsc_options = "-log_view"

    merge!(params, Dict(:h=>h,:G=>G, :GG=>GG, :gg=>gg, :Cᵢ=>[4,36],:dΩ=>dΩ,:trials=>trials,:tests=>tests,:M=>M,
    :petsc_options=>petsc_options,:time_step=>time_step,:benchmark=>false,:method=>:VMS))

  
    solve_case(params)


    

  end #end with_backend
  




  with_debug() do distribute
    main(rank_partition,distribute)
 end

#  with_mpi() do distribute
#   main(rank_partition,distribute)
# end
#mpiexecjl --project=. -n 4 julia TaylorGreen_Segregated.jl


