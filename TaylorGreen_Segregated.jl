using Pkg
Pkg.activate(".")

using Revise
using Gridap
using GridapDistributed
using IterativeSolvers
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

include("AnalyticalSolution.jl")
include("SpaceConditions.jl")
include("MatrixCreation.jl")
include("AddNewTags.jl")
include("StabParams.jl")
include("LinearUtilities.jl")
include("StabilizedEquations.jl")


include("SolversOptions.jl")

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
      :θ => 0.5)

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
    merge!(params, Dict(:u0 => velocity, :model => model))
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
      :p0 => pa)
  
    merge!(params, new_dict)
  
    @unpack D, N, t0, dt, tF, ν, θ = params
  
   
  
    #Assemble Matrices
  
  
    time_step = dt:dt:tF
  
  
  
    #SEGREGATED 
    uh0 = interpolate_everywhere(velocity(0.0), U(0.0))
    ph0 = interpolate_everywhere(pa(0.0), P(0.0))
  
    u_adv = interpolate_everywhere(velocity(0.0), U(0))
    
    trials = [U,P]
    tests = [V,Q]
  
    h = h_param(Ω, D)
    G, GG, gg = G_params(Ω,params)
    merge!(params, Dict(:h=>h,:G=>G, :GG=>GG, :gg=>gg, :Cᵢ=>[4,36],:dΩ=>dΩ))

    Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
    Mat_ML, Mat_inv_ML, Mat_S, vec_Au, vec_Ap = initialize_matrices_and_vectors(trials,tests, 0.0, u_adv, params; method=:VMS)
  
     
    #Vectors initialization
   
    vec_pm = GridapDistributed.change_ghost(get_free_dof_values(ph0), Mat_Aup)
    vec_um = GridapDistributed.change_ghost(get_free_dof_values(uh0), Mat_Auu)
  
  
    vec_am = pzeros(Mat_ML)
    vec_sum_pm = pzeros(Mat_Aup)
    Δa_star = pzeros(Mat_Apu)
    Δpm1 = pzeros(Mat_S)
    Δa = pzeros(Mat_Tpu)
  
    b1 = pzeros(vec_Au)
    b2 = pzeros(vec_Ap)
  
    
    M = 6
    petsc_options = "-log_view"
    # options = ""

  
    ũ_vector = create_ũ_vector(vec_um)


    for tn in time_step
      err = 1
      m = 0
      GridapPETSc.with(args=split(petsc_options)) do

        solver_vel = PETScLinearSolver(vel_kspsetup)
        ss1 = symbolic_setup(solver_vel, Mat_ML)
        ns1 = numerical_setup(ss1, Mat_ML)

        solver_pres = PETScLinearSolver(pres_kspsetup)
        ss2 = symbolic_setup(solver_pres, Mat_S)
        ns2 = numerical_setup(ss2, Mat_S)

        time_solve = @elapsed begin 
      while m<=M


          vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_Aup)
          vec_um = GridapDistributed.change_ghost(vec_um, Mat_Auu)
          vec_am = GridapDistributed.change_ghost(vec_am, Mat_ML)
          @time Δpm1 .=  pzeros(Mat_S.col_partition)
  
          println("solving velocity")
            
            @time b1 .= -Mat_Auu * vec_um - Mat_Aup * vec_pm - Mat_ML * vec_am +
            Mat_Auu * dt * vec_am + (1 - θ) * Mat_Aup * vec_sum_pm + vec_Au
  
            @time Δa_star = pzeros(Mat_ML)
            @time solve!(Δa_star,ns1,b1)
            
  
  
          vec_um = GridapDistributed.change_ghost(vec_um, Mat_Apu)
          vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_App)
          vec_am = GridapDistributed.change_ghost(vec_am, Mat_Tpu)
  
  
          println("solving pressure")
  
  
          @time Δa_star = GridapDistributed.change_ghost(Δa_star, Mat_Tpu)

          #-Vec_A because changing sign in the continuity equations
          @time b2 .= Mat_Tpu * Δa_star + Mat_Apu * (vec_um + dt * Δa_star) + Mat_App * vec_pm + Mat_Tpu * vec_am - vec_Ap

          @time solve!(Δpm1,ns2,b2)
  
        
  
   
  
  
  
        println("update end")
        Δpm1 = GridapDistributed.change_ghost(Δpm1, Mat_Aup)
  
          Δa .= Δa_star - θ .* Mat_inv_ML .* (Mat_Aup * Δpm1)
  
          vec_um .+=  dt * Δa
          vec_pm .+= Δpm1
  
  
          println("inner iter = $m")
          if m == 0
            vec_sum_pm .= Δpm1
            vec_am .= Δa
          else
            vec_sum_pm .+= Δpm1
            vec_am .+= Δa
  
          end
  
          m = m + 1
          ua_n = get_free_dof_values( interpolate_everywhere(velocity(tn), U(tn)))
          err = norm(ua_n - vec_um)
  
        println("error = $err")
      end  #end while
    end #end elapsed
    println("solution time")
    println(time_solve)
      GridapPETSc.GridapPETSc.gridap_petsc_gc()
    end #end GridapPETSc

 
      uh_tn = FEFunction(U(tn), vec_um)
      ph_tn = FEFunction(P(tn), vec_pm)
      writevtk(Ω, "TG_segregated_$tn.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> velocity(tn), "ph" => ph_tn, "ph_analytic"=> pa(tn)])


    update_ũ_vector!(ũ_vector,vec_um)
    uh_tn = FEFunction(U(tn), update_ũ(ũ_vector))

      println("update_matrices")
    @time begin

    Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
    Mat_ML, Mat_inv_ML, Mat_S, vec_Au, vec_Ap = matrices_and_vectors(trials, tests, tn, uh_tn, params; method=:VMS)
    
 
    end

    end #end for

  end #end with_backend
  


with_mpi() do distribute
  main(rank_partition,distribute)
end

   with_debug() do distribute
     main(rank_partition,distribute)
  end

 
#mpiexecjl --project=. -n 4 julia TaylorGreen_Segregated.jl
