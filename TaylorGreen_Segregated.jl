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

using GridapDistributed: Algebra
using Gridap:FESpaces
using Gridap.Arrays
using Gridap.CellData
using GridapSolvers.LinearSolvers

include("AnalyticalSolution.jl")
include("SpaceConditions.jl")
include("parallel_matrix.jl")
include("AddNewTags.jl")
include("HParam.jl")

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
      :N => 250,
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
    h = h_param(Ω, D)
  
    U0 = U(0.0)
    P0 = P(0.0)
    X0 = X(0.0)
  
  
    #Assemble Matrices
  
  
    time_step = dt:dt:tF
  
  
  
    #SEGREGATED 
    uh0 = interpolate_everywhere(velocity(0.0), U(0.0))
    ph0 = interpolate_everywhere(pa(0.0), P(0.0))
  
    u_adv = interpolate_everywhere(velocity(0.0), U(0))
    
    trials = [U,P]
    tests = [V,Q]
  
    merge!(params, Dict(:h=>h,:dΩ=>dΩ))
  
    Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
    Mat_ML, Mat_inv_ML, Mat_S, vec_Auu, vec_Aup,vec_Apu, vec_App = initialize_matrices_and_vectors(trials,tests, 0.0, u_adv, params)
  
    vec_Ap = vec_App+ vec_Apu
    vec_Au = vec_Aup+ vec_Auu
  
  
    #Vectors initialization
  
  
  
    vec_pm = GridapDistributed.change_ghost(get_free_dof_values(ph0), Mat_Aup.col_partition)
    vec_um = GridapDistributed.change_ghost(get_free_dof_values(uh0), Mat_Auu.col_partition)
  
  
    vec_am = pzeros(Mat_ML.col_partition)
    vec_sum_pm = pzeros(Mat_Aup.col_partition)
    Δa_star = pzeros(Mat_Apu.col_partition)
    Δpm1 = pzeros(Mat_S.col_partition)
    Δa = pzeros(Mat_Tpu.col_partition)
  
    b1 = pzeros(vec_Au.index_partition)
    b2 = pzeros(vec_Ap.index_partition)
  
  
    function vel_kspsetup(ksp)
      pc = Ref{GridapPETSc.PETSC.PC}()
      @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[], GridapPETSc.PETSC.KSPGMRES)
      @check_error_code GridapPETSc.PETSC.KSPSetReusePreconditioner(ksp[], GridapPETSc.PETSC.PETSC_TRUE)
      @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[], pc)
      @check_error_code GridapPETSc.PETSC.PCSetType(pc[], GridapPETSc.PETSC.PCGAMG)
      # 
    
    end
    
    function pres_kspsetup(ksp)
      pc = Ref{GridapPETSc.PETSC.PC}()
      @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[], GridapPETSc.PETSC.KSPCG)
      @check_error_code GridapPETSc.PETSC.KSPSetReusePreconditioner(ksp[], GridapPETSc.PETSC.PETSC_TRUE)
      @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[], pc)
      @check_error_code GridapPETSc.PETSC.PCSetType(pc[], GridapPETSc.PETSC.PCGAMG)
      # 
    
    end
    
    M = 6
    # options = "-log_view"
    options = ""

  
  
    for tn in time_step
      err = 1
      m = 0
      GridapPETSc.with(args=split(options)) do
        solver_vel = PETScLinearSolver(vel_kspsetup)
        ss1 = symbolic_setup(solver_vel, Mat_ML)
        ns1 = numerical_setup(ss1, Mat_ML)

        solver_pres = PETScLinearSolver(pres_kspsetup)
        ss2 = symbolic_setup(solver_pres, Mat_S)
        ns2 = numerical_setup(ss2, Mat_S)

        time_solve = @elapsed begin 
      while m<=M


          vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_Aup.col_partition)
          vec_um = GridapDistributed.change_ghost(vec_um, Mat_Auu.col_partition)
          vec_am = GridapDistributed.change_ghost(vec_am, Mat_ML.col_partition)
          @time Δpm1 .=  pzeros(Mat_S.col_partition)
  
          println("solving velocity")
            
            @time b1 .= -Mat_Auu * vec_um - Mat_Aup * vec_pm - Mat_ML * vec_am +
            Mat_Auu * dt * vec_am + (1 - θ) * Mat_Aup * vec_sum_pm + vec_Au
  
            @time Δa_star = LinearSolvers.allocate_col_vector(Mat_ML)
            @time solve!(Δa_star,ns1,b1)
            
  
  
          vec_um = GridapDistributed.change_ghost(vec_um, Mat_Apu.col_partition)
          vec_pm = GridapDistributed.change_ghost(vec_pm, Mat_App.col_partition)
          vec_am = GridapDistributed.change_ghost(vec_am, Mat_Tpu.col_partition)
  
  
          println("solving pressure")
  
  
          @time Δa_star = GridapDistributed.change_ghost(Δa_star, Mat_Tpu.col_partition)

          #-Vec_A because changing sign in the continuity equations
          @time b2 .= Mat_Tpu * Δa_star + Mat_Apu * (vec_um + dt * Δa_star) + Mat_App * vec_pm + Mat_Tpu * vec_am - vec_Ap

          @time solve!(Δpm1,ns2,b2)
  
        
  
   
  
  
  
        println("update end")
        Δpm1 = GridapDistributed.change_ghost(Δpm1, Mat_Aup.col_partition)
  
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
    
      println("update_matrices")
  @time begin
    Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
    Mat_ML, Mat_inv_ML, Mat_S, vec_Auu, vec_Aup,vec_Apu, vec_App = initialize_matrices_and_vectors(trials,tests, tn, uh_tn, params)
    
    vec_Ap = vec_App+ vec_Apu
    vec_Au = vec_Aup+ vec_Auu
 
  end

      # writevtk(Ω, "TG_segregated_$tn.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> velocity(tn), "ph" => ph_tn, "ph_analytic"=> pa(tn)])
    end #end for
  
  end #end with_backend
  
  
function GridapDistributed.change_ghost(a::PVector,b::AbstractArray)
  GridapDistributed.change_ghost(a,PRange(b))
end

with_mpi() do distribute
  main(rank_partition,distribute)
end

# with_debug() do distribute
#   main(rank_partition,distribute)
# end

#mpiexecjl --project=. -n 4 julia TaylorGreen_Segregated_v2.jl