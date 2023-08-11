using Pkg
Pkg.activate(".")


using Revise
using Gridap
using GridapDistributed
using IterativeSolvers
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


params = Dict(
      :N => 32,
      :D => 2, #Dimension
      :order => 1, 
      :t0 => 0.0,
      :dt => 0.01,
      :tF => 1.0,
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
    model = CartesianDiscreteModel(domain, partition; isperiodic=(true, true))
  
    # hf_gen!(params)
    velocity, pa, ωa = analytical_solution(diameter, Vs, Ua, Va, params[:ν])
    merge!(params, Dict(:u0 => velocity, :model => model))
    V, Q, U, P, Y, X, model = CreateTGSpaces(model, params, pa) #We update model with the new label of the center point
  
  
     
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
  
  
  
    vec_pm = get_free_dof_values(ph0)
    vec_um = get_free_dof_values(uh0)
  
    lpdofs = length(vec_pm)
    ludofs = length(vec_um)
  
    vec_am = zeros(ludofs)
    vec_sum_pm = zeros(lpdofs)
    Δa_star = zeros(ludofs)
    Δpm1 = zeros(lpdofs)
    Δa = zeros(ludofs)
  
    b1 = zeros(ludofs)
    b2 = zeros(lpdofs)
  
  
  
    M = 5
  
  
  
    for tn in time_step
      err = 1
      m = 0
      Pl = JacobiLinearSolver()
      solver_vel = LinearSolvers.GMRESSolver(20,Pl,1.e-8)
      ns1 = numerical_setup(symbolic_setup(solver_vel,Mat_ML),Mat_ML)
  
      Pl = JacobiLinearSolver()
      solver_pres = LinearSolvers.GMRESSolver(20,Pl,1.e-6)
      ns2 = numerical_setup(symbolic_setup(solver_pres,Mat_S),Mat_S)
  
      while m<=M
  

          Δpm1 .= zeros(lpdofs)
  
          println("solving velocity")
            
            @time b1 = -Mat_Auu * vec_um - Mat_Aup * vec_pm - Mat_ML * vec_am +
            Mat_Auu * dt * vec_am + (1 - θ) * Mat_Aup * vec_sum_pm + vec_Au
  
            Δa_star = LinearSolvers.allocate_col_vector(Mat_ML)
            solve!(Δa_star,ns1,b1)
              
  
          println("solving pressure")
  
  
  
          #-Vec_A because changing sign in the continuity equations
          # @time begin
          #   b2 .= Mat_Tpu * Δa_star + Mat_Apu * (vec_um + dt * Δa_star) + Mat_App * vec_pm + Mat_Tpu * vec_am - vec_Ap
  
          #   P_rich_sm  =  SymGaussSeidelSmoother(10) #RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0)
          #   ss2 = symbolic_setup(P_rich_sm,Mat_S)
          #   ns2 = numerical_setup(ss2,Mat_S)
          #   IterativeSolvers.cg!(Δpm1,Mat_S,b2;
          #   verbose=i_am_main(parts),Pl=ns2,
          #   reltol=1.0e-8,
          #   log=true)
  
          #  end #end begin
  
          b2 = Mat_Tpu * Δa_star + Mat_Apu * (vec_um + dt * Δa_star) + Mat_App * vec_pm + Mat_Tpu * vec_am - vec_Ap
            
            
            solve!(Δpm1,ns2,b2)
  
        
  
   
  
  
  
        println("update end")
  
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
  
      end #end while
  
      u_adv = FEFunction(U(tn), vec_um)
      println("update_matrices")
  
    Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, 
    Mat_ML, Mat_inv_ML, Mat_S, vec_Auu, vec_Aup,vec_Apu, vec_App = initialize_matrices_and_vectors(trials,tests, tn, u_adv, params)
    
    vec_Ap = vec_App+ vec_Apu
    vec_Au = vec_Aup+ vec_Auu
  
      uh_tn = FEFunction(U(tn), vec_um)
      ph_tn = FEFunction(P(tn), vec_pm)
  
      writevtk(Ω, "TG_segregated_$tn.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> velocity(tn), "ph" => ph_tn, "ph_analytic"=> pa(tn)])
    end #end for
  
  
