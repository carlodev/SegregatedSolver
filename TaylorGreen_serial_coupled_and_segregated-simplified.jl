include("../../ExoFlow.jl")
include("../SpaceConditions.jl")
include("../AnalyticalSolution.jl")



params = Dict(
  :N => 32,
  :D => 2, #Dimension
  :order => 1,
  
  :t0 => 0.0,
  :dt => 0.01,
  :tF => 0.1,
  :t_endramp => 1.0,


  :case => "TaylorGreen",
  :solver => :petsc,
  :method => :VMS,
  :ode_method => :AlphaMethod,
  :θ=>1.0,
  :ρ∞ => 0.8,
  :Re => 100_000,
  :c => 1, #chord lenght [m], used for naca (=1), cylinder (=0.1), liddriven = (1), 0.5
  :u_in => 1.0,  # =1.0 for lid driven 
  :periodic => false,

  :printmodel => true,
  
  :mesh_gen => false,

  :linear => false,
  :steady => false,

  :debug_mode => false,

  :mesh_file => "NACA0012_2D_improved.msh",
  :Cᵢ => [4, 36],    
  :options => petsc_options(),
  :nls_trace =>true,
  :nls_iter => 20,

  :ν => 1.0e-5,  #channel = 0.0001472, 
  :ρ => 1.0, #kg/m3 density
  :body_force => 0.0, #channel = 0.00337204
  
  :np_x => 2, #number of processors in X
  :np_y => 2, #number of processors in Y
  :np_z => 1, #number of processors in Z

  :restart => false,
  :restart_file => "Du89_2p1.csv",
  :TI =>0.01,

  )
  
  params= initialize_parameters(params)
  diameter = 0.5 #0.5 [m] vortex dimension

  Vs = 1 #1[m/s]swirling speed
  Ua = 0.3 #0.3 [m/s]convective velocity in x
  Va = 0.2 #0.2 [m/s]convective velocity in y
  params[:ν] = 0.001 #0.001 m2/s 
  #Domain and mesh definition

    #Domain and mesh definition
    domain = (-diameter, diameter, -diameter, diameter)
    partition = (params[:N], params[:N])
    model = CartesianDiscreteModel(domain, partition; isperiodic=(true, true))
  
    hf_gen!(params)
    velocity, pa, ωa = analytical_solution(diameter, Vs, Ua, Va, params[:ν])
    merge!(params, Dict(:u0 => velocity, :model => model))
    V, Q, U, P, Y, X, model = CreateTGSpaces(model, params, pa) #We update model with the new label of the center point

    println("spaces created")
  
  
    printmodel(params, model)
  
  
  
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
  
@unpack D, N, t0,dt,tF,ν,θ = params
  dt = 0.01
  h = h_param(Ω, D)
  U0 = U(0.0)
  P0 = P(0.0)
  X0 = X(0.0)
    
  uh0 = interpolate_everywhere(velocity(0.0), U(0.0))
  ph0 = interpolate_everywhere(pa(0.0), P(0.0))
  xh0 = interpolate_everywhere([uh0, ph0], X0)
  
  
  
  val(x) = x
  val(x::Gridap.Fields.ForwardDiff.Dual) = x.value
  function τsu(u, h)
      r = 1
      τ₂ = h^2 / (4*ν)
      τ₃ = dt / 2
  
      u = val(norm(u))
      if iszero(u)
        println("is zero")
          return (1/τ₂^r  + 1/τ₃^r)^(-1/r)
      end
      τ₁ = h / (2 * u)
      return (1 / τ₁^r + 1 / τ₂^r + 1 / τ₃^r)^(-1/r)
  
  end
  
  function τb(u, h)
    return (u ⋅ u) * τsu(u, h)
  end

mutable struct assembly_spaces
    Q#::Gridap.FESpaces.UnconstrainedFESpace
    V#::Gridap.FESpaces.UnconstrainedFESpace
    P0
    U0
    
    UV
    PV
    UQ
    PQ
    function assembly_spaces(Q,V,P0,U0)
      Tm = Gridap.Algebra.SparseMatrixCSC{Float64,Int32}
      Tv = Vector{Float64}
        
      UV = Gridap.FESpaces.SparseMatrixAssembler(Tm, Tv, U0, V)
      PV = Gridap.FESpaces.SparseMatrixAssembler(Tm, Tv, P0, V)
      UQ = Gridap.FESpaces.SparseMatrixAssembler(Tm, Tv, U0, Q)
      PQ = Gridap.FESpaces.SparseMatrixAssembler(Tm, Tv, P0, Q)
      
      new(Q,V,P0,U0,UV,PV,UQ,PQ)
    end
end
  


#Assemble Matrices

u_dofs = size(get_free_dof_ids(U0))[1]
p_dofs = size(get_free_dof_ids(P0))[1]


u_adv = interpolate_everywhere(velocity(0.0), U(0.0))
as = assembly_spaces(Q,V,P0,U0)


time_step = dt:dt:100*dt


#COUPLED
#Assemble Matrices
Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, Mat_Tpp, Mat_Tup = initialize_Mat(p_dofs, u_dofs; formulation = :coupled)

update_Mat!(Mat_Tuu, Mat_Tpu, Mat_Auu,  Mat_Aup,  Mat_Apu,  Mat_App, u_adv, as; simplified = false)



#Asseble vectors
vec_pm, vec_sum_pm, vec_um, vec_am = initialize_vec(ph0, uh0)
vec_vec_um = [vec_um vec_um vec_um vec_um] 
tn0 = 0.0

uh_tn = FEFunction(U(tn0), vec_um) 

ph_tn = FEFunction(P(tn0), vec_pm)

tsu = τsu∘(u_adv, h)
writevtk(Ω, "TG_coupled_$tn0.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> velocity(tn0), "ph_analytic"=> pa(tn0), "ph" => ph_tn, "tau_su"=>tsu])



Mat_Aee = [Mat_App Mat_Apu; Mat_Aup Mat_Auu]
Mat_Tee = [Mat_Tpp Mat_Tpu; Mat_Tup Mat_Tuu]




using Plots, PyPlot
PyPlot.spy(Mat_Aee)

#Initial Solution
vec_xn0 = Vector([vec_pm; vec_um])
θ=1.0

vθ= [ones(p_dofs); θ.*ones(u_dofs)]
#Left side matrix
Mat_AT = (1 ./dt) .*Mat_Tee + vθ.*Mat_Aee 
norm(Mat_Aee)
norm(A_sysmat)

norm(Mat_Tee)
norm(T_sysmat)

using Statistics
#Right side terms
b1 = (Mat_AT - Mat_Aee) * vec_xn0


using IterativeSolvers, Statistics, LinearAlgebra, IncompleteLU


coeff = [2.1875, -2.1875, 1.3125, -0.3125]



cconv(u_adv, ∇u) = u_adv ⋅ (∇u)

Tuu(u, v) = ∫((v + τsu∘(u_adv, h) * (cconv ∘ (u_adv, ∇(v)))) ⊙ u)dΩ
Tpu(u, q) = ∫((τsu∘(u_adv, h)) * (∇(q)) ⊙ u)dΩ
Auu1(u,v) = ∫(ν* ∇(v) ⊙ ∇(u) + (cconv ∘ (u_adv, ∇(u))) ⋅ v  + ((τsu∘(u_adv, h)) *(cconv ∘ (u_adv, ∇(v)))) ⊙(cconv ∘ (u_adv, ∇(u))))dΩ

Auu2(u,v) = ∫(((τb∘(u_adv, h)) * (∇⋅v)) ⊙ (∇ ⋅ u) + 0.5 .*u_adv⋅(v + (τsu∘(u_adv, h))*(cconv ∘ (u_adv, ∇(v))))⋅(∇ ⋅ u))dΩ

Auu(u,v) = Auu1(u,v) + Auu2(u,v)

Aup(p, v) = ∫( - (∇ ⋅ v) * p + ((τsu∘(u_adv, h))*(cconv ∘ (u_adv, ∇(v)))) ⊙ ∇(p))dΩ

Apu(u, q) = ∫( q *(∇ ⋅ u)+  0.5 .* (τsu∘(u_adv, h))⋅(∇(q))⋅u_adv⋅(∇ ⋅ u) + (τsu∘(u_adv, h))* (∇(q)) ⊙ (cconv ∘ (u_adv, ∇(u))))dΩ

App(p,q) =  ∫(((τsu∘(u_adv, h)) * ∇(q)) ⊙ (∇(p)) )dΩ



function define_vectors(t, V, Q, P, U)
  rhs(v) = 0.0
  vec_Auu = get_vector(AffineFEOperator(Auu, rhs, U(t),V))
  vec_Aup = get_vector(AffineFEOperator(Aup, rhs, P(t),V))
  vec_Apu = get_vector(AffineFEOperator(Apu, rhs, U(t),Q))
  vec_App = get_vector(AffineFEOperator(App, rhs, P(t),Q))

  vec_Tuu = get_vector(AffineFEOperator(Tuu, rhs, U(t), V))
  vec_Tpu = get_vector(AffineFEOperator(Tpu, rhs, U(t), Q))

return [vec_App + vec_Apu; vec_Aup + vec_Auu], [vec_Tpu; vec_Tuu]
end


vec_A, vec_T = define_vectors(0.0, V, Q, P, U)

for tn in time_step
    
Pl = IncompleteLU.ilu(Mat_AT, τ=0.0001)
    vec_xn1, history = IterativeSolvers.gmres(Mat_AT, b1 .+ vec_A; Pl = Pl, restart = 30, maxiter= 1500, abstol=1e-12, reltol=1e-12, log = true)
    

    println(history)
  
    vec_pm = vec_xn1[1:p_dofs]
    vec_um = vec_xn1[p_dofs+1:end]
    
    
    println(Statistics.maximum(vec_xn1[1:p_dofs]))
    println(Statistics.minimum(vec_xn1[1:p_dofs]))

    println("Time =$tn")
    
    update_vec_vec_um!(vec_vec_um, vec_um)
    update_vec_um!(vec_um, vec_vec_um, coeff)
    
 
   
    u_adv =  FEFunction(U(tn), vec_um) 
    uh_tn = FEFunction(U(tn), vec_um) 
    ph_tn = FEFunction(P(tn), vec_pm) 
    auu1 = norm(Mat_Auu)
    update_Mat!(Mat_Tuu, Mat_Tpu, Mat_Auu,  Mat_Aup,  Mat_Apu,  Mat_App, u_adv, as; simplified = false)
    auu2 = norm(Mat_Auu)
    println("Matrix norm update = $(auu1 -auu2)")
    Mat_Aee[:,:] = [Mat_App Mat_Apu; Mat_Aup Mat_Auu]
    Mat_Tee[:,:] = [Mat_Tpp Mat_Tpu; Mat_Tup Mat_Tuu]
    Mat_AT[:,:] = (1 ./dt) .*Mat_Tee +vθ.*Mat_Aee
    
    b1 .= (Mat_AT - Mat_Aee) * vec_xn1
    
   
    vec_A, vec_T = define_vectors(tn, V, Q, P, U)

# writevtk(Ω, "TG_coupled_$tn.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> velocity(tn), "ph_analytic"=> pa(tn), "ph" => ph_tn])#end

end


S(p,q) =   -θ * ∫((dt .+ τsu∘(u_adv, h) ) ⋅ ((∇(q)') ⊙ (∇(p)) ))dΩ
θ = 0.5

#SEGREGATED 
uh0 = interpolate_everywhere(velocity(0.0), U(0.0))
ph0 = interpolate_everywhere(pa(0.0), P(0.0))
uh1 = interpolate_everywhere(velocity(dt), U(0.0))
ph1 = interpolate_everywhere(pa(dt), P(0.0))


u_adv = interpolate_everywhere(velocity(0.0), U(0))
Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, Mat_ML, Mat_inv_ML = initialize_Mat(p_dofs, u_dofs; formulation = :segregated)

update_Mat!(Mat_Tuu, Mat_Tpu, Mat_Auu,  Mat_Aup,  Mat_Apu,  Mat_App, u_adv, as; simplified = false)
Mat_ML[:,:]  = Mat_Tuu .+ θ * dt .* Mat_Auu #It is not in update_Mat because this matrix is not computed in the coupled case

inv_lump_vel_mass!(Mat_inv_ML, Mat_ML)

println(typeof(Mat_inv_ML))

#To verify that the inverse is ok
x_test = rand(u_dofs)
N = Mat_ML*x_test
x_test_res = Mat_inv_ML*N
x_test 
x_test_res
maximum(abs.(x_test - x_test_res))<1e-14


#Vectors

vec_pm, vec_sum_pm, vec_um, vec_am = initialize_vec(ph0, uh0)
vec_A, vec_T = define_vectors(0.0, V, Q, P, U)
vec_vec_um = [vec_um vec_um vec_um vec_um]
Δa_star =spzeros(Float64, Int32, u_dofs) #Inital Non zero
Δpm1 = zeros(p_dofs)
b1 = zeros(u_dofs)
A2 = similar(Mat_Tpp)
b2 = zeros(p_dofs)

vec_pm
vec_um

M = 20
a_target =  (get_free_dof_values(uh1)-get_free_dof_values(uh0))./dt
dp_target = (get_free_dof_values(ph1)-get_free_dof_values(ph0))
a_star_target = a_target + Mat_inv_ML.*θ*Mat_Aup*dp_target

for tn in time_step
for m = 0:1:M-1

  b1 .= - Mat_Auu * vec_um - Mat_Aup * vec_pm - Mat_ML * vec_am +
              Mat_Auu * dt * vec_am + (1 - θ) * Mat_Aup * vec_sum_pm + vec_A[p_dofs+1:end]

    Pl1 = IncompleteLU.ilu(Mat_ML, τ=0.0001)
    Δa_star,history = IterativeSolvers.gmres(Mat_ML, b1; Pl = Pl1, abstol = 1e-12, log=true)
    history
    norm(Δa_star-a_star_target)./norm(a_star_target)*100
    
    # A2[:,:] = (Mat_Tpu + dt * Mat_Apu) * Mat_inv_ML * θ * Mat_Aup - Mat_App
    A2[:,:] = assemble_matrix(S,P(tn),Q)

    #-Vec_A because changing sign in the continuity equations
    b2 .= Mat_Tpu*Δa_star +  Mat_Apu * (vec_um .+ dt .* Δa_star) + Mat_App * vec_pm + Mat_Tpu*vec_am - vec_A[1:p_dofs]
    
    Pl2 = IncompleteLU.ilu(A2, τ=0.0001)
    Δpm1, history =  IterativeSolvers.gmres(A2, b2; Pl = Pl2, abstol = 1e-12, log = true)

    history
    Δa = Δa_star - Mat_inv_ML * θ * Mat_Aup * Δpm1
    
    vec_um .= vec_um + dt * Δa
    vec_pm .= vec_pm + Δpm1

    println(Statistics.maximum(vec_pm))
    println(Statistics.minimum(vec_pm))
println("inner iter = $m")
    if m == 0
        vec_sum_pm .= Δpm1
        vec_am .=Δa
    else
        vec_sum_pm .+=  Δpm1
        vec_am .+= Δa

    end
    
end  


        u_adv =  FEFunction(U(tn), vec_um) 
        update_Mat!(Mat_Tuu, Mat_Tpu, Mat_Auu,  Mat_Aup,  Mat_Apu,  Mat_App, u_adv, as)
        Mat_ML[:,:]  = Mat_Tuu .+ θ * dt .* Mat_Auu
        inv_lump_vel_mass!(Mat_inv_ML,  Mat_ML)
        
        vec_A, vec_T = define_vectors(tn, V, Q, P, U)


#end
    uh_tn = FEFunction(U(tn),vec_um)
    ph_tn = FEFunction(P(tn),vec_pm)

writevtk(Ω, "TG_segregated_$tn.vtu", cellfields = ["uh" => uh_tn, "uh_analytic"=> velocity(tn), "ph" => ph_tn, "ph_analytic"=> pa(tn)])
end
