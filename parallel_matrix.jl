
function allocate_Mat_inv_ML(Mat_ML::PSparseMatrix) 
  return PVector(0.0,Mat_ML.rows)
end

function allocate_Mat_inv_ML(Mat_ML::SparseMatrixCSC) 
  l = size(Mat_ML)[1]

  return zeros(l)
end

function inv_lump_vel_mass!(Mat_inv_ML::PVector,Mat_ML::PSparseMatrix)
    values = map_parts(Mat_ML.values) do val
        N = maximum(rowvals(val))
    
        V = zeros(N)
        vals = nonzeros(val)
        
        j = 1
        for i in rowvals(val)
            V[i] += vals[j]
            j+1
        end
        V = 1 ./V

    end
    Mat_inv_ML .= PVector(values,Mat_ML.rows) 
end


function inv_lump_vel_mass!(Mat_inv_ML::Vector, Mat_ML::SparseMatrixCSC)
  inv_ML_vec = 1 ./ sum(Mat_ML, dims=2)[:,1]
      if !isempty(inv_ML_vec[inv_ML_vec.==Inf])
      error("The matrix ML can not be inverted because after lumping zero values are detected")
  end
  
  Mat_inv_ML.=inv_ML_vec
  
end




function initialize_matrices_and_vectors(trials,tests, t::Real, u_adv, params)
  
  return _matrices_and_vectors!(trials,tests, t, u_adv,params)
end

function _matrices_and_vectors!(trials, tests, t::Real, u_adv, params)
    @unpack ν, dt,h, dΩ, θ = params

    cconv(uadv, ∇u) = uadv ⋅ (∇u)

    val(x) = x
    val(x::Gridap.Fields.ForwardDiff.Dual) = x.value
    function τsu(u, h)
      r = 1
      τ₂ = h^2 / (4 * ν)
      τ₃ = dt / 2
    
      u = val(norm(u))
      if iszero(u)
        println("is zero")
        return (1 / τ₂^r + 1 / τ₃^r)^(-1 / r)
      end
      τ₁ = h / (2 * u)
      return (1 / τ₁^r + 1 / τ₂^r + 1 / τ₃^r)^(-1 / r)
    
    end
    
    function τb(u, h)
      return (u ⋅ u) * τsu(u, h)
    end
    
    
    Tuu(u, v) = ∫((v + τsu ∘ (u_adv, h) * (cconv ∘ (u_adv, ∇(v)))) ⊙ u)dΩ
    Tpu(u, q) = ∫((τsu ∘ (u_adv, h)) * (∇(q)) ⊙ u)dΩ
    Auu1(u, v) = ∫(ν * ∇(v) ⊙ ∇(u) + (cconv ∘ (u_adv, ∇(u))) ⋅ v + ((τsu ∘ (u_adv, h)) * (cconv ∘ (u_adv, ∇(v)))) ⊙ (cconv ∘ (u_adv, ∇(u))))dΩ
  
    Auu2(u, v) = ∫(((τb ∘ (u_adv, h)) * (∇ ⋅ v)) ⊙ (∇ ⋅ u) + 0.5 .* u_adv ⋅ (v + (τsu ∘ (u_adv, h)) * (cconv ∘ (u_adv, ∇(v)))) ⋅ (∇ ⋅ u))dΩ
  
    Auu(u, v) = Auu1(u, v) + Auu2(u, v)
  
    Aup(p, v) = ∫(-(∇ ⋅ v) * p + ((τsu ∘ (u_adv, h)) * (cconv ∘ (u_adv, ∇(v)))) ⊙ ∇(p))dΩ
  
    Apu(u, q) = ∫(q * (∇ ⋅ u) + 0.5 .* (τsu ∘ (u_adv, h)) ⋅ (∇(q)) ⋅ u_adv ⋅ (∇ ⋅ u) + (τsu ∘ (u_adv, h)) * (∇(q)) ⊙ (cconv ∘ (u_adv, ∇(u))))dΩ
  
    App(p, q) = ∫(((τsu ∘ (u_adv, h)) * ∇(q)) ⊙ (∇(p)))dΩ
  
    ML(u, v) = Tuu(u, v) + (θ * dt) * Auu(u, v)
  
    S(p, q) = -θ * ∫((dt .+ τsu ∘ (u_adv, h)) ⋅ ((∇(q)') ⊙ (∇(p))))dΩ

    rhs(v) = 0.0

    U,P = trials
    V,Q = tests

    # if Vecs === nothing

      Af_Tuu = AffineFEOperator(Tuu,rhs,U(t),V)
      Af_Tpu = AffineFEOperator(Tpu,rhs,U(t),Q)

      Af_Auu = AffineFEOperator(Auu,rhs,U(t),V)
      Af_Aup = AffineFEOperator(Aup,rhs,P(t),V)
      Af_Apu = AffineFEOperator(Apu,rhs,U(t),Q)
      Af_App = AffineFEOperator(App,rhs,P(t),Q)

      Af_ML = AffineFEOperator(ML,rhs,U(t),V)
      Af_S = AffineFEOperator(S,rhs,P(t),Q)

      Mat_Tuu = get_matrix(Af_Tuu)
      Mat_Tpu = get_matrix(Af_Tpu)

      Mat_Auu = get_matrix(Af_Auu)
      Mat_Aup = get_matrix(Af_Aup)
      Mat_Apu = get_matrix(Af_Apu)
      Mat_App = get_matrix(Af_App)
  
      Mat_ML = get_matrix(Af_ML)
      Mat_S = get_matrix(Af_S)
      
      Vec_Auu = get_vector(Af_Auu)
      Vec_Aup = get_vector(Af_Aup)
      Vec_Apu = get_vector(Af_Apu)
      Vec_App = get_vector(Af_App)

      Mat_inv_ML = allocate_Mat_inv_ML(Mat_ML)
      inv_lump_vel_mass!(Mat_inv_ML,Mat_ML)

      Vec_Ap = Vec_Apu + Vec_App
      Vec_Au = Vec_Auu + Vec_Aup
      return  Mat_Tuu, Mat_Tpu, Mat_Auu, Mat_Aup, Mat_Apu, Mat_App, Mat_ML, Mat_inv_ML, Mat_S, Vec_Auu, Vec_Aup,Vec_Apu, Vec_App

      # else

      # Vec_Auu, Vec_Aup, Vec_Apu, Vec_App = Vecs
      # println("t = $t")
      # update_vector!(Auu,rhs,Vec_Auu,U,V,t)
      # update_vector!(Aup,rhs,Vec_Aup,P,V,t)
      # update_vector!(Apu,rhs,Vec_Apu,U,Q,t)
      # update_vector!(App,rhs,Vec_App,P,Q,t)
    
      # println("update matrix finished")
         
    end
    


function update_matrix_vector!(a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,trial::GridapDistributed.DistributedSingleFieldFESpace,test::GridapDistributed.DistributedSingleFieldFESpace,t::Real)
 
end


function update_matrix_vector!(a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,trial::FESpace,test::FESpace,t::Real)
 
end

function update_matrixvector!(a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,trial::TransientTrialFESpace,test::Union{GridapDistributed.DistributedSingleFieldFESpace, FESpace},t::Real)
  afop = AffineFEOperator(a,l,trial(t),test)
  println(norm(get_vector(afop)))
  b.= get_vector(afop)
end