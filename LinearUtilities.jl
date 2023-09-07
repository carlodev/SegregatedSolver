# #MPI Backend
# """
# It instantiate an AbstactVector of 4 elements. Each element is an AbstractVector where the values of the velocity in each node is stored.
# The first element refers to the time step n, the second to the time step n-1, the third n-2 and the last n-3.
# """
# function create_ũ_vector(zfields::MPIArray)
#     zfv = deepcopy(get_part(zfields))
#     zfv1 = get_free_dof_values(zfv)
#     return [zfv1, zfv1, zfv1, zfv1]
# end

# """
# It updates the vector which stores the values of velocity at previous time steps.
# """
# function  update_ũ_vector!(ũ_vec::Vector{Vector{Float64}}, uhfields::MPIArray)
#     uh_new = get_free_dof_values(get_part(uhfields))
#     circshift!(ũ_vec,1)
#     ũ_vec[1] = deepcopy(uh_new)
# end

# """
# It updates the convective velocity exitmation ``\\tilde{u}`` for the approximation of the non linear term: ``\\tilde{u} \\cdot \\nabla(u)``
# """
# function update_ũ(ũ_vec::Vector{Vector{Float64}}, coeff::Vector{Float64})
#   println("update u")
#     updt_ũ = ũ_vec[1]*coeff[1] + ũ_vec[2] *coeff[2] + ũ_vec[3] *coeff[3] + ũ_vec[4]*coeff[4]
#     return updt_ũ
# end


# function update_free_values!(zfields::MPIArray, zt::Vector{Float64})
#     copyto!(zfields.part.free_values, zt)
# end




# #Sequential Backend
# """
#   create_ũ_vector(zfields::DebugArray)
# """
# function create_ũ_vector(zfields::DebugArray)
#     u_vec = Vector[]
#     for p = 1:1:length(zfields.parts)
#       zfv = get_free_dof_values(get_part(zfields,p))
#       push!(u_vec, [zfv, zfv, zfv, zfv])
#     end
#   return u_vec  
# end

# function update_ũ_vector!(ũ_vec::Vector{Vector}, zfields::DebugArray)
# for p = 1:1:length(zfields.parts)
#       zfv = get_free_dof_values(get_part(zfields,p))
#       circshift!(ũ_vec[p],1)
#       ũ_vec[p][1] = deepcopy(zfv)
#     end
# end
  


# function update_free_values!(zfields::DebugArray, zt::Vector{Vector})
#     for p = 1:1:length(zfields.parts)
#       copyto!(zfields.parts[p].free_values, zt[p])
#     end
# end



#### Segregated
#Sequential

function create_ũ_vector(zfv1::AbstractVector)
    return [deepcopy(zfv1), deepcopy(zfv1), deepcopy(zfv1), deepcopy(zfv1)]
end


"""
It updates the vector which stores the values of velocity at previous time steps.
"""
# function  update_ũ_vector!(ũ_vec::Vector{Vector{Float64}}, uh_new::Vector{Float64})
#     circshift!(ũ_vec,-1)
#     ũ_vec[1] = deepcopy(uh_new)
# end


function  update_ũ_vector!(ũ_vec::Vector, uh_new::AbstractVector)
  circshift!(ũ_vec,-1)
  ũ_vec[1] = deepcopy(uh_new)
end

function update_ũ(ũ_vec::Vector, coeff::Vector{Float64})
  println("update u")
    updt_ũ = ũ_vec[1]*coeff[1] + ũ_vec[2] *coeff[2] + ũ_vec[3] *coeff[3] + ũ_vec[4]*coeff[4]
    return updt_ũ
end

# """
# It updates the convective velocity exitmation ``\\tilde{u}`` for the approximation of the non linear term: ``\\tilde{u} \\cdot \\nabla(u)``
# """
# function update_ũ(ũ_vec::Vector{Vector{Float64}}, coeff::Vector{Float64})
#     updt_ũ = ũ_vec[1]*coeff[1] + ũ_vec[2] *coeff[2] + ũ_vec[3] *coeff[3] + ũ_vec[4]*coeff[4]
#     return updt_ũ
# end


# function update_free_values!(zfields::SingleFieldFEFunction, zt::Vector{Float64})
#     copyto!(zfields.free_values, zt)
# end


"""
    update_linear!(params::Dict{Symbol,Any})

Wrapper function for updating the ũ_vector containing the speed values at previous time steps 
and ũ which is the approximation of ũ for the non linear terms, ũ⋅(∇u)
"""
function update_linear!(params::Dict{Symbol,Any}, uh_tn)
  if params[:linear]
    update_ũ_vector!(params[:ũ_vector], uh_tn.fields)
    zt = update_ũ(params[:ũ_vector], params[:ũ_coeff])
    update_free_values!(params[:ũ].fields, zt)
  end
end


  
function GridapDistributed.change_ghost(a::PVector,b::AbstractArray)
  GridapDistributed.change_ghost(a,PRange(b))
end