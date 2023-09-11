
#### Segregated
#Sequential

function create_ũ_vector(zfv1::AbstractVector)
    return [deepcopy(zfv1), deepcopy(zfv1), deepcopy(zfv1), deepcopy(zfv1)]
end


"""
It updates the vector which stores the values of velocity at previous time steps.
"""
function  update_ũ_vector!(ũ_vec::Vector, uh_new::AbstractVector)
  circshift!(ũ_vec,-1)
  ũ_vec[1] = deepcopy(uh_new)
end


function update_ũ(ũ_vec::Vector)
  coeff = [2.1875, -2.1875, 1.3125, -0.3125]
  updt_ũ = ũ_vec[1]*coeff[1] + ũ_vec[2] *coeff[2] + ũ_vec[3] *coeff[3] + ũ_vec[4]*coeff[4]
    return updt_ũ
end


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




#Extensions for scripts semplification
function GridapDistributed.change_ghost(a::PVector,M::PSparseMatrix)
  col_part = M.col_partition
  GridapDistributed.change_ghost(a,col_part)
end
  
function GridapDistributed.change_ghost(a::PVector,b::AbstractArray)
  GridapDistributed.change_ghost(a,PRange(b))
end

function PartitionedArrays.pzeros(M::PSparseMatrix)
  pzeros(M.col_partition)
end

function PartitionedArrays.pzeros(a::PVector)
  pzeros(a.index_partition)
end