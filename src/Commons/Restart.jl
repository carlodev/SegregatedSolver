#For periodic case look for ghost nodes, that's the isempty function; The y direction is not periodic for the channel; look for genralization?

"""
    find_idx(p::VectorValue{2, Float64}, params; atol=1e-6)

For a given point `p` it search on the `restart_file` provided by the user the closer node and provides the index of such node.
It takes into account also the channel periodic case, changing the coordiantes of the periodic directions. 
"""
function find_idx(p::VectorValue{2, Float64}, params; atol=1e-6)
  vv1 = findall(x->isapprox(x, p[1];  atol=atol), params[:restart_df].Points_0) 
  vv2 = findall(x->isapprox(x, p[2];  atol=atol), params[:restart_df].Points_1)

  if isempty(vv1) && params[:case] == "Channel"
    q = -sign(p[1]) * params[:Lx] + p[1]
    vv1 = findall(x-> isapprox(x, q;  atol=atol), params[:restart_df].Points_0) 
  end

  idx = vv1[findall(in(vv2),vv1)]
  if isempty(idx)
    idx = find_idx(p, params; atol = atol*10)
  end
  
  return idx[1]
end

"""
    find_idx(p::VectorValue{3, Float64}, params; atol = 1e-4)

For a given point `p` it search on the `restart_file` provided by the user the closer node and provides the index of such node.
It takes into account also the channel periodic case, changing the coordiantes of the periodic directions. 
"""
function find_idx(p::VectorValue{3, Float64}, params; atol = 1e-4)
  vv1 = findall(x->isapprox(x, p[1];  atol=atol), params[:restart_df].Points_0) 
  vv2 = findall(x->isapprox(x, p[2];  atol=atol), params[:restart_df].Points_1)
  vv3 = findall(x->isapprox(x, p[3];  atol=atol), params[:restart_df].Points_2)
  
  if isempty(vv1) && params[:case] == "Channel"
    q = -sign(p[1]) * params[:Lx] + p[1]
    vv1 = findall(x-> isapprox(x, q;  atol=atol), params[:restart_df].Points_0) 
  end

  if isempty(vv3) && params[:case] == "Channel"
    q = -sign(p[3]) * params[:Lz] + p[3]
    vv3 = findall(x->isapprox(x, q; atol=atol) , params[:restart_df].Points_2) 
  end



  vv12 = vv1[findall(in(vv2),vv1)]
  idx = vv12[findall(in(vv3),vv12)]  
  
  if isempty(idx)
    idx = find_idx(p, params; atol = atol*10)
  end
  
  return idx[1]
end

function uh(p::VectorValue{2, Float64}, params::Dict{Symbol, Any}, idx::Int)

  VectorValue(params[:restart_df].uh_0[idx][1]  .* 0.5, params[:restart_df].uh_1[idx][1])
end


function uh(p::VectorValue{3, Float64}, params::Dict{Symbol, Any}, idx::Int)
  VectorValue(params[:restart_df].uh_0[idx][1], params[:restart_df].uh_1[idx][1], params[:restart_df].uh_2[idx][1])
end

"""
    uh_restart(p, params::Dict{Symbol, Any})

For a given point `p` it calles `find_idx` which provide the line of the `csv` file corresponding to that point. Then, it calles `uh` which provide the `VectorValue` of the velocity at that point.
"""
function uh_restart(p, params::Dict{Symbol, Any})
  
  idx = find_idx(p, params)
  return uh(p, params, idx)

end

"""
    ph_restart(p, params::Dict{Symbol, Any})

For a given point `p` it calles `find_idx` which provide the line of the `csv` file corresponding to that point. Then, it calles `ph` which provide the scalar pressure value at that point.
"""
function ph_restart(p, params::Dict{Symbol, Any})
  idx = find_idx(p, params)
  ph = params[:restart_df].ph[idx][1]
  return ph
end

"""
    restart_uh_field(params::Dict{Symbol, Any})

It provides a suitable function which gives for each point the specified velocity in `restart_file`. It is used as initial condition for restarting a simulation at a specific time step.
"""
function restart_uh_field(params::Dict{Symbol, Any})
  println("Restarting uh0 ...")
  # u0(t::Real) = x -> u0(x, t::Real)
  u0(x) = uh_restart(x, params)

  return u0

end

"""
    restart_ph_field(params::Dict{Symbol, Any})

It provides a suitable function which gives for each point the specified pressure in `restart_file`. It is used as initial condition for restarting a simulation at a specific time step.
"""
function restart_ph_field(params::Dict{Symbol, Any})
  println("Restarting ph0 ...")
  
  init_pres = DataFrames.haskey(params[:restart_df],:ph)

  # p0(x, t::Real) = (init_pres) ?   ph_restart(x, params) : 0.0
  # p0(t::Real) = x -> p0(x, t::Real)
  p0(x) = (init_pres) ?   ph_restart(x, params) : 0.0

  return p0

end
