
function h_param(Ω::GridapDistributed.DistributedTriangulation, D::Int64)
    h = map(Ω.trians) do trian
        h_param(trian, D)
    end
    h = CellData.CellField(h, Ω)
    h
end


function h_param(Ω::Triangulation, D::Int64)
    h = lazy_map(h -> h^(1 / D), get_cell_measure(Ω))

    h
end



function G_params(Ω::GridapDistributed.DistributedTriangulation, params)
    @time G, GG, gg = map(Ω.trians) do trian
  
      G_params(trian, params)
    end
  
    G = CellData.CellField(G, Ω)
    GG = CellData.CellField(GG, Ω)
    gg = CellData.CellField(gg, Ω)
    G, GG, gg
  end
  
  function G_params(trian::Gridap.Geometry.BodyFittedTriangulation, params) #trian == Ω
    D = params[:D]
  
    ξₖ = get_cell_map(trian)
    Jt = lazy_map(Broadcasting(∇), ξₖ)
    inv_Jt = lazy_map(Operation(inv), Jt)
  
    if D == 2
      eval_point = Point(0.5, 0.5)
    else
      eval_point = Point(0.5, 0.5, 0.5)
    end
  
    d = lazy_map(evaluate, inv_Jt, Fill(eval_point, num_cells(trian)))
    dtrans = lazy_map(Broadcasting(transpose), d)
    G = lazy_map(Broadcasting(⋅), d, dtrans)
    GG = lazy_map(Broadcasting(⊙), G, G)
  
  
    function gg_operation(d)
  
      if D == 2
  
        return (d[1] + d[3])^2 + (d[2] + d[4])^2
  
  
      elseif D == 3
  
        return (d[1] + d[4] + d[7])^2 + (d[2] + d[5] + d[8])^2 + (d[3] + d[6] + d[9])^2
  
      end
  
    end
  
    gg = lazy_map(Broadcasting(gg_operation), d)
  
  
    G, GG, gg
  end #end G_params - Triangulation
  