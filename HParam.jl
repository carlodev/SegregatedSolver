
function h_param(Ω::GridapDistributed.DistributedTriangulation, D::Int64)
    h = map_parts(Ω.trians) do trian
        h_param(trian, D)
    end
    h = CellData.CellField(h, Ω)
    h
end


function h_param(Ω::Triangulation, D::Int64)
    h = lazy_map(h -> h^(1 / D), get_cell_measure(Ω))

    h
end
