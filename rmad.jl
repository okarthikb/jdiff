mutable struct Node
  value::Float64
  grad::Float64
  children::Vector{Node}
  backward_fn::Union{Nothing, Function}

  Node(value::Float64, children=Node[], backward_fn=nothing) = new(value, 0.0, children, backward_fn)
end

function Base.:+(a::Node, b::Node)
  out = Node(a.value + b.value, [a, b])
  out.backward_fn = function ()
    a.grad += out.grad
    b.grad += out.grad
  end
  return out
end

function Base.:-(a::Node, b::Node)
  out = Node(a.value - b.value, [a, b])
  out.backward_fn = function ()
    a.grad += out.grad
    b.grad -= out.grad
  end
  return out
end

function Base.:*(a::Node, b::Node)
  out = Node(a.value * b.value, [a, b])
  out.backward_fn = function ()
    a.grad += out.grad * b.value
    b.grad += out.grad * a.value
  end
  return out
end

function Base.:/(a::Node, b::Node)
  out = Node(a.value / b.value, [a, b])
  out.backward_fn = function ()
    a.grad += out.grad / b.value
    b.grad -= out.grad * a.value / b.value^2
  end
  return out
end

function backward(node::Node)
  nodes = Node[]
  visited = Set{Node}()

  function dfs(node)
    push!(visited, node)
    for child in node.children
      if !(child in visited)
        dfs(child)
      end
    end
    push!(nodes, node)
  end

  dfs(node)
  node.grad = 1.0
  for node in reverse(nodes)
    if node.backward_fn != nothing
      node.backward_fn()
    end
  end
end
